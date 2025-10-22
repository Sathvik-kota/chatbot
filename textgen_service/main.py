# textgen_service/main.py
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Try multiple import locations for langchain community modules / text splitters
try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.llms import HuggingFaceHub
except Exception:
    HuggingFaceHub = None

# Try vectorstore import (langchain_community wrapper)
try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except Exception:
    LC_Chroma = None

# Try text splitter imports from possible packages
RecursiveCharacterTextSplitter = None
for mod in (
    "langchain_community.text_splitter",
    "langchain_text_splitters",
    "langchain.text_splitter",
):
    try:
        m = __import__(mod, fromlist=["RecursiveCharacterTextSplitter"])
        RecursiveCharacterTextSplitter = getattr(m, "RecursiveCharacterTextSplitter")
        break
    except Exception:
        RecursiveCharacterTextSplitter = None

# Fallback to a simple naive splitter if none available
if RecursiveCharacterTextSplitter is None:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            # naive chunker
            chunks = []
            i = 0
            n = len(text)
            while i < n:
                chunks.append(text[i:i+self.chunk_size])
                i += self.chunk_size - self.chunk_overlap
            return chunks

# RetrievalQA import (try common locations)
try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

# chromadb direct client (we'll use PersistentClient if available)
try:
    import chromadb
    from chromadb.config import Settings as ChromadbSettings  # may or may not exist
except Exception:
    chromadb = None

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
# update this filename to the one you placed in documents/
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "ai_cybersecurity_dataset-sampled-5k.csv")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

# ---------- Globals ----------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "db_client": None,
    "rag_chain": None
}

# ---------- Helpers ----------
def load_csv_report():
    if not os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"[DATA] Knowledge CSV not found at {KNOWLEDGE_FILE_PATH}")
        return None
    try:
        df = pd.read_csv(KNOWLEDGE_FILE_PATH)
        df.columns = df.columns.str.strip()
        print(f"[DATA] Loaded CSV rows={len(df)} columns={list(df.columns)[:20]}")
        return df
    except Exception as e:
        print("[DATA] Failed to load CSV:", e)
        traceback.print_exc()
        return None

def create_embedding_function():
    if SentenceTransformerEmbeddings is None:
        print("[EMB] SentenceTransformerEmbeddings import not found.")
        return None
    try:
        emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("[EMB] Embedding function initialized.")
        return emb
    except Exception as e:
        print("[EMB] Failed to init embedding function:", e)
        traceback.print_exc()
        return None

def init_chroma_persistent(embedding_function):
    """
    Try several ways to initialize a Chroma vectorstore that's compatible with:
     - langchain_community.vectorstores.Chroma (wrapper)
     - direct Chroma wrapper with persist_directory
     - fallback to in-memory
    Returns tuple (vector_store, client_description)
    """
    # 1) Try langchain_community wrapper that may accept persist_directory
    if LC_Chroma is not None:
        try:
            # try with persist_directory kw first (newer wrappers)
            vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
            print("[CHROMA] Initialized langchain_community.Chroma with persist_directory")
            return vs, "langchain_community.Chroma(persist_directory)"
        except Exception:
            try:
                # older signature: client=chromadb.PersistentClient(...)
                if chromadb is not None:
                    try:
                        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                    except Exception:
                        # older API
                        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                else:
                    client = None
                vs = LC_Chroma(client=client, collection_name="rag_collection", embedding_function=embedding_function)
                print("[CHROMA] Initialized langchain_community.Chroma with client")
                return vs, "langchain_community.Chroma(client)"
            except Exception:
                print("[CHROMA] langchain_community.Chroma initialization failed; falling through.")
                traceback.print_exc()

    # 2) Try to use chromadb directly + basic wrapper from langchain-community (if available)
    try:
        if chromadb is not None:
            try:
                client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                print("[CHROMA] chromadb.PersistentClient created")
            except Exception:
                # older API name
                client = chromadb.Client(path=CHROMA_DB_PATH) if hasattr(chromadb, "Client") else None
            # try LC_Chroma with client if LC_Chroma exists
            if LC_Chroma is not None and client is not None:
                try:
                    vs = LC_Chroma(client=client, collection_name="rag_collection", embedding_function=embedding_function)
                    print("[CHROMA] LC_Chroma initialized using chromadb client")
                    return vs, "lc_chroma_with_client"
                except Exception:
                    traceback.print_exc()
    except Exception:
        traceback.print_exc()

    # 3) Try to fall back to LC_Chroma without persistence
    if LC_Chroma is not None:
        try:
            vs = LC_Chroma(embedding_function=embedding_function)
            print("[CHROMA] LC_Chroma initialized in-memory (no persist)")
            return vs, "lc_chroma_inmemory"
        except Exception:
            traceback.print_exc()

    # 4) As a last resort return None
    print("[CHROMA] All Chroma initialization attempts failed.")
    return None, None

def vectorstore_has_data(vs):
    """Try multiple methods to detect if the vectorstore already has data"""
    try:
        # many wrappers expose _collection.count()
        if hasattr(vs, "_collection") and hasattr(vs._collection, "count"):
            c = vs._collection.count()
            return c > 0, c
    except Exception:
        traceback.print_exc()
    # try wrapper API get() -> documents
    try:
        if hasattr(vs, "get"):
            res = vs.get()
            docs = res.get("documents") if isinstance(res, dict) else None
            if docs is not None:
                return len(docs) > 0, len(docs)
    except Exception:
        traceback.print_exc()
    # unknown, assume empty
    return False, 0

def ingest_dataframe_to_vs(df, vs):
    # Convert each row to a single text (column:value | ...)
    docs = df.fillna("").apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()
    # chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text("\n\n".join(docs))
    try:
        vs.add_texts(texts=chunks)
        print(f"[INGEST] Added {len(docs)} docs -> {len(chunks)} chunks to vector store")
        # some wrappers auto-persist; if they expose persist(), call it
        if hasattr(vs, "persist"):
            try:
                vs.persist()
                print("[INGEST] Called vs.persist()")
            except Exception:
                pass
        return True
    except Exception as e:
        print("[INGEST] Failed to add texts to vector store:", e)
        traceback.print_exc()
        return False

def try_create_llm_and_rag(vs):
    """
    Create a HuggingFaceHub LLM (with explicit task), then RetrievalQA.
    Tries primary then fallback.
    """
    if HuggingFaceHub is None:
        print("[LLM] HuggingFaceHub import not available.")
        return None

    if RetrievalQA is None:
        print("[LLM] RetrievalQA import not available.")
        return None

    if not HF_TOKEN:
        print("[LLM] HF token not set.")
        return None

    def make_rag(repo_id, task):
        try:
            llm = HuggingFaceHub(repo_id=repo_id, task=task, huggingfacehub_api_token=HF_TOKEN, model_kwargs={"temperature":0.3, "max_new_tokens":512})
            rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vs.as_retriever())
            print(f"[LLM] Created LLM+RAG for {repo_id} (task={task})")
            return rag
        except Exception as e:
            print(f"[LLM] Failed creating LLM for {repo_id} (task={task}): {e}")
            traceback.print_exc()
            return None

    rag = make_rag(PRIMARY_REPO_ID, "text-generation")
    if rag is not None:
        return rag
    print("[LLM] Primary failed; trying small fallback.")
    rag = make_rag(FALLBACK_SMALL_REPO_ID, "text2text-generation")
    return rag

# ---------- FastAPI lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] Starting up textgen service...")

    # 1) Embedding
    ml_models["embedding_function"] = create_embedding_function()

    # 2) Initialize Chroma (try several ways)
    vs, desc = init_chroma_persistent(ml_models["embedding_function"])
    if vs is not None:
        ml_models["vector_store"] = vs
        ml_models["db_client"] = getattr(vs, "_client", None) or getattr(vs, "client", None)
        print(f"[LIFESPAN] Vector store ready (init method: {desc})")
    else:
        ml_models["vector_store"] = None
        print("[LIFESPAN] Vector store not available after init attempts.")

    # 3) Ingest CSV into vector store if it's empty
    vs = ml_models.get("vector_store")
    if vs:
        try:
            has_data, count = vectorstore_has_data(vs)
            print(f"[LIFESPAN] Vector store data check -> has_data={has_data} count={count}")
            if not has_data:
                df = load_csv_report()
                if df is not None and len(df) > 0:
                    success = ingest_dataframe_to_vs(df, vs)
                    if not success:
                        print("[LIFESPAN] Ingestion failed.")
                else:
                    print("[LIFESPAN] No dataframe to ingest.")
            else:
                print("[LIFESPAN] Skipping ingestion; vector store already contains data.")
        except Exception:
            print("[LIFESPAN] Error during ingestion check/ingest.")
            traceback.print_exc()
    else:
        print("[LIFESPAN] No vector store available; skipping ingestion.")

    # 4) Initialize RAG chain
    ml_models["rag_chain"] = None
    if ml_models.get("vector_store") and HF_TOKEN:
        rag = try_create_llm_and_rag(ml_models["vector_store"])
        if rag:
            ml_models["rag_chain"] = rag
            print("[LIFESPAN] RAG chain initialized.")
        else:
            print("[LIFESPAN] RAG chain failed to initialize.")
    else:
        if not HF_TOKEN:
            print("[LIFESPAN] HUGGINGFACEHUB_API_TOKEN not set; cannot initialize LLM.")
        else:
            print("[LIFESPAN] vector_store missing; skipping RAG initialization.")

    print("[LIFESPAN] Startup complete. Status:",
          {"vector_store_ready": bool(ml_models.get("vector_store")),
           "rag_chain_ready": bool(ml_models.get("rag_chain")),
           "hf_token_set": bool(HF_TOKEN)})
    yield

    # shutdown cleanup
    ml_models.clear()
    print("[LIFESPAN] Shutdown complete.")

# ---------- FastAPI app ----------
app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Text Generation (RAG) Service running."}

@app.get("/status")
def status():
    return {
        "vector_store_ready": bool(ml_models.get("vector_store")),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN),
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not ml_models.get("rag_chain"):
        raise HTTPException(status_code=503, detail="RAG chain not available.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        # prefer .run if present, otherwise .invoke -> be robust
        rag = ml_models["rag_chain"]
        try:
            if hasattr(rag, "run"):
                out = rag.run(req.query)
            elif hasattr(rag, "invoke"):
                result = rag.invoke({"query": req.query})
                # RetrievalQA.invoke may return dict with 'result' or string
                if isinstance(result, dict):
                    out = result.get("result") or result.get("answer") or str(result)
                else:
                    out = result
            else:
                out = rag(req.query) if callable(rag) else None
        except Exception:
            # try alternative
            traceback.print_exc()
            if hasattr(rag, "invoke"):
                result = rag.invoke({"query": req.query})
                out = result.get("result") if isinstance(result, dict) else result
            else:
                raise

        return QueryResponse(answer=str(out) if out is not None else "No answer generated.")
    except Exception as e:
        print("[GEN] Generation error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
