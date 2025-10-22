# textgen_service/main.py
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# try imports (be robust to different langchain/chroma versions)
try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.llms import HuggingFaceHub
except Exception:
    HuggingFaceHub = None

try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except Exception:
    LC_Chroma = None

# Try different text splitter locations
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

# fallback naive splitter
if RecursiveCharacterTextSplitter is None:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            chunks = []
            i, n = 0, len(text)
            while i < n:
                chunks.append(text[i:i+self.chunk_size])
                i += max(1, self.chunk_size - self.chunk_overlap)
            return chunks

# RetrievalQA import (try common locations)
try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

# chromadb import (optional)
try:
    import chromadb
except Exception:
    chromadb = None

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
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
    # Try langchain_community.Chroma with persist_directory
    if LC_Chroma is not None:
        try:
            vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
            print("[CHROMA] Initialized langchain_community.Chroma with persist_directory")
            return vs
        except Exception:
            traceback.print_exc()
        try:
            # try client-based init if chromadb available
            if chromadb is not None:
                try:
                    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                except Exception:
                    # some versions use chromadb.Client
                    client = chromadb.Client(path=CHROMA_DB_PATH) if hasattr(chromadb, "Client") else None
            else:
                client = None
            if LC_Chroma is not None:
                vs = LC_Chroma(client=client, collection_name="rag_collection", embedding_function=embedding_function)
                print("[CHROMA] Initialized langchain_community.Chroma with client")
                return vs
        except Exception:
            traceback.print_exc()

    # fallback: LC_Chroma in-memory
    if LC_Chroma is not None:
        try:
            vs = LC_Chroma(embedding_function=embedding_function)
            print("[CHROMA] LC_Chroma initialized in-memory")
            return vs
        except Exception:
            traceback.print_exc()

    print("[CHROMA] All Chroma initialization attempts failed")
    return None

def vectorstore_has_data(vs):
    # explicit handling: return (has_data, count)
    if vs is None:
        return False, 0
    # 1) try _collection.count()
    try:
        col = getattr(vs, "_collection", None)
        if col is not None and hasattr(col, "count"):
            cnt = col.count()
            return (cnt > 0), cnt
    except Exception:
        traceback.print_exc()
    # 2) try vs.get()
    try:
        if hasattr(vs, "get"):
            res = vs.get()
            if isinstance(res, dict):
                docs = res.get("documents") or res.get("ids") or []
                return (len(docs) > 0), len(docs)
    except Exception:
        traceback.print_exc()
    # unknown -> assume empty
    return False, 0

def ingest_dataframe_to_vs(df, vs):
    if df is None or vs is None:
        return False
    docs = df.fillna("").apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text("\n\n".join(docs))
    try:
        vs.add_texts(texts=chunks)
        print(f"[INGEST] Added {len(docs)} docs -> {len(chunks)} chunks")
        # call persist if available
        if hasattr(vs, "persist"):
            try:
                vs.persist()
                print("[INGEST] Called vs.persist()")
            except Exception:
                pass
        return True
    except Exception as e:
        print("[INGEST] Failed to add texts:", e)
        traceback.print_exc()
        return False

def try_create_llm_and_rag(vs):
    if HuggingFaceHub is None:
        print("[LLM] HuggingFaceHub import not available.")
        return None
    if RetrievalQA is None:
        print("[LLM] RetrievalQA not available.")
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
    print("[LLM] Primary failed; trying fallback")
    return make_rag(FALLBACK_SMALL_REPO_ID, "text2text-generation")

# ---------- FastAPI lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] Starting up textgen service...")

    # embeddings
    ml_models["embedding_function"] = create_embedding_function()

    # init Chroma
    vs = init_chroma_persistent(ml_models["embedding_function"])
    # store the exact object even if its __bool__ might be False when empty
    ml_models["vector_store"] = vs
    ml_models["db_client"] = getattr(vs, "_client", None) or getattr(vs, "client", None)
    print(f"[LIFESPAN] Vector store object set: type={type(vs).__name__ if vs is not None else None}")

    # ingest if vectorstore exists (explicit is not None check)
    if ml_models["vector_store"] is not None:
        try:
            has_data, count = vectorstore_has_data(ml_models["vector_store"])
            print(f"[LIFESPAN] vectorstore_has_data -> has_data={has_data} count={count}")
            if not has_data:
                df = load_csv_report()
                if df is not None and len(df) > 0:
                    ok = ingest_dataframe_to_vs(df, ml_models["vector_store"])
                    if not ok:
                        print("[LIFESPAN] ingestion reported failure")
                else:
                    print("[LIFESPAN] dataframe empty or not available")
            else:
                print("[LIFESPAN] vector store already contains data; skipping ingestion")
        except Exception:
            print("[LIFESPAN] Error during ingestion")
            traceback.print_exc()
    else:
        print("[LIFESPAN] No vector store available; skipping ingestion (explicit None)")

    # create RAG chain if vector_store object exists and HF token set
    ml_models["rag_chain"] = None
    if ml_models["vector_store"] is not None and HF_TOKEN:
        rag = try_create_llm_and_rag(ml_models["vector_store"])
        if rag is not None:
            ml_models["rag_chain"] = rag
            print("[LIFESPAN] RAG chain initialized")
        else:
            print("[LIFESPAN] RAG chain failed to initialize")
    else:
        if not HF_TOKEN:
            print("[LIFESPAN] HF token not set; skipping LLM init")
        else:
            print("[LIFESPAN] vector_store is None; skipping LLM init")

    # final status: use explicit "is not None" for vector store readiness
    status = {
        "vector_store_ready": ml_models.get("vector_store") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "hf_token_set": bool(HF_TOKEN)
    }
    print("[LIFESPAN] Startup complete. Status:", status)
    yield
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
    # compute has_data lazily to avoid heavy calls during status
    vs = ml_models.get("vector_store")
    has_data, count = vectorstore_has_data(vs) if vs is not None else (False, 0)
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": has_data,
        "vector_store_count": int(count),
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "hf_token_set": bool(HF_TOKEN),
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if ml_models.get("rag_chain") is None:
        raise HTTPException(status_code=503, detail="RAG chain not available.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        rag = ml_models["rag_chain"]
        out = None
        # try various interfaces
        try:
            if hasattr(rag, "run"):
                out = rag.run(req.query)
            elif hasattr(rag, "invoke"):
                res = rag.invoke({"query": req.query})
                out = res.get("result") if isinstance(res, dict) else res
            else:
                out = rag(req.query) if callable(rag) else None
        except Exception:
            traceback.print_exc()
            if hasattr(rag, "invoke"):
                res = rag.invoke({"query": req.query})
                out = res.get("result") if isinstance(res, dict) else res
            else:
                raise
        return QueryResponse(answer=str(out) if out is not None else "No answer generated.")
    except Exception as e:
        print("[GEN] Generation error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
