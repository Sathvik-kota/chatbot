# textgen_service/main.py
"""
Robust TextGen (RAG) service tuned to:
  /content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv

This version:
 - Keeps robust/langchain-chroma import fallbacks
 - Uses huggingface_hub.InferenceClient for generation (Mistral / Mixtral)
 - Retains endpoints: /status, /upload-csv, /force-ingest, /generate-text
 - Safe, defensive code paths for many langchain / chroma versions
"""

import os
import traceback
import pandas as pd
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

# -------------------------
# HF Inference Client (for Mistral/Mixtral)
# -------------------------
from huggingface_hub import InferenceClient

# -------------------------
# Try imports (robust)
# -------------------------
try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.llms import HuggingFaceHub  # kept as optional fallback
except Exception:
    HuggingFaceHub = None

try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except Exception:
    LC_Chroma = None

# RetrievalQA
try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

# Text splitter (try common packages; fallback to simple splitter)
RecursiveCharacterTextSplitter = None
for mod in ("langchain_text_splitters", "langchain.text_splitter", "langchain_community.text_splitter"):
    try:
        m = __import__(mod, fromlist=["RecursiveCharacterTextSplitter"])
        RecursiveCharacterTextSplitter = getattr(m, "RecursiveCharacterTextSplitter")
        break
    except Exception:
        RecursiveCharacterTextSplitter = None

if RecursiveCharacterTextSplitter is None:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=100):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text: str):
            chunks = []
            i, n = 0, len(text)
            while i < n:
                chunks.append(text[i:i+self.chunk_size])
                i += max(1, self.chunk_size - self.chunk_overlap)
            return chunks

# -------------------------
# Configuration (user-specified CSV path)
# -------------------------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")

# Exact path user provided earlier
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Model to call via InferenceClient (change via env if you wish)
PRIMARY_REPO_ID = os.getenv("RAG_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

# -------------------------
# Globals
# -------------------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "rag_chain": None,   # will hold RAGWrapper instance or None
    "hf_inference_client": None
}

# -------------------------
# Helper functions
# -------------------------
def find_csv_path(basename: str = DEFAULT_CSV_BASENAME) -> Optional[str]:
    """Look for CSV in several likely locations (including the exact path)."""
    candidates = [
        HARDCODED_CSV_PATH,
        os.path.join(DOCUMENTS_DIR, basename),
        os.path.join(ROOT_DIR, basename),
        os.path.join(os.getcwd(), basename),
        basename
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def load_dataframe_from_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        print(f"[DATA] Loaded CSV: {path} rows={len(df)} cols={len(df.columns)}")
        return df
    except Exception:
        traceback.print_exc()
        return None

def create_embedding_function():
    if SentenceTransformerEmbeddings is None:
        print("[EMB] SentenceTransformerEmbeddings not available.")
        return None
    try:
        emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("[EMB] Embedding initialized.")
        return emb
    except Exception:
        traceback.print_exc()
        return None

def init_chroma(embedding_function):
    """Try several initialization patterns for langchain_community.Chroma wrappers."""
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    except Exception:
        pass

    if LC_Chroma is None:
        print("[CHROMA] langchain_community.Chroma import not available.")
        return None

    # pattern 1: persist_directory (common)
    try:
        vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        print("[CHROMA] Initialized with persist_directory.")
        return vs
    except Exception:
        traceback.print_exc()

    # pattern 2: client-based (older)
    try:
        import chromadb
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        except Exception:
            client = chromadb.Client(path=CHROMA_DB_PATH) if hasattr(chromadb, "Client") else None
        if client is not None:
            vs = LC_Chroma(client=client, collection_name="rag_collection", embedding_function=embedding_function)
            print("[CHROMA] Initialized with chromadb client.")
            return vs
    except Exception:
        traceback.print_exc()

    # fallback: in-memory
    try:
        vs = LC_Chroma(embedding_function=embedding_function)
        print("[CHROMA] Initialized in-memory Chroma.")
        return vs
    except Exception:
        traceback.print_exc()

    return None

def vs_count_estimate(vs) -> int:
    """Try multiple ways to estimate count."""
    if vs is None:
        return 0
    try:
        col = getattr(vs, "_collection", None)
        if col is not None and hasattr(col, "count"):
            return int(col.count())
    except Exception:
        traceback.print_exc()
    try:
        if hasattr(vs, "get"):
            res = vs.get()
            if isinstance(res, dict):
                docs = res.get("documents") or res.get("ids") or []
                return int(len(docs))
    except Exception:
        traceback.print_exc()
    return 0

def _try_add_texts(vs, texts: List[str]):
    """Try a few add APIs and return (ok, method)."""
    # 1) add_texts
    try:
        if hasattr(vs, "add_texts"):
            vs.add_texts(texts=texts)
            return True, "add_texts"
    except Exception:
        traceback.print_exc()
    # 2) add_documents
    try:
        if hasattr(vs, "add_documents"):
            docs = [{"page_content": t, "metadata": {}} for t in texts]
            vs.add_documents(docs)
            return True, "add_documents"
    except Exception:
        traceback.print_exc()
    # 3) low-level collection
    try:
        col = getattr(vs, "_collection", None) or getattr(vs, "collection", None)
        if col is not None and hasattr(col, "add"):
            try:
                col.add(documents=texts)
            except TypeError:
                col.add(documents=texts, metadatas=[{}]*len(texts), ids=[None]*len(texts))
            return True, "_collection.add"
    except Exception:
        traceback.print_exc()
    # 4) iterative fallback
    try:
        for t in texts:
            if hasattr(vs, "add_texts"):
                vs.add_texts(texts=[t])
        return True, "iterative_add_texts"
    except Exception:
        traceback.print_exc()
    return False, None

def ingest_dataframe(df: pd.DataFrame, vs, batch_docs:int=500):
    """Ingest dataframe into vectorstore in batches. Returns (ok, info)."""
    if df is None or vs is None:
        return False, "no_df_or_vs"
    # convert rows to compact text strings
    docs = df.fillna("").apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()
    total_added = 0
    for i in range(0, len(docs), batch_docs):
        batch = docs[i:i+batch_docs]
        ok, method = _try_add_texts(vs, batch)
        if not ok:
            return False, f"batch_failed_at_{i}"
        total_added += len(batch)
    # persist if available
    try:
        if hasattr(vs, "persist"):
            vs.persist()
    except Exception:
        traceback.print_exc()
    return True, {"method": method, "docs_added": total_added}

# -------------------------
# HF InferenceClient RAG wrapper
# -------------------------
def extract_text_from_inference_response(resp: Any) -> str:
    """
    Normalize the InferenceClient.text_generation response.
    It can be dict, list, or other forms depending on hf-hub version.
    """
    try:
        if resp is None:
            return ""
        # If it's dict with 'generated_text'
        if isinstance(resp, dict):
            # some versions: {'generated_text': '...'}
            if "generated_text" in resp:
                return resp["generated_text"]
            # or {'generated_texts': [...]}
            if "generated_texts" in resp and isinstance(resp["generated_texts"], list) and resp["generated_texts"]:
                return resp["generated_texts"][0]
            # fallback: str(resp)
            return str(resp)
        # If it's a list of dicts
        if isinstance(resp, list) and len(resp) > 0:
            first = resp[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            return str(first)
        # otherwise fallback to string cast
        return str(resp)
    except Exception:
        traceback.print_exc()
        return str(resp)

class RAGWrapperInference:
    """
    Simple RAG wrapper that:
     - retrieves relevant docs via vectorstore retriever (if available)
     - constructs a prompt with context + question
     - calls huggingface_hub.InferenceClient.text_generation
    """
    def __init__(self, vectorstore, hf_client: InferenceClient, max_context_chars: int = 7000):
        self.vs = vectorstore
        self.client = hf_client
        self.max_context_chars = max_context_chars

    def _retrieve_context(self, query: str, top_k: int = 4) -> str:
        # Try as_retriever API
        try:
            if hasattr(self.vs, "as_retriever"):
                retr = self.vs.as_retriever()
                # many retrievers expect .get_relevant_documents
                if hasattr(retr, "get_relevant_documents"):
                    docs = retr.get_relevant_documents(query)
                elif hasattr(retr, "get_relevant_items"):
                    docs = retr.get_relevant_items(query)
                else:
                    docs = []
            elif hasattr(self.vs, "similarity_search"):
                docs = self.vs.similarity_search(query, k=top_k)
            else:
                docs = []
            # Normalize content
            contents = []
            for d in (docs or []):
                try:
                    c = getattr(d, "page_content", None) or d.get("page_content") if isinstance(d, dict) else str(d)
                except Exception:
                    c = str(d)
                if c:
                    contents.append(c)
            ctx = "\n\n".join(contents[:top_k])
            # trim to max_context_chars (keep the end)
            if len(ctx) > self.max_context_chars:
                ctx = ctx[-self.max_context_chars:]
            return ctx
        except Exception:
            traceback.print_exc()
            return ""

    def run(self, query: str) -> str:
        context = self._retrieve_context(query)
        # Build prompt. Use a clear instruction.
        prompt = (
            "You are an assistant answering cybersecurity questions using the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer concisely and clearly:"
        )
        try:
            # Call InferenceClient.text_generation (may return dict/list)
            resp = self.client.text_generation(prompt, max_new_tokens=512)
            out = extract_text_from_inference_response(resp)
            return out
        except Exception:
            # try a lower-level call (some hf-hub versions have .pipeline or .generate)
            traceback.print_exc()
            try:
                resp2 = self.client.generate(prompt, max_new_tokens=512)  # older/newer fallback
                return extract_text_from_inference_response(resp2)
            except Exception:
                traceback.print_exc()
                raise

# -------------------------
# Try initialize HF Inference client + RAG
# This will be called from lifespan below (defensive)
# -------------------------
def try_init_inference_rag(vs) -> Optional[RAGWrapperInference]:
    if vs is None:
        print("[LLM] Vector store missing; cannot initialize RAG.")
        return None
    if not HF_TOKEN:
        print("[LLM] HF token not set; cannot initialize RAG.")
        return None

    # Try primary model via InferenceClient
    try:
        print(f"[LLM] Creating InferenceClient for model: {PRIMARY_REPO_ID}")
        client = InferenceClient(model=PRIMARY_REPO_ID, token=HF_TOKEN)
        wrapper = RAGWrapperInference(vs, client)
        print("[LLM] InferenceClient created for PRIMARY_REPO_ID.")
        ml_models["hf_inference_client"] = client
        return wrapper
    except Exception:
        traceback.print_exc()

    # Try fallback small model
    try:
        print(f"[LLM] Attempting fallback InferenceClient for model: {FALLBACK_SMALL_REPO_ID}")
        client2 = InferenceClient(model=FALLBACK_SMALL_REPO_ID, token=HF_TOKEN)
        wrapper2 = RAGWrapperInference(vs, client2)
        ml_models["hf_inference_client"] = client2
        print("[LLM] InferenceClient created for FALLBACK_SMALL_REPO_ID.")
        return wrapper2
    except Exception:
        traceback.print_exc()

    return None

# -------------------------
# FastAPI lifespan
# -------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting up...")
    # embeddings
    ml_models["embedding_function"] = create_embedding_function()
    # chroma
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]
    print(f"[LIFESPAN] vector_store object: {type(vs).__name__ if vs is not None else None}")

    # find CSV (includes the hardcoded path you provided)
    csv_path = find_csv_path()
    if csv_path is None:
        print("[LIFESPAN] No sampled CSV found automatically. Use /upload-csv or place CSV at the reported path and call /force-ingest.")
    else:
        print(f"[LIFESPAN] Found CSV at: {csv_path}")

    # ingest automatically if vector store empty and CSV found
    if vs is not None:
        try:
            count = vs_count_estimate(vs)
            print(f"[LIFESPAN] vector store estimated count: {count}")
            if count == 0 and csv_path:
                print("[LIFESPAN] Attempting automatic ingestion from CSV (this may take time)...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None and len(df) > 0:
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    if ok:
                        print(f"[LIFESPAN] Auto-ingest success: {info}")
                    else:
                        print(f"[LIFESPAN] Auto-ingest failed: {info}")
                else:
                    print("[LIFESPAN] CSV load returned no rows.")
            else:
                print("[LIFESPAN] Skipping auto-ingest (either count>0 or csv missing)")
        except Exception:
            traceback.print_exc()
    else:
        print("[LIFESPAN] No vector store; skipping ingestion.")

    # init HF InferenceClient + RAG wrapper
    ml_models["rag_chain"] = None
    if ml_models.get("vector_store") is not None and HF_TOKEN:
        rag = try_init_inference_rag(ml_models["vector_store"])
        if rag:
            ml_models["rag_chain"] = rag
            print("[LIFESPAN] RAG chain (InferenceClient-based) initialized.")
        else:
            print("[LIFESPAN] RAG chain initialization failed.")
    else:
        if not HF_TOKEN:
            print("[LIFESPAN] HF token not set; skipping LLM init.")
        else:
            print("[LIFESPAN] vector_store missing; skipping LLM init.")

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": ml_models.get("vector_store") is not None,
        "vector_store_count": vs_count_estimate(ml_models.get("vector_store")) if ml_models.get("vector_store") is not None else 0,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "hf_token_set": bool(HF_TOKEN),
        "csv_found": bool(csv_path)
    })
    yield
    ml_models.clear()
    print("[LIFESPAN] shutdown complete.")

# -------------------------
# FastAPI app & endpoints
# -------------------------
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
    vs = ml_models.get("vector_store")
    has_data = False
    count = 0
    if vs is not None:
        try:
            count = vs_count_estimate(vs)
            has_data = count > 0
        except Exception:
            traceback.print_exc()
            has_data = False
            count = 0
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": has_data,
        "vector_store_count": int(count),
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "hf_token_set": bool(HF_TOKEN),
        "csv_found": bool(find_csv_path())
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), save_name: Optional[str] = Form(None)):
    """Upload CSV to documents/ (useful if the CSV is not already placed)."""
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        filename = save_name or file.filename or DEFAULT_CSV_BASENAME
        dest = os.path.join(DOCUMENTS_DIR, filename)
        contents = await file.read()
        with open(dest, "wb") as f:
            f.write(contents)
        return {"status": "saved", "path": dest}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/force-ingest")
def force_ingest(sample_limit: Optional[int] = None, batch_size: int = 500):
    """
    Force ingestion of the CSV into the vector store.
    - sample_limit: optional int to ingest only first N rows (useful for testing)
    - batch_size: number of docs per add_texts() call
    """
    vs = ml_models.get("vector_store")
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not available.")
    csv_path = find_csv_path()
    if csv_path is None:
        raise HTTPException(status_code=404, detail=f"CSV not found. Expected one of: {HARDCODED_CSV_PATH}, {os.path.join(DOCUMENTS_DIR, DEFAULT_CSV_BASENAME)}")
    df = load_dataframe_from_csv(csv_path)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV empty or unreadable.")
    if sample_limit is not None:
        df = df.head(sample_limit)
    ok, info = ingest_dataframe(df, vs, batch_docs=batch_size)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {info}")
    final_count = vs_count_estimate(vs)
    return {"status": "ingested", "method_info": info, "final_count": int(final_count)}

@app.post("/generate-text", response_model=QueryResponse)
def generate_text(req: QueryRequest):
    if ml_models.get("rag_chain") is None:
        raise HTTPException(status_code=503, detail="RAG chain not available.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        rag = ml_models["rag_chain"]
        # Preferred interface: run(query)
        if hasattr(rag, "run"):
            out = rag.run(req.query)
        else:
            # last-resort try different interfaces
            try:
                if hasattr(rag, "invoke"):
                    res = rag.invoke({"query": req.query})
                    out = res.get("result") if isinstance(res, dict) else res
                elif callable(rag):
                    out = rag(req.query)
                else:
                    raise RuntimeError("RAG object has no runnable interface")
            except Exception:
                traceback.print_exc()
                raise
        return QueryResponse(answer=str(out) if out is not None else "No answer generated.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
