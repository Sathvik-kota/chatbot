# textgen_service/main.py
"""
Robust TextGen (RAG) service tuned to:
  /content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv

Features:
 - Safe imports (works across several langchain / chroma versions)
 - /upload-csv to upload CSV (optional)
 - /force-ingest to ingest CSV into Chroma (batching + robust add methods)
 - /status to inspect readiness
 - /generate-text to run the RAG chain (if initialized)
"""

import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional

# -------------------------
# Try imports (robust)
# -------------------------
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
        def split_text(self, text):
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

# EXACT path you reported
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"

# Also look for these common names/locations
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

# -------------------------
# Globals
# -------------------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "rag_chain": None
}

# -------------------------
# Helper functions
# -------------------------
def find_csv_path(basename: str = DEFAULT_CSV_BASENAME) -> Optional[str]:
    """Look for CSV in several likely locations (including the exact path you gave)."""
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

def _try_add_texts(vs, texts):
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

def try_init_llm_and_rag(vs):
    """Create LLM (HuggingFaceHub) and RetrievalQA. Returns rag or None."""
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
            return rag
        except Exception:
            traceback.print_exc()
            return None

    rag = make_rag(PRIMARY_REPO_ID, "text-generation")
    if rag is not None:
        return rag
    return make_rag(FALLBACK_SMALL_REPO_ID, "text2text-generation")

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

    # try LLM/RAG initialization
    ml_models["rag_chain"] = None
    if ml_models.get("vector_store") is not None and HF_TOKEN:
        rag = try_init_llm_and_rag(ml_models["vector_store"])
        if rag:
            ml_models["rag_chain"] = rag
            print("[LIFESPAN] RAG chain initialized.")
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
        out = None
        # try run -> invoke -> fallback
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
