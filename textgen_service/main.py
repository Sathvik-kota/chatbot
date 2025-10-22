# textgen_service/main.py
"""
RAG service that robustly uses huggingface_hub.InferenceClient (handles multiple client API shapes).
Place your CSV at:
  /content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv
or upload via /upload-csv then call /force-ingest.
"""

import os
import traceback
import requests
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Try InferenceClient import
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

# langchain-community embedding / chroma imports (best-effort)
try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except Exception:
    LC_Chroma = None

# Try text splitter
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

# ---------------- Config ----------------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
HARDCODED_CSV_PATH = os.path.join(ROOT_DIR, "ai_cybersecurity_dataset-sampled-5k.csv")
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_REPO_ID = "google/flan-t5-small"
TOP_K = 4

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "hf_inference_client": None,
    "hf_model_id": None,
}

# ---------------- Helpers ----------------

def find_csv_path(basename: str = DEFAULT_CSV_BASENAME) -> Optional[str]:
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
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    except Exception:
        pass
    if LC_Chroma is None:
        print("[CHROMA] langchain_community.Chroma import not available.")
        return None
    try:
        vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        print("[CHROMA] Initialized with persist_directory.")
        return vs
    except Exception:
        traceback.print_exc()
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
    try:
        vs = LC_Chroma(embedding_function=embedding_function)
        print("[CHROMA] Initialized in-memory Chroma.")
        return vs
    except Exception:
        traceback.print_exc()
    return None

def vs_count_estimate(vs) -> int:
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
    try:
        if hasattr(vs, "add_texts"):
            vs.add_texts(texts=texts)
            return True, "add_texts"
    except Exception:
        traceback.print_exc()
    try:
        if hasattr(vs, "add_documents"):
            docs = [{"page_content": t, "metadata": {}} for t in texts]
            vs.add_documents(docs)
            return True, "add_documents"
    except Exception:
        traceback.print_exc()
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
    try:
        for t in texts:
            if hasattr(vs, "add_texts"):
                vs.add_texts(texts=[t])
        return True, "iterative_add_texts"
    except Exception:
        traceback.print_exc()
    return False, None

def ingest_dataframe(df: pd.DataFrame, vs, batch_docs: int = 500):
    if df is None or vs is None:
        return False, "no_df_or_vs"
    docs = df.fillna("").apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()
    total_added = 0
    for i in range(0, len(docs), batch_docs):
        batch = docs[i:i+batch_docs]
        ok, method = _try_add_texts(vs, batch)
        if not ok:
            return False, f"batch_failed_at_{i}"
        total_added += len(batch)
    try:
        if hasattr(vs, "persist"):
            vs.persist()
    except Exception:
        traceback.print_exc()
    return True, {"method": method, "docs_added": total_added}

def check_inference_endpoint(repo_id: str, token: str, timeout: int = 8) -> bool:
    if not token:
        return False
    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.head(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return True
        if r.status_code in (401, 403, 404):
            return False
        r2 = requests.get(url, headers=headers, timeout=timeout)
        return r2.status_code == 200
    except Exception:
        return False

def build_prompt_with_context(query: str, docs: list) -> str:
    ctx_items = []
    for d in docs:
        if hasattr(d, "page_content"):
            ctx_items.append(d.page_content.strip())
        elif isinstance(d, dict) and "page_content" in d:
            ctx_items.append(str(d["page_content"]).strip())
        else:
            ctx_items.append(str(d).strip())
    context = "\n\n---\n\n".join(ctx_items) if ctx_items else ""
    prompt = "Use the following context to answer the question concisely.\n\n"
    if context:
        prompt += f"CONTEXT:\n{context}\n\n"
    prompt += f"QUESTION:\n{query}\n\nAnswer:"
    return prompt

# ---------- Robust inference call ----------
def generate_with_inference(client, model_id: str, retriever, query: str, top_k: int = TOP_K, max_new_tokens: int = 256):
    # Retrieve documents
    docs = []
    try:
        retr = None
        if hasattr(retriever, "get_relevant_documents"):
            retr = retriever
        elif hasattr(retriever, "as_retriever"):
            retr = retriever.as_retriever()
        if retr is not None:
            if hasattr(retr, "get_relevant_documents"):
                docs = retr.get_relevant_documents(query)
            elif hasattr(retr, "get_documents"):
                docs = retr.get_documents(query)
    except Exception:
        traceback.print_exc()
        docs = []

    docs = docs[:top_k] if docs else []

    prompt = build_prompt_with_context(query, docs)

    # Try multiple client interfaces in order
    last_exc = None
    tried = []
    # 1) text_generation (common)
    try:
        if hasattr(client, "text_generation"):
            tried.append("text_generation")
            gen = client.text_generation(prompt, max_new_tokens=max_new_tokens)
            return _parse_inference_response(gen, "text_generation")
    except Exception as e:
        last_exc = e
        traceback.print_exc()

    # 2) generate
    try:
        if hasattr(client, "generate"):
            tried.append("generate")
            gen = client.generate(prompt, max_new_tokens=max_new_tokens)
            return _parse_inference_response(gen, "generate")
    except Exception as e:
        last_exc = e
        traceback.print_exc()

    # 3) __call__ style: client(inputs=..., parameters={})
    try:
        if callable(client):
            tried.append("__call__")
            try:
                gen = client(inputs=prompt, parameters={"max_new_tokens": max_new_tokens})
            except TypeError:
                gen = client(prompt, parameters={"max_new_tokens": max_new_tokens})
            return _parse_inference_response(gen, "__call__")
    except Exception as e:
        last_exc = e
        traceback.print_exc()

    # 4) try InferenceClient.text_generation_stream or text_generation if available (some versions)
    try:
        for alt in ("text_generation_stream", "text_generation_streaming"):
            if hasattr(client, alt):
                tried.append(alt)
                gen = getattr(client, alt)(prompt, max_new_tokens=max_new_tokens)
                return _parse_inference_response(gen, alt)
    except Exception as e:
        last_exc = e
        traceback.print_exc()

    # 5) raw HTTP fallback to Hugging Face Inference endpoint
    try:
        tried.append("raw_http")
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        gen = r.json()
        return _parse_inference_response(gen, "raw_http")
    except Exception as e:
        last_exc = e
        traceback.print_exc()

    raise RuntimeError(f"Inference failed. Tried methods: {tried}. Last error: {last_exc}")

def _parse_inference_response(gen, method_name: str) -> str:
    """
    Best-effort parsing of many possible shapes returned by different inference client versions.
    """
    try:
        # list of dicts e.g. [{"generated_text": "..."}]
        if isinstance(gen, list):
            first = gen[0]
            if isinstance(first, dict):
                for key in ("generated_text", "generated_texts", "text", "output", "result"):
                    if key in first:
                        val = first[key]
                        if isinstance(val, list):
                            return " ".join(val)
                        return str(val)
                # fallback: join stringified dict
                return str(first)
            else:
                return str(first)

        # dict
        if isinstance(gen, dict):
            for key in ("generated_text", "generated_texts", "text", "output", "result"):
                if key in gen:
                    val = gen[key]
                    if isinstance(val, list):
                        return " ".join(val)
                    return str(val)
            # Some responses nest outputs
            if "outputs" in gen and isinstance(gen["outputs"], list) and gen["outputs"]:
                out0 = gen["outputs"][0]
                if isinstance(out0, dict) and "generated_text" in out0:
                    return str(out0["generated_text"])
            return str(gen)

        # object with attributes
        if hasattr(gen, "generated_text"):
            return getattr(gen, "generated_text")
        if hasattr(gen, "content"):
            return getattr(gen, "content")
        if hasattr(gen, "text"):
            return getattr(gen, "text")

        return str(gen)
    except Exception:
        traceback.print_exc()
        return str(gen)

def try_init_inference_client(vs):
    if InferenceClient is None:
        print("[LLM] huggingface_hub.InferenceClient not available.")
        return None, None
    if not HF_TOKEN:
        print("[LLM] HF token not set.")
        return None, None

    # primary
    if check_inference_endpoint(PRIMARY_REPO_ID, HF_TOKEN):
        try:
            client = InferenceClient(model=PRIMARY_REPO_ID, token=HF_TOKEN)
            return client, PRIMARY_REPO_ID
        except Exception:
            traceback.print_exc()

    # fallback
    if check_inference_endpoint(FALLBACK_REPO_ID, HF_TOKEN):
        try:
            client = InferenceClient(model=FALLBACK_REPO_ID, token=HF_TOKEN)
            return client, FALLBACK_REPO_ID
        except Exception:
            traceback.print_exc()

    return None, None

# ---------------- FastAPI lifecycle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting up...")
    ml_models["embedding_function"] = create_embedding_function()
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]
    print(f"[LIFESPAN] vector_store object: {type(vs).__name__ if vs is not None else None}")

    csv_path = find_csv_path()
    if csv_path:
        print(f"[LIFESPAN] Found CSV at: {csv_path}")
    else:
        print("[LIFESPAN] No sampled CSV found automatically.")

    if vs is not None and csv_path:
        try:
            count = vs_count_estimate(vs)
            print(f"[LIFESPAN] vector store estimated count: {count}")
            if count == 0:
                print("[LIFESPAN] Auto-ingesting CSV (may take time)...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None and len(df) > 0:
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] ingest result:", ok, info)
        except Exception:
            traceback.print_exc()
    else:
        print("[LIFESPAN] skipping auto-ingest (no vs or no csv)")

    ml_models["hf_inference_client"], ml_models["hf_model_id"] = try_init_inference_client(vs)
    print("[LIFESPAN] inference client:", bool(ml_models["hf_inference_client"]), "model:", ml_models.get("hf_model_id"))
    print("[LIFESPAN] startup complete")
    yield
    ml_models.clear()
    print("[LIFESPAN] shutdown complete")

# ---------------- App ----------------

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
    count = vs_count_estimate(vs) if vs is not None else 0
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": count > 0,
        "vector_store_count": int(count),
        "inference_client_ready": ml_models.get("hf_inference_client") is not None,
        "hf_model_id": ml_models.get("hf_model_id"),
        "hf_token_set": bool(HF_TOKEN),
        "csv_found": bool(find_csv_path())
    }

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), save_name: Optional[str] = Form(None)):
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
    vs = ml_models.get("vector_store")
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not available.")
    csv_path = find_csv_path()
    if csv_path is None:
        raise HTTPException(status_code=404, detail="CSV not found.")
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
    client = ml_models.get("hf_inference_client")
    model_id = ml_models.get("hf_model_id")
    vs = ml_models.get("vector_store")
    if client is None or model_id is None:
        raise HTTPException(status_code=503, detail="Inference client not available. Check HF token and model availability.")
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not available.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        retriever = None
        try:
            retriever = vs.as_retriever()
        except Exception:
            retriever = vs
        answer = generate_with_inference(client, model_id, retriever, req.query, top_k=TOP_K, max_new_tokens=256)
        return QueryResponse(answer=str(answer))
    except Exception as e:
        traceback.print_exc()
        # If HTTPError from HF, include its text
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    print("Run: uvicorn textgen_service.main:app --host 0.0.0.0 --port 8002")
