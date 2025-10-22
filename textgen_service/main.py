"""
Robust RAG service with multi-path LLM support:

Priority:
  1) huggingface_hub.InferenceClient -> preferred (Inference API)
  2) langchain_community.llms.HuggingFaceHub -> fallback LLM via HF Hub (must have HF token)
  3) If neither available, the server will report the missing pieces clearly.

Make sure your CSV is at:
  /content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv
or upload via /upload-csv endpoint and call /force-ingest.
"""
import os
import traceback
import requests
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Try imports (be tolerant across environments) ---
try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except Exception:
    LC_Chroma = None

try:
    from langchain_community.llms import HuggingFaceHub as LC_HuggingFaceHub
except Exception:
    LC_HuggingFaceHub = None

# --- NEW IMPORT ---
try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None
# --------------------

# RetrievalQA import (try usual places)
try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

# Text splitter try/fallback
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
# Exact file path you reported
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"    # may be gated / not hosted for inference
# *** UPDATED FALLBACK_REPO_ID ***
# All instruction-tuned models are 404.
# Reverting to the *only* model that was accessible: google-t5/t5-small
FALLBACK_REPO_ID = "google-t5/t5-small"

TOP_K = 4

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "hf_inference_client": None,
    "hf_model_id": None,
    "rag_chain": None,
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
    # Try persist_directory constructor (common)
    try:
        vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        print("[CHROMA] Initialized with persist_directory.")
        return vs
    except Exception:
        traceback.print_exc()
    # Try client-based init with chromadb
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
    # in-memory fallback
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
    # multiple strategies to add texts (various versions of wrappers)
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
        # treat 401/403/404 as unavailable
        if r.status_code in (401, 403, 404):
            return False
        r2 = requests.get(url, headers=headers, timeout=timeout)
        return r2.status_code == 200
    except Exception:
        return False

def build_prompt_with_context(query: str, docs: list, model_id: str = "") -> str:
    """Builds a prompt, optimizing the format for the specific model_id."""
    ctx_items = []
    for d in docs:
        if hasattr(d, "page_content"):
            ctx_items.append(d.page_content.strip())
        elif isinstance(d, dict) and "page_content" in d:
            ctx_items.append(str(d["page_content"]).strip())
        else:
            ctx_items.append(str(d).strip())
    context = "\n\n---\n\n".join(ctx_items) if ctx_items else "No context provided."

    # --- UPDATED: Select prompt format based on model ---
    # T5 models prefer a "task" format
    if "t5" in model_id.lower():
        # --- UPDATED: Use a summarization-style prefix to force a different task ---
        # This is a bit of a trick to stop it from just repeating the context.
        return f"""summarize: {context}

Given the context above, {query}"""

    # Default to instruction format for models like Mixtral
    prompt = "Use the following context to answer the question concisely.\n\n"
    if context:
        prompt += f"CONTEXT:\n{context}\n\n"
    prompt += f"QUESTION:\n{query}\n\nAnswer:"
    return prompt

# ---------- Inference helpers (robust) ----------
def _parse_inference_response(gen, method_name: str) -> str:
    try:
        if isinstance(gen, list):
            first = gen[0]
            if isinstance(first, dict):
                # --- UPDATED: Added 'translation_text' for T5 models ---
                for key in ("generated_text", "generated_texts", "text", "output", "result", "translation_text"):
                    if key in first:
                        val = first[key]
                        if isinstance(val, list):
                            return " ".join(val)
                        return str(val)
                return str(first)
            else:
                return str(first)
        if isinstance(gen, dict):
            # --- UPDATED: Added 'translation_text' for T5 models ---
            for key in ("generated_text", "generated_texts", "text", "output", "result", "translation_text"):
                if key in gen:
                    val = gen[key]
                    if isinstance(val, list):
                        return " ".join(val)
                    return str(val)
            if "outputs" in gen and isinstance(gen["outputs"], list) and gen["outputs"]:
                out0 = gen["outputs"][0]
                if isinstance(out0, dict) and "generated_text" in out0:
                    return str(out0["generated_text"])
            return str(gen)
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

def generate_with_inference(client, model_id: str, retriever, query: str, top_k: int = TOP_K, max_new_tokens: int = 256):
    # retrieve
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
    # --- UPDATED: Pass model_id to build the correct prompt ---
    prompt = build_prompt_with_context(query, docs, model_id)

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

    # 3) callable client (some versions)
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

    # 4) streaming or alternate names
    try:
        for alt in ("text_generation_stream", "text_generation_streaming"):
            if hasattr(client, alt):
                tried.append(alt)
                gen = getattr(client, alt)(prompt, max_new_tokens=max_new_tokens)
                return _parse_inference_response(gen, alt)
    except Exception as e:
        last_exc = e
        traceback.print_exc()

    # 5) raw HTTP fallback
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

def try_init_inference_client(vs):
    # Try to initialize huggingface_hub.InferenceClient for primary, then fallback model
    if InferenceClient is None:
        print("[LLM] InferenceClient not installed.")
        return None, None
    if not HF_TOKEN:
        print("[LLM] HF token not set.")
        return None, None
    # primary
    if check_inference_endpoint(PRIMARY_REPO_ID, HF_TOKEN):
        try:
            client = InferenceClient(model=PRIMARY_REPO_ID, token=HF_TOKEN)
            print(f"[LLM] InferenceClient initialized for {PRIMARY_REPO_ID}")
            return client, PRIMARY_REPO_ID
        except Exception:
            traceback.print_exc()
    # fallback public
    if check_inference_endpoint(FALLBACK_REPO_ID, HF_TOKEN):
        try:
            client = InferenceClient(model=FALLBACK_REPO_ID, token=HF_TOKEN)
            print(f"[LLM] InferenceClient initialized for fallback {FALLBACK_REPO_ID}")
            return client, FALLBACK_REPO_ID
        except Exception:
            traceback.print_exc()
    print("[LLM] No inference endpoints available for primary/fallback.")
    return None, None

def try_create_langchain_llm_and_rag(vs):
    # Try to create a langchain RetrievalQA using HuggingFaceHub (community) as a fallback
    if LC_HuggingFaceHub is None:
        print("[LLM] langchain_community.HuggingFaceHub not available.")
        return None
    if not HF_TOKEN:
        print("[LLM] HF token not set for HuggingFaceHub.")
        return None
    try:
        # use a small public model as fallback; specify task to satisfy the pydantic validation
        print(f"[LLM] Trying HuggingFaceHub with {FALLBACK_REPO_ID}")
        # --- UPDATED: Switched task back to T5's task ---
        llm = LC_HuggingFaceHub(repo_id=FALLBACK_REPO_ID, task="text2text-generation", huggingfacehub_api_token=HF_TOKEN, model_kwargs={"temperature":0.3, "max_new_tokens":512})

        # --- UPDATED: Dynamic prompt template based on the fallback model ---
        # This fixes a bug where the T5 prompt was hard-coded.
        template = ""
        if "t5" in FALLBACK_REPO_ID.lower():
            template = """summarize: {context}

Given the context above, {question}"""
        else:
            # Default instruction prompt for models like GPT-2, BLOOM, Falcon
            template = """Use the following context to answer the question concisely.

CONTEXT:
{context}

QUESTION:
{question}

Answer:"""

        prompt = None
        if PromptTemplate is not None:
            try:
                prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                print(f"[LLM] Created custom prompt template for {FALLBACK_REPO_ID}.")
            except Exception:
                traceback.print_exc()
                prompt = None
        # --------------------------------------------------------------------

        if RetrievalQA is not None:
            # --- UPDATED: Pass the custom prompt if it was created ---
            chain_type_kwargs = {}
            if prompt is not None:
                chain_type_kwargs["prompt"] = prompt

            rag = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vs.as_retriever(),
                chain_type_kwargs=chain_type_kwargs
            )
            # ----------------------------------------------------------
            print("[LLM] RetrievalQA (HuggingFaceHub) created.")
            return rag
        else:
            print("[LLM] RetrievalQA not available; cannot create chain.")
            return None
    except Exception:
        traceback.print_exc()
        return None

# ---------------- FastAPI lifecycle ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting up...")
    # embedding
    ml_models["embedding_function"] = create_embedding_function()
    # chroma
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]
    print(f"[LIFESPAN] vector_store object: {type(vs).__name__ if vs is not None else None}")

    # find CSV and auto-ingest if empty
    csv_path = find_csv_path()
    if csv_path:
        print(f"[LIFESPAN] Found CSV at: {csv_path}")
    else:
        print("[LIFESPAN] No sampled CSV found automatically.")

    if vs is not None and csv_path:
        try:
            cnt = vs_count_estimate(vs)
            print(f"[LIFESPAN] vector store estimated count: {cnt}")
            if cnt == 0:
                print("[LIFESPAN] Auto-ingesting CSV (may take time)...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None and len(df) > 0:
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] ingest result:", ok, info)
        except Exception:
            traceback.print_exc()
    else:
        print("[LIFESPAN] Skipping auto-ingest (no vs or no csv)")

    # Try inference client (primary -> fallback)
    client, model_id = try_init_inference_client(vs)
    ml_models["hf_inference_client"] = client
    ml_models["hf_model_id"] = model_id

    # If inference client unavailable, try langchain HuggingFaceHub fallback to create RetrievalQA chain
    if ml_models["hf_inference_client"] is None and vs is not None:
        rag = try_create_langchain_llm_and_rag(vs)
        ml_models["rag_chain"] = rag
        if rag is not None:
            print("[LIFESPAN] Using RetrievalQA via HuggingFaceHub (fallback).")
    else:
        print("[LIFESPAN] Using InferenceClient for hosted inference (preferred).")

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs is not None else 0,
        "inference_client_ready": ml_models.get("hf_inference_client") is not None,
        "hf_model_id": ml_models.get("hf_model_id"),
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "hf_token_set": bool(HF_TOKEN),
        "csv_found": bool(csv_path)
    })
    yield
    ml_models.clear()
    print("[LIFESPAN] shutdown complete.")

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
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
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
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    # Prefer InferenceClient path
    client = ml_models.get("hf_inference_client")
    model_id = ml_models.get("hf_model_id")
    vs = ml_models.get("vector_store")
    # If inference client available, use it (preferred)
    if client is not None and model_id is not None:
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
            raise HTTPException(status_code=500, detail=f"Generation failed (inference client): {e}")
    # Else, use a local RetrievalQA chain via HuggingFaceHub (if present)
    rag = ml_models.get("rag_chain")
    if rag is not None:
        try:
            # Many RetrievalQA interfaces: prefer .run, then .invoke
            if hasattr(rag, "run"):
                out = rag.run(req.query)
                return QueryResponse(answer=str(out))
            elif hasattr(rag, "invoke"):
                res = rag.invoke({"query": req.query})
                return QueryResponse(answer=str(res.get("result") if isinstance(res, dict) else res))
            else:
                out = rag(req.GagQueryRequest) if callable(rag) else None
                return QueryResponse(answer=str(out))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Generation failed (HuggingFaceHub fallback): {e}")
    # Nothing available
    raise HTTPException(status_code=503, detail="No inference client or fallback LLM available. Check HF token and model access.")

if __name__ == "__main__":
    print("Run: uvicorn textgen_service.main:app --host 0.0.0.0 --port 8002")




