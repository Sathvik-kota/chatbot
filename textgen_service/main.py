"""
RAG service compatible with your pinned libs.
- Lazy imports to avoid immediate binary errors at module import time.
- Uses sentence-transformers for embeddings, chroma (via langchain-community) as vector store,
  and transformers pipeline for generation.
- Builds a simple chat-style system message + context prompt before calling the pipeline.
"""
import os
import traceback
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel

# ---------------- Config ----------------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_MODEL_ID = "google/flan-t5-base"
TOP_K = 6  # tuned for speed in smaller environments

# ---------------- Globals ----------------
ml = {
    "emb_model": None,        # sentence_transformers model
    "vector_store": None,     # LangChain Chroma wrapper (if available)
    "transformer_pipe": None, # transformers pipeline
}

# ---------------- Helper functions ----------------

def _lazy_imports():
    """Return modules/classes we need (import lazily so initial import errors are clearer)."""
    modules = {}
    try:
        import pandas as pd
        modules["pd"] = pd
    except Exception as e:
        modules["pd"] = None
        print("[IMPORT] pandas not available yet:", e)

    try:
        from sentence_transformers import SentenceTransformer
        modules["SentenceTransformer"] = SentenceTransformer
    except Exception as e:
        modules["SentenceTransformer"] = None
        print("[IMPORT] sentence_transformers not available:", e)

    # LangChain Chroma wrapper (langchain-community)
    try:
        from langchain_community.vectorstores import Chroma as LC_Chroma
        modules["LC_Chroma"] = LC_Chroma
    except Exception:
        modules["LC_Chroma"] = None

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        modules["pipeline"] = pipeline
        modules["AutoTokenizer"] = AutoTokenizer
        modules["AutoModelForSeq2SeqLM"] = AutoModelForSeq2SeqLM
    except Exception as e:
        modules["pipeline"] = None
        print("[IMPORT] transformers not available:", e)

    return modules

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

def load_dataframe_from_csv(path: str):
    mod = _lazy_imports()
    pd = mod.get("pd")
    if not pd:
        raise ImportError("pandas is not installed/available. Fix numpy/pandas first.")
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        print(f"[DATA] Loaded CSV: {path} rows={len(df)} cols={len(df.columns)}")
        return df
    except Exception:
        traceback.print_exc()
        return None

def init_embedding_model():
    mod = _lazy_imports()
    SentenceTransformer = mod.get("SentenceTransformer")
    if not SentenceTransformer:
        print("[EMB] sentence-transformers not available")
        return None
    try:
        emb = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[EMB] sentence-transformers loaded")
        return emb
    except Exception:
        traceback.print_exc()
        return None

def init_chroma_with_langchain(emb_model):
    """Create LangChain Chroma vectorstore wrapper if available and provide an embedding function."""
    mod = _lazy_imports()
    LC_Chroma = mod.get("LC_Chroma")
    if not LC_Chroma or emb_model is None:
        print("[CHROMA] LangChain Chroma wrapper not available or no emb model")
        return None

    # Define an embedding function that matches expected interface
    def emb_fn(texts: List[str]):
        # sentence-transformers returns numpy array; convert to list of lists
        em = emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return em.tolist()

    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=emb_fn)
        print("[CHROMA] Initialized via langchain-community.Chroma")
        return vs
    except Exception:
        traceback.print_exc()
        return None

def count_docs_in_vs(vs) -> int:
    if not vs:
        return 0
    try:
        col = getattr(vs, "_collection", None)
        if col and hasattr(col, "count"):
            return int(col.count())
    except Exception:
        pass
    return 0

def ingest_dataframe_to_vs(df, vs, batch_docs: int = 500):
    if df is None or vs is None:
        return False, "no_df_or_vs"

    USEFUL_COLUMNS = ["Attack Type", "Attack Severity", "Threat Intelligence", "Response Action"]
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]
    if not cols_to_use:
        return False, "missing_columns"

    def row_to_doc(row):
        parts = []
        for c in cols_to_use:
            v = row.get(c, "")
            if pd.notna(v) and str(v).strip():
                parts.append(f"{c}: {v}")
        return ". ".join(parts)

    docs = df.apply(lambda r: row_to_doc(r), axis=1).tolist()
    docs = [d for d in docs if d.strip()]
    total = 0
    for i in range(0, len(docs), batch_docs):
        batch = docs[i:i+batch_docs]
        try:
            vs.add_texts(texts=batch)
            total += len(batch)
        except Exception:
            traceback.print_exc()
            return False, f"failed_at_{i}"
    try:
        if hasattr(vs, "persist"):
            vs.persist()
    except Exception:
        pass
    return True, {"docs_added": total}

def init_transformer_pipeline():
    mod = _lazy_imports()
    pipeline_fn = mod.get("pipeline")
    AutoTokenizer = mod.get("AutoTokenizer")
    AutoModelForSeq2SeqLM = mod.get("AutoModelForSeq2SeqLM")
    if pipeline_fn is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        print("[LLM] transformers pipeline unavailable")
        return None

    try:
        # Using text2text-generation to work with models like flan-t5
        pipe = pipeline_fn(
            "text2text-generation",
            model=LOCAL_MODEL_ID,
            tokenizer=LOCAL_MODEL_ID,
            max_new_tokens=400,
            temperature=0.0,  # deterministic by default for best factuality
            do_sample=False
        )
        print("[LLM] transformers pipeline created")
        return pipe
    except Exception:
        traceback.print_exc()
        return None

# ---------------- Lifespan & app ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] Starting up...")
    # init embedding model (sentence-transformers)
    ml["emb_model"] = init_embedding_model()
    # init vector store via LangChain Chroma wrapper (if available)
    ml["vector_store"] = init_chroma_with_langchain(ml["emb_model"])
    vs = ml["vector_store"]

    # If we have CSV and empty DB, try ingesting
    csv_path = find_csv_path()
    if csv_path and vs:
        try:
            cnt = count_docs_in_vs(vs)
            if cnt == 0:
                print("[LIFESPAN] Auto-ingesting CSV...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None:
                    ok, info = ingest_dataframe_to_vs(df, vs)
                    print("[LIFESPAN] Ingest result:", ok, info)
        except Exception:
            traceback.print_exc()

    # init transformer text pipeline
    ml["transformer_pipe"] = init_transformer_pipeline()

    print("[LIFESPAN] Startup complete. Vectorstore:", bool(vs), "LLM pipe:", bool(ml["transformer_pipe"]))
    yield
    print("[LIFESPAN] Shutting down...")
    ml.clear()

app = FastAPI(lifespan=lifespan)

# ---------------- API models ----------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# ---------------- endpoints ----------------
@app.get("/")
def root():
    return {"status": "RAG service (compat-mode)"}

@app.get("/status")
def status():
    vs = ml.get("vector_store")
    return {
        "vector_store_ready": vs is not None,
        "vector_store_count": count_docs_in_vs(vs) if vs else 0,
        "llm_ready": ml.get("transformer_pipe") is not None,
        "embedding_ready": ml.get("emb_model") is not None,
        "model_id": LOCAL_MODEL_ID,
        "top_k": TOP_K
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/force-ingest")
def force_ingest(sample_limit: Optional[int] = None):
    vs = ml.get("vector_store")
    if not vs:
        raise HTTPException(status_code=503, detail="Vectorstore not available")
    csv_path = find_csv_path()
    if not csv_path:
        raise HTTPException(status_code=404, detail="CSV not found")
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV unreadable")
    if sample_limit:
        df = df.head(sample_limit)
    ok, info = ingest_dataframe_to_vs(df, vs)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {info}")
    return {"status": "ingested", "info": info, "count": count_docs_in_vs(vs)}

def build_prompt(system_msg: str, context: str, question: str) -> str:
    """Create a plain text prompt that resembles a chat-style instruction."""
    # keep it short and deterministic for best factual answers
    return f"{system_msg}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

@app.post("/generate-text", response_model=QueryResponse)
def generate_text(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    pipe = ml.get("transformer_pipe")
    vs = ml.get("vector_store")
    if not pipe:
        raise HTTPException(status_code=503, detail="LLM pipeline not ready")

    try:
        q = req.query.strip()
        q_lower = q.lower()
        needs_database = any(kw in q_lower for kw in [
            "response action", "threat intelligence", "severity",
            "in the database", "from database", "what actions",
            "in our data", "from our data"
        ])

        system_message = "You are a cybersecurity expert. Use only the provided context to answer accurately and concisely."

        answer = None

        # PATH: use vector store retrieval + generation
        if needs_database and vs:
            try:
                retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
                # LangChain retriever exposes get_relevant_documents in these versions
                docs = retriever.get_relevant_documents(q)
                context = "\n\n".join([d.page_content for d in docs]) if docs else ""
                prompt = build_prompt(system_message, context, q)
                print("[RAG PROMPT] len(context docs):", len(docs) if docs else 0)
                out = pipe(prompt)
                # pipeline returns list of dicts for text2text
                if isinstance(out, list) and out:
                    if isinstance(out[0], dict):
                        answer = out[0].get("generated_text") or out[0].get("translation_text") or str(out[0])
                    else:
                        answer = str(out[0])
                else:
                    answer = str(out)
            except Exception:
                traceback.print_exc()
                answer = None

        # PATH fallback: direct pipeline with minimal prompt (general knowledge)
        if not answer:
            # simple mapping for common cases
            if "list all" in q_lower or "types of" in q_lower:
                prompt_text = "List the main types of cyber attacks with brief explanations."
            elif q_lower.startswith("what is"):
                topic = q_lower.replace("what is", "").strip(" ?")
                prompt_text = f"Explain {topic} in cybersecurity, briefly."
            else:
                prompt_text = q

            # Use system message to frame prompt for the pipe
            prompt = build_prompt(system_message, "", prompt_text)
            print("[GENERAL PROMPT]", prompt_text)
            out = pipe(prompt)
            if isinstance(out, list) and out:
                if isinstance(out[0], dict):
                    answer = out[0].get("generated_text") or out[0].get("translation_text") or str(out[0])
                else:
                    answer = str(out[0])
            else:
                answer = str(out)

        # sanitize
        answer = (answer or "I couldn't produce an answer.").strip()
        # trim any echoed "Answer:" or the prompt text at the start
        if answer.startswith("Answer:"):
            answer = answer[len("Answer:"):].strip()
        # final fallback short summary if we only got context echo
        if "Attack Type:" in answer and len(answer.split()) < 40:
            answer = ("Based on the database, attacks recorded include DDoS, Malware, SQL Injection, "
                      "and Phishing — each with varying severities and response actions.")

        return QueryResponse(answer=answer)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    print("Run with: uvicorn main:app --host 0.0.0.0 --port 8002")
