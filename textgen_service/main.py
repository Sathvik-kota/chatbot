"""
Robust RAG service with LOCAL MODEL support - FINAL FIXED VERSION
Simple prompt that actually generates proper answers
"""
import os
import traceback
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Try imports (be tolerant across environments) ---
try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except Exception:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except Exception:
    LC_Chroma = None

try:
    from langchain_huggingface import HuggingFacePipeline
except Exception:
    try:
        from langchain_community.llms import HuggingFacePipeline
    except Exception:
        HuggingFacePipeline = None

try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None

try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain_community.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
    AutoModelForCausalLM = None
    pipeline = None

# ---------------- Config ----------------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Use a SMALL local model that works well for Q&A
LOCAL_MODEL_ID = "google/flan-t5-base"  # Using base for better quality

TOP_K = 1

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,
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

def create_local_llm():
    """Create a local HuggingFace Pipeline model with optimal generation settings"""
    if HuggingFacePipeline is None:
        print("[LLM] HuggingFacePipeline not available.")
        return None
    if pipeline is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        print("[LLM] transformers library not available.")
        return None

    try:
        print(f"[LLM] Loading local model: {LOCAL_MODEL_ID}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)

        # For T5/Flan-T5 models (seq2seq)
        if "t5" in LOCAL_MODEL_ID.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
            task = "text2text-generation"
        else:
            # For causal LM models
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID)
            task = "text-generation"

        # Create pipeline with OPTIMIZED generation parameters
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_length=512,  # Total max length
            min_length=40,  # Minimum length
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)

        print(f"[LLM] Local model loaded successfully: {LOCAL_MODEL_ID}")
        return llm

    except Exception as e:
        print(f"[LLM] Failed to load local model: {e}")
        traceback.print_exc()
        return None

def create_rag_chain(llm, vs):
    """Create RAG chain with SIMPLE, EFFECTIVE prompt template"""
    if llm is None or vs is None:
        print("[RAG] Cannot create chain: llm or vs is None")
        return None

    if RetrievalQA is None:
        print("[RAG] RetrievalQA not available")
        return None

    try:
        # *** FIXED: This new prompt allows the model to ignore irrelevant context ***
        template = """Use the following context ONLY if it is relevant to the question.
If the context is not relevant, answer the question using your own knowledge about cybersecurity.

Context: {context}

Question: {question}

Answer:"""

        prompt = None
        if PromptTemplate is not None:
            try:
                prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                print("[RAG] Created simple prompt template")
            except Exception:
                traceback.print_exc()

        # Create RAG chain
        chain_type_kwargs = {}
        if prompt is not None:
            chain_type_kwargs["prompt"] = prompt

        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=False
        )

        print("[RAG] RAG chain created successfully")
        return rag

    except Exception as e:
        print(f"[RAG] Failed to create chain: {e}")
        traceback.print_exc()
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting up...")

    # Embedding
    ml_models["embedding_function"] = create_embedding_function()

    # Chroma
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]
    print(f"[LIFESPAN] vector_store object: {type(vs).__name__ if vs is not None else None}")

    # Find CSV and auto-ingest if empty
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

    # Create local LLM
    print("[LIFESPAN] Loading local LLM (this may take a minute)...")
    ml_models["local_llm"] = create_local_llm()

    # Create RAG chain
    if ml_models["local_llm"] is not None and vs is not None:
        ml_models["rag_chain"] = create_rag_chain(ml_models["local_llm"], vs)

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs is not None else 0,
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "csv_found": bool(csv_path)
    })

    yield
    ml_models.clear()
    print("[LIFESPAN] shutdown complete.")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Text Generation (RAG) Service running with LOCAL models."}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    count = vs_count_estimate(vs) if vs is not None else 0
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": count > 0,
        "vector_store_count": int(count),
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "model_id": LOCAL_MODEL_ID,
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

    rag = ml_models.get("rag_chain")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG chain not available. Model may still be loading.")

    try:
        # Try different invocation methods
        if hasattr(rag, "run"):
            out = rag.run(req.query)
            return QueryResponse(answer=str(out).strip())
        elif hasattr(rag, "invoke"):
            res = rag.invoke({"query": req.query})
            answer = res.get("result") if isinstance(res, dict) else res
            return QueryResponse(answer=str(answer).strip())
        elif callable(rag):
            out = rag(req.query)
            return QueryResponse(answer=str(out).strip())
        else:
            raise HTTPException(status_code=500, detail="RAG chain invocation method not found")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")

