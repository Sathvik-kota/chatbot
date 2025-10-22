import os
import pandas as pd
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline

# ==============================================================
# CONFIGURATION
# ==============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ai_cybersecurity_dataset_sampled_5k.csv")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app = FastAPI(title="SIEM Text Generation Service")

ml_models = {
    "embeddings": None,
    "vector_store": None,
    "rag_chain": None,
}

# ==============================================================
# UTILITIES
# ==============================================================

def load_csv_report():
    """Load the local SIEM dataset CSV."""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"[LOAD] Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        print("[LOAD] Failed to read CSV:", e)
        traceback.print_exc()
        return None

def _count_vectorstore(vs):
    try:
        if hasattr(vs, "_collection") and hasattr(vs._collection, "count"):
            return int(vs._collection.count())
    except Exception:
        traceback.print_exc()
    try:
        if hasattr(vs, "get"):
            res = vs.get()
            if isinstance(res, dict) and "documents" in res:
                return len(res["documents"])
    except Exception:
        traceback.print_exc()
    return 0

def _try_add_texts(vs, chunks):
    """Try multiple signatures for adding text to Chroma wrapper."""
    try:
        if hasattr(vs, "add_texts"):
            vs.add_texts(texts=chunks)
            return True, "add_texts"
    except Exception as e:
        print("[INGEST] add_texts failed:", e)
        traceback.print_exc()

    try:
        if hasattr(vs, "add_documents"):
            docs = [{"page_content": t, "metadata": {}} for t in chunks]
            vs.add_documents(docs)
            return True, "add_documents"
    except Exception as e:
        print("[INGEST] add_documents failed:", e)
        traceback.print_exc()

    try:
        col = getattr(vs, "_collection", None)
        if col is not None and hasattr(col, "add"):
            col.add(documents=chunks)
            return True, "_collection.add"
    except Exception as e:
        print("[INGEST] _collection.add failed:", e)
        traceback.print_exc()

    return False, None


# ==============================================================
# MODEL INITIALIZATION
# ==============================================================

@app.on_event("startup")
async def startup_event():
    print("[LIFESPAN] Starting up textgen service...")

    # Initialize embeddings
    ml_models["embeddings"] = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("[EMB] Embedding function initialized.")

    # Initialize Chroma vector store
    ml_models["vector_store"] = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=ml_models["embeddings"],
    )
    print("[CHROMA] Initialized Chroma vector store.")

    # Initialize text generation model
    try:
        generator = pipeline(
            "text-generation",
            model="distilgpt2",
            max_length=256,
            temperature=0.7,
        )
        ml_models["generator"] = generator
        print("[GEN] Loaded distilgpt2 text generator.")
    except Exception as e:
        print("[GEN] Failed to load generator:", e)

    # Prepare RAG chain if data is available
    vs = ml_models["vector_store"]
    if vs:
        count = _count_vectorstore(vs)
        if count > 0:
            retriever = vs.as_retriever(search_kwargs={"k": 3})
            ml_models["rag_chain"] = RetrievalQA.from_chain_type(
                llm=None,  # optional, you can replace with real HF model later
                chain_type="stuff",
                retriever=retriever,
                verbose=True,
            )
            print(f"[RAG] Chain initialized with {count} documents.")
        else:
            print("[RAG] Vector store empty; chain not initialized.")

    print("[LIFESPAN] Startup complete.")
    print(get_status())


@app.on_event("shutdown")
async def shutdown_event():
    print("[LIFESPAN] Shutting down service...")

# ==============================================================
# ROUTES
# ==============================================================

class QueryRequest(BaseModel):
    query: str


@app.get("/status")
def get_status():
    vs = ml_models.get("vector_store")
    rag = ml_models.get("rag_chain")
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": _count_vectorstore(vs) > 0 if vs else False,
        "vector_store_count": _count_vectorstore(vs) if vs else 0,
        "rag_chain_ready": rag is not None,
        "hf_token_set": bool(HF_TOKEN),
    }


@app.post("/generate-text")
def generate_text(request: QueryRequest):
    query = request.query
    generator = ml_models.get("generator")
    if not generator:
        raise HTTPException(status_code=500, detail="Generator model not available")

    try:
        result = generator(query, max_length=128, num_return_sequences=1)
        return {"query": query, "response": result[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@app.post("/force-ingest")
async def force_ingest(background_tasks: BackgroundTasks):
    """Force ingestion of CSV into Chroma vectorstore."""
    vs = ml_models.get("vector_store")
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store not available (None).")

    df = load_csv_report()
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV not found or empty.")

    # Convert all rows to text form
    docs = df.fillna("").apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text("\n\n".join(docs))
    print(f"[INGEST] Prepared {len(docs)} docs → {len(chunks)} chunks")

    ok, method = _try_add_texts(vs, chunks)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to add documents to vector store (check logs).")

    # persist data
    if hasattr(vs, "persist"):
        try:
            vs.persist()
            print("[INGEST] Persist called.")
        except Exception:
            traceback.print_exc()

    count = _count_vectorstore(vs)
    return {
        "status": "success",
        "added_chunks": len(chunks),
        "method_used": method,
        "vectorstore_count": count,
    }
