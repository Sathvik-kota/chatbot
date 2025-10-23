"""
Robust RAG service with LOCAL MODEL support
Uses PromptTemplate to format a single string input for T5 models.
"""
import os
import traceback
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Try imports (be tolerant across environments) ---
# Per user instruction, T5 models expect a single string.
# We will remove the ChatPromptTemplate logic and rely on PromptTemplate
# to ensure a single string is always generated.
ChatPromptTemplate = None

try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None

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
LOCAL_MODEL_ID = "google/flan-t5-base"
TOP_K = 10

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,  # LangChain HuggingFacePipeline wrapper (if used)
    "rag_chain": None,
    # optional direct pipeline store (not required)
    "hf_pipeline": None
}

# ---------------- Chat prompt setup ----------------
# We will not use ChatPromptTemplate as T5 models work best with
# a single formatted string.
chat_prompt_obj = None

# This text_template will be our primary template, formatted as a single string.
text_template = """You are a cybersecurity expert assistant. Answer the question based ONLY on the provided context from the database.

Context from database:
{context}

Question: {question}

Instructions:
1. Carefully read ALL the context documents above
2. If the question asks to "list" or "what are the types", extract ALL UNIQUE items mentioned across all context documents
3. If the question asks to "explain" or "what is", provide a detailed 3-5 sentence explanation
4. For listing questions: Create a clear list or enumeration of unique items found in the context
5. For explanation questions: Provide comprehensive details with examples from the context
6. NEVER repeat the same item multiple times
7. If information is not in the context, say "I cannot find this information in the provided context."
8. Do NOT use general knowledge - use ONLY what's in the context above

Answer:"""

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
    # try direct chromadb client fallback
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

def _try_add_texts(vs, texts: List[str]):
    try:
        if hasattr(vs, "add_texts"):
            vs.add_texts(texts=texts)
            return True, "add_texts"
    except Exception:
        traceback.print_exc()
    try:
        if hasattr(vs, "add_documents"):
            # Workaround for add_documents expecting Document objects
            from langchain.docstore.document import Document
            docs = [Document(page_content=t, metadata={}) for t in texts]
            vs.add_documents(docs)
            return True, "add_documents"
    except Exception:
        traceback.print_exc()
    try:
        col = getattr(vs, "_collection", None) or getattr(vs, "collection", None)
        if col is not None and hasattr(col, "add"):
            try:
                # Generate simple IDs for chromadb
                ids = [f"doc_{i}" for i in range(len(texts))]
                col.add(documents=texts, metadatas=[{}]*len(texts), ids=ids)
            except Exception:
                 # Fallback if ids are not the issue
                 col.add(documents=texts)
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

    USEFUL_COLUMNS = ["Attack Type", "Attack Severity", "Threat Intelligence", "Response Action"]
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]
    if not cols_to_use:
        print("[INGEST] ERROR: Could not find expected columns.")
        print(f"[INGEST] Available columns: {list(df.columns)}")
        return False, "missing_expected_columns"

    print(f"[INGEST] Using text columns: {cols_to_use}")

    def create_doc(row):
        parts = []
        for col in cols_to_use:
            if pd.notna(row[col]) and str(row[col]).strip():
                parts.append(f"{col}: {row[col]}")
        return ". ".join(parts) if parts else ""

    docs = df.apply(create_doc, axis=1).tolist()
    docs = [d for d in docs if d.strip()]
    if not docs:
        print("[INGEST] ERROR: No documents were created from the DataFrame.")
        return False, "no_documents_created"

    print(f"[INGEST] Created {len(docs)} clean documents")
    print(f"[INGEST] Sample document: {docs[0][:200]}...")

    total_added
