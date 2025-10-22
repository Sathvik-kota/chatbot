"""
Robust RAG service with LOCAL MODEL support - HYBRID VERSION
Handles BOTH context-grounded answers AND general knowledge fallback
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

LOCAL_MODEL_ID = "google/flan-t5-base"

TOP_K = 10  # Retrieve more documents for better coverage

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
    """Creates CLEAN documents from relevant columns"""
    if df is None or vs is None:
        return False, "no_df_or_vs"

    USEFUL_COLUMNS = [
        "Attack Type",
        "Attack Severity", 
        "Threat Intelligence",
        "Response Action"
    ]
    
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[col_name.lower()] for col_name in USEFUL_COLUMNS if col_name.lower() in df_cols_lower]

    if not cols_to_use:
        print("[INGEST] ERROR: Could not find expected columns")
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
        print("[INGEST] ERROR: No documents created")
        return False, "no_documents_created"
    
    print(f"[INGEST] Created {len(docs)} clean documents")
    print(f"[INGEST] Sample: {docs[0][:200]}...")
    
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
    """Create LLM optimized for both grounded and general answers"""
    if HuggingFacePipeline is None or pipeline is None or AutoTokenizer is None:
        print("[LLM] Required libraries not available.")
        return None
    
    try:
        print(f"[LLM] Loading model: {LOCAL_MODEL_ID}")
        
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        
        if "t5" in LOCAL_MODEL_ID.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
            task = "text2text-generation"
        else:
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID)
            task = "text-generation"
        
        # Balanced settings for both context-grounded and general knowledge
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            min_length=30,
            temperature=0.4,           # Balanced creativity
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=False
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print(f"[LLM] Model loaded successfully")
        return llm
        
    except Exception as e:
        print(f"[LLM] Failed to load model: {e}")
        traceback.print_exc()
        return None

def create_rag_chain(llm, vs):
    """
    Create HYBRID RAG chain that intelligently uses context OR general knowledge
    """
    if llm is None or vs is None:
        print("[RAG] Cannot create chain: llm or vs is None")
        return None
    
    if RetrievalQA is None:
        print("[RAG] RetrievalQA not available")
        return None
    
    try:
        # *** HYBRID PROMPT: Uses context when relevant, general knowledge otherwise ***
        template = """You are a helpful AI assistant with cybersecurity expertise.

Context from database:
{context}

Question: {question}

Instructions:
1. First, carefully examine if the provided context contains information relevant to answering the question
2. If the context IS relevant:
   - Base your answer primarily on the context
   - Extract and list unique items if it's a listing question
   - Provide detailed explanations if it's an explanation question
   - Cite that your answer is "based on the database"
3. If the context is NOT relevant or empty:
   - Answer using your general cybersecurity knowledge
   - Provide comprehensive information (definitions, full forms, explanations)
   - Be clear that you're providing general information
4. For listing questions:
   - Extract ALL unique items from context (if relevant)
   - OR provide a comprehensive general list (if context not relevant)
5. For explanation questions:
   - Provide 3-5 detailed sentences
6. NEVER repeat the same information multiple times
7. NEVER output instructions or meta-commentary

Answer:"""
        
        prompt = None
        if PromptTemplate is not None:
            try:
                prompt = PromptTemplate(
                    template=template, 
                    input_variables=["context", "question"]
                )
                print("[RAG] Created hybrid RAG prompt (context + general knowledge)")
            except Exception:
                traceback.print_exc()
        
        chain_type_kwargs = {}
        if prompt is not None:
            chain_type_kwargs["prompt"] = prompt
        
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        
        print("[RAG] Hybrid RAG chain created successfully")
        return rag
        
    except Exception as e:
        print(f"[RAG] Failed to create chain: {e}")
        traceback.print_exc()
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting up...")
    
    ml_models["embedding_function"] = create_embedding_function()
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]

    csv_path = find_csv_path()
    if csv_path:
        print(f"[LIFESPAN] Found CSV: {csv_path}")
    
    if vs is not None and csv_path:
        try:
            cnt = vs_count_estimate(vs)
            if cnt == 0:
                print("[LIFESPAN] Auto-ingesting CSV...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None and len(df) > 0:
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] Ingest result:", ok, info)
        except Exception:
            traceback.print_exc()

    print("[LIFESPAN] Loading LLM...")
    ml_models["local_llm"] = create_local_llm()
    
    if ml_models["local_llm"] is not None and vs is not None:
        ml_models["rag_chain"] = create_rag_chain(ml_models["local_llm"], vs)
    
    print("[LIFESPAN] Startup complete")
    
    yield
    ml_models.clear()
    print("[LIFESPAN] Shutdown complete")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Hybrid RAG Service with context + general knowledge"}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    count = vs_count_estimate(vs) if vs is not None else 0
    return {
        "vector_store_ready": vs is not None,
        "vector_store_count": int(count),
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "model_id": LOCAL_MODEL_ID,
        "top_k": TOP_K,
        "mode": "hybrid_rag"
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
        raise HTTPException(status_code=503, detail="Vector store not available")
    csv_path = find_csv_path()
    if csv_path is None:
        raise HTTPException(status_code=404, detail="CSV not found")
    df = load_dataframe_from_csv(csv_path)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV empty")
    if sample_limit is not None:
        df = df.head(sample_limit)
    
    try:
        if hasattr(vs, "_collection"):
            vs._collection.delete(ids=vs._collection.get()['ids'])
            print("[INGEST] Old data cleared")
    except Exception as e:
        print(f"[INGEST] Error clearing: {e}")

    ok, info = ingest_dataframe(df, vs, batch_docs=batch_size)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {info}")
    
    return {"status": "ingested", "info": info, "count": vs_count_estimate(vs)}

@app.get("/get-csv-columns")
def get_csv_columns():
    csv_path = find_csv_path()
    if csv_path is None:
        raise HTTPException(status_code=404, detail="CSV not found")
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV unreadable")
    return {"columns": list(df.columns)}

@app.post("/generate-text", response_model=QueryResponse)
def generate_text(req: QueryRequest):
    """
    HYBRID RAG: Uses context when relevant, general knowledge otherwise
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    rag = ml_models.get("rag_chain")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG chain not available")
    
    try:
        print("\n" + "="*80)
        print(f"[QUERY] {req.query}")
        print("="*80)
        
        out_text = None
        source_docs = []
        
        # Invoke RAG chain
        if hasattr(rag, "invoke"):
            res = rag.invoke({"query": req.query})
            if isinstance(res, dict):
                out_text = res.get("result")
                source_docs = res.get("source_documents", [])
            else:
                out_text = str(res)
        elif hasattr(rag, "__call__"):
            res = rag({"query": req.query})
            if isinstance(res, dict):
                out_text = res.get("result")
                source_docs = res.get("source_documents", [])
            else:
                out_text = str(res)
        else:
            raise HTTPException(status_code=500, detail="RAG invocation failed")
        
        # Log retrieved context
        if source_docs:
            print(f"\n[CONTEXT] Retrieved {len(source_docs)} documents")
            # Extract attack types for debugging
            attack_types = set()
            for i, doc in enumerate(source_docs[:3]):  # Show first 3
                print(f"  Doc {i+1}: {doc.page_content[:200]}...")
                if "Attack Type:" in doc.page_content:
                    try:
                        at = doc.page_content.split("Attack Type:")[1].split(".")[0].strip()
                        attack_types.add(at)
                    except:
                        pass
            if attack_types:
                print(f"  Found attack types: {attack_types}")
        else:
            print("\n[CONTEXT] No documents retrieved (will use general knowledge)")
        
        if out_text is None:
            raise HTTPException(status_code=500, detail="Failed to get result")
        
        out_text = str(out_text).strip()
        
        # Clean up any instruction leakage
        bad_phrases = [
            "Instructions:",
            "Do NOT",
            "NEVER",
            "You must",
            "based ONLY on"
        ]
        for phrase in bad_phrases:
            if phrase in out_text:
                # If instructions leaked, try to extract just the answer
                parts = out_text.split("\n\n")
                for part in parts:
                    if not any(bp in part for bp in bad_phrases) and len(part) > 50:
                        out_text = part
                        break
        
        print(f"\n[ANSWER] {out_text}")
        print(f"[LENGTH] {len(out_text.split())} words")
        print("="*80 + "\n")
        
        return QueryResponse(answer=out_text)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")
