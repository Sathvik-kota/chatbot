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

TOP_K = 1 # Use only the single best match

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

    # --- *** FIXED LOGIC: Using REAL columns from your CSV *** ---
    
    # Define columns that likely contain useful, descriptive text. Case-insensitive.
    USEFUL_COLUMNS = [
        "attack type",
        "attack severity",
        "threat intelligence",
        "response action",
        # Keep these just in case
        "description", 
        "summary", 
        "text", 
        "alert description", 
        "details"
    ]
    
    # Get lowercased versions of actual columns
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Find the original-cased column names that match our useful list
    cols_to_use = [df_cols_lower[col_name] for col_name in USEFUL_COLUMNS if col_name in df_cols_lower]

    docs = []
    if cols_to_use:
        print(f"[INGEST] Found useful text columns: {cols_to_use}")
        # Create a "document" by joining just these columns, adding the column name as context
        # e.g., "Attack Type: DDoS. Attack Severity: High..."
        def create_doc(row):
            return ". ".join([f"{col}: {row[col]}" for col in cols_to_use if pd.notna(row[col]) and row[col]])
        
        docs = df.apply(create_doc, axis=1).tolist()
        
    else:
        # Fallback to old (noisy) method with a clear warning
        print("[INGEST] WARNING: No specific text columns found (e.g., 'Description', 'Summary', 'Attack Type').")
        print("[INGEST] Falling back to ingesting ALL columns. This may lead to irrelevant/noisy answers.")
        docs = df.fillna("").apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()
    
    # Filter out any empty documents that might have been created
    docs = [d for d in docs if d.strip()]
    if not docs:
        print("[INGEST] ERROR: No documents were created from the DataFrame.")
        return False, "no_documents_created"
    # --- END FIXED LOGIC ---

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
        
        # --- *** SIMPLIFIED Generation Parameters *** ---
        # We are removing min_length and other "creative" settings
        # to make the model more factual and less likely to hallucinate.
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256, # Changed from max_length
            temperature=0.7,
            top_p=0.9,
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
        # --- *** FEW-SHOT PROMPT *** ---
        # We give the model EXAMPLES of how to behave. This is a very
        # powerful way to get the correct behavior from flan-t5.
        template = """You are a cybersecurity expert. Follow the examples.

Example 1:
Context: Attack Type: Malware. Attack Severity: Medium. Threat Intelligence: Contained.
Question: What is SQL Injection?
Answer: SQL Injection (SQLi) is a type of cyberattack where an attacker inserts malicious SQL code into queries to manipulate a database.

Example 2:
Context: Attack Type: DDoS. Attack Severity: High. Threat Intelligence: Contained. Response Action: Eradicated.
Question: What was the response action for the DDoS attack?
Answer: The response action for the DDoS attack was: Eradicated.

---
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
            return_source_documents=True  # *** CHANGED: We want to see the documents
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
    
    # --- Clear the old (noisy) data first ---
    try:
        print("[INGEST] Clearing old data from vector store...")
        if hasattr(vs, "_collection"):
            vs._collection.delete(ids=vs._collection.get()['ids'])
            print("[INGEST] Old data cleared.")
        else:
            print("[INGEST] Could not automatically clear old data. Re-ingesting anyway.")
    except Exception as e:
        print(f"[INGEST] Error clearing old data: {e}. Continuing...")
        traceback.print_exc()

    # --- Ingest the new (clean) data ---
    ok, info = ingest_dataframe(df, vs, batch_docs=batch_size)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {info}")
    
    final_count = vs_count_estimate(vs)
    return {"status": "ingested", "method_info": info, "final_count": int(final_count)}

# --- NEW DEBUG ENDPOINT ---
@app.get("/get-csv-columns")
def get_csv_columns():
    csv_path = find_csv_path()
    if csv_path is None:
        raise HTTPException(status_code=404, detail="CSV not found.")
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV empty or unreadable.")
    return {"columns": list(df.columns)}
# --- END NEW ENDPOINT ---

@app.post("/generate-text", response_model=QueryResponse)
def generate_text(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    rag = ml_models.get("rag_chain")
    llm = ml_models.get("local_llm") # Get the raw LLM

    if rag is None or llm is None:
        raise HTTPException(status_code=503, detail="RAG chain or LLM not available. Model may still be loading.")
    
    try:
        # --- NEW DEBUGGING LOGIC ---
        print("\n--- NEW QUERY RECEIVED ---")
        print(f"Query: {req.query}")
        
        # --- FIXED INVOCATION LOGIC ---
        out_text = None
        source_docs = []
        
        # --- *** NEW ADAPTIVE LOGIC *** ---
        # If it's a "What is" question, bypass RAG and just use the LLM's own knowledge.
        # This prevents context-poisoning for definitions.
        query_lower = req.query.strip().lower()
        if query_lower.startswith("what is") or query_lower.startswith("what are"):
            print("[INFO] Definitional question detected. Bypassing RAG to use model's internal knowledge.")
            if hasattr(llm, "invoke"):
                out_text = llm.invoke(req.query)
            elif hasattr(llm, "run"):
                out_text = llm.run(req.query)
            elif callable(llm):
                out_text = llm(req.query)
            
            # The raw llm.invoke() on a HuggingFacePipeline often returns the full text
            if isinstance(out_text, list) and out_text:
                if isinstance(out_text[0], dict):
                    out_text = out_text[0].get('generated_text')
            
            out_text = str(out_text)

        else:
            print("[INFO] Specific question detected. Using RAG chain.")
            # We MUST use .invoke() because return_source_documents=True
            if hasattr(rag, "invoke"):
                # .invoke returns a dict, which should have sources
                res = rag.invoke({"query": req.query})
                if isinstance(res, dict):
                    out_text = res.get("result")
                    source_docs = res.get("source_documents")
                else:
                    out_text = str(res)
            elif hasattr(rag, "run"):
                 # This is now a fallback, but it shouldn't be hit
                print("[WARN] Using .run() fallback, source documents will not be logged.")
                out_text = rag.run(req.query)
            elif callable(rag):
                # Fallback for older chain types
                print("[WARN] Using callable() fallback, source documents will not be logged.")
                out_text = rag(req.query)
            else:
                raise HTTPException(status_code=500, detail="RAG chain invocation method not found")

        # --- LOG THE RETRIEVED CONTEXT (if RAG was used) ---
        if source_docs:
            print(f"[DEBUG] Retrieved {len(source_docs)} source document(s):")
            for i, doc in enumerate(source_docs):
                print(f"  DOC {i+1}: {doc.page_content[:500]}...") # Log first 500 chars
        else:
            # This is now expected for "what is" questions
            if not (query_lower.startswith("what is") or query_lower.startswith("what are")):
                print("[DEBUG] No source documents were returned by the chain (or .invoke() failed).")
        
        if out_text is None:
             print("[ERROR] Failed to extract 'result' from RAG chain response.")
             raise HTTPException(status_code=500, detail="Failed to get 'result' from RAG chain.")

        print(f"Generated Answer: {out_text}")
        print("--- QUERY COMPLETE ---")
        
        return QueryResponse(answer=str(out_text).strip())

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")

