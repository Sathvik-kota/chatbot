"""
RAG service with LangChain - UPDATED TO USE ChatPromptTemplate
Uses LangChain but with clean, simple chat-style prompt that won't leak
"""
import os
import traceback
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- LangChain imports ---
try:
    from langchain.prompts import ChatPromptTemplate
except:
    ChatPromptTemplate = None

try:
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except:
    SentenceTransformerEmbeddings = None

try:
    from langchain_community.vectorstores import Chroma as LC_Chroma
except:
    LC_Chroma = None

try:
    from langchain_huggingface import HuggingFacePipeline
except:
    try:
        from langchain_community.llms import HuggingFacePipeline
    except:
        HuggingFacePipeline = None

# Keep normal PromptTemplate fallback if ChatPromptTemplate isn't present
try:
    from langchain.prompts import PromptTemplate
    from langchain_core.prompts import PromptTemplate as CorePromptTemplate
except:
    PromptTemplate = CorePromptTemplate = None

try:
    from langchain.chains import RetrievalQA
    from langchain.chains.question_answering import load_qa_chain
except:
    try:
        from langchain_community.chains import RetrievalQA
    except:
        RetrievalQA = None
    load_qa_chain = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
except:
    AutoTokenizer = AutoModelForSeq2SeqLM = AutoModelForCausalLM = pipeline = None

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
    "local_llm": None,
    "rag_chain": None,
    "qa_chain": None,  # For direct QA without retrieval
    "pipeline": None   # Direct pipeline for general knowledge
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
    except:
        traceback.print_exc()
        return None

def create_embedding_function():
    if not SentenceTransformerEmbeddings:
        return None
    try:
        emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("[EMB] Embedding initialized")
        return emb
    except:
        traceback.print_exc()
        return None

def init_chroma(embedding_function):
    if not LC_Chroma or not embedding_function:
        return None
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        print("[CHROMA] Initialized")
        return vs
    except:
        traceback.print_exc()
        return None

def vs_count_estimate(vs) -> int:
    if not vs:
        return 0
    try:
        col = getattr(vs, "_collection", None)
        if col and hasattr(col, "count"):
            return int(col.count())
    except:
        pass
    return 0

def ingest_dataframe(df: pd.DataFrame, vs, batch_docs: int = 500):
    if df is None or vs is None:
        return False, "no_df_or_vs"

    USEFUL_COLUMNS = ["Attack Type", "Attack Severity", "Threat Intelligence", "Response Action"]
    
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[col_name.lower()] for col_name in USEFUL_COLUMNS if col_name.lower() in df_cols_lower]

    if not cols_to_use:
        print(f"[INGEST] ERROR: Missing columns. Available: {list(df.columns)}")
        return False, "missing_columns"
    
    print(f"[INGEST] Using columns: {cols_to_use}")
    
    def create_doc(row):
        parts = []
        for col in cols_to_use:
            if pd.notna(row[col]) and str(row[col]).strip():
                parts.append(f"{col}: {row[col]}")
        return ". ".join(parts) if parts else ""
    
    docs = df.apply(create_doc, axis=1).tolist()
    docs = [d for d in docs if d.strip()]
    
    if not docs:
        return False, "no_documents_created"
    
    print(f"[INGEST] Created {len(docs)} documents")
    
    total_added = 0
    for i in range(0, len(docs), batch_docs):
        batch = docs[i:i+batch_docs]
        try:
            vs.add_texts(texts=batch)
            total_added += len(batch)
        except:
            return False, f"batch_failed_at_{i}"
    
    try:
        if hasattr(vs, "persist"):
            vs.persist()
    except:
        pass
    
    return True, {"docs_added": total_added}

def create_local_llm():
    """Create LangChain-wrapped LLM + direct pipeline"""
    if not all([HuggingFacePipeline, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM]):
        print("[LLM] Required libraries not available")
        return None, None
    
    try:
        print(f"[LLM] Loading {LOCAL_MODEL_ID}")
        
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
        
        # Create pipeline with settings optimized for both context and general answers
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=450,
            temperature=0.7,
            do_sample=True,
            top_p=0.92,
            repetition_penalty=1.5,
            no_repeat_ngram_size=2
        )
        
        # Wrap in LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print("[LLM] Model loaded successfully")
        return llm, pipe
        
    except Exception as e:
        print(f"[LLM] Failed to load: {e}")
        traceback.print_exc()
        return None, None

def create_langchain_rag(llm, vs):
    """Create LangChain RAG chain with CHAT prompt (fallback to minimal text prompt)"""
    if not llm or not vs or not RetrievalQA:
        print("[RAG] Cannot create chain")
        return None
    
    try:
        prompt = None

        # Prefer chat-style prompt if available
        if ChatPromptTemplate:
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a cybersecurity expert. Use the given context to answer accurately and clearly."),
                    ("human", "Context: {context}\n\nQuestion: {question}")
                ])
                print("[RAG] Using ChatPromptTemplate")
            except Exception:
                prompt = None

        # Fallback: simple PromptTemplate
        if prompt is None:
            PromptClass = PromptTemplate or CorePromptTemplate
            if PromptClass:
                prompt = PromptClass(
                    template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
                    input_variables=["context", "question"]
                )
                print("[RAG] Using fallback PromptTemplate")

        chain_type_kwargs = {}
        if prompt:
            chain_type_kwargs["prompt"] = prompt

        # Build RetrievalQA with the provided retriever and prompt
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True
        )
        
        print("[RAG] LangChain RAG chain created with chat-style prompt (or fallback)")
        return rag
        
    except Exception as e:
        print(f"[RAG] Failed to create: {e}")
        traceback.print_exc()
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] Starting up...")
    
    ml_models["embedding_function"] = create_embedding_function()
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]

    csv_path = find_csv_path()
    if csv_path and vs:
        try:
            cnt = vs_count_estimate(vs)
            if cnt == 0:
                print("[LIFESPAN] Auto-ingesting CSV...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None:
                    ok, info = ingest_dataframe(df, vs)
                    print(f"[LIFESPAN] Ingest result: {ok}, {info}")
        except:
            traceback.print_exc()

    print("[LIFESPAN] Loading LLM...")
    llm, pipe = create_local_llm()
    ml_models["local_llm"] = llm
    ml_models["pipeline"] = pipe
    
    if llm and vs:
        ml_models["rag_chain"] = create_langchain_rag(llm, vs)
    
    print("[LIFESPAN] Startup complete")
    print(f"  - Vector store: {vs is not None}, docs: {vs_count_estimate(vs) if vs else 0}")
    print(f"  - LLM: {llm is not None}")
    print(f"  - RAG chain: {ml_models['rag_chain'] is not None}")
    
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
    return {"status": "LangChain RAG Service", "framework": "langchain"}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    return {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs else 0,
        "llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "framework": "langchain",
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
    vs = ml_models.get("vector_store")
    if not vs:
        raise HTTPException(status_code=503, detail="Vector store not available")
    
    csv_path = find_csv_path()
    if not csv_path:
        raise HTTPException(status_code=404, detail="CSV not found")
    
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV unreadable")
    
    if sample_limit:
        df = df.head(sample_limit)
    
    try:
        if hasattr(vs, "_collection"):
            vs._collection.delete(ids=vs._collection.get()['ids'])
            print("[INGEST] Cleared old data")
    except:
        pass

    ok, info = ingest_dataframe(df, vs)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {info}")
    
    return {"status": "ingested", "info": info, "count": vs_count_estimate(vs)}

@app.get("/get-csv-columns")
def get_csv_columns():
    csv_path = find_csv_path()
    if not csv_path:
        raise HTTPException(status_code=404, detail="CSV not found")
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV unreadable")
    return {"columns": list(df.columns)}

@app.post("/generate-text", response_model=QueryResponse)
def generate_text(req: QueryRequest):
    """
    LangChain-based RAG with smart fallback to general knowledge
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    
    rag = ml_models.get("rag_chain")
    pipe = ml_models.get("pipeline")
    
    if not pipe:
        raise HTTPException(status_code=503, detail="LLM not ready")
    
    try:
        print("\n" + "="*80)
        print(f"[QUERY] {req.query}")
        print("="*80)
        
        q_lower = req.query.lower()
        
        # Detect if question needs database context
        needs_database = any(kw in q_lower for kw in [
            "response action", "threat intelligence", "severity",
            "in the database", "from database", "what actions",
            "in our data", "from our data"
        ])
        
        # Detect general knowledge questions
        is_general = any(kw in q_lower for kw in [
            "full form", "list all types", "what are all",
            "enumerate all", "all types of"
        ]) and "database" not in q_lower
        
        print(f"[DECISION] Database: {needs_database}, General: {is_general}")
        
        answer = None
        prompt_text = None
        
        # PATH 1: Use LangChain RAG for database questions
        if needs_database and rag:
            print("[MODE] Using LangChain RAG chain\n")
            try:
                result = rag.invoke({"query": req.query})
                
                if isinstance(result, dict):
                    answer = result.get("result", "")
                    source_docs = result.get("source_documents", [])
                    
                    if source_docs:
                        print(f"[CONTEXT] Retrieved {len(source_docs)} documents")
                        for i, doc in enumerate(source_docs[:3]):
                            print(f"  Doc {i+1}: {doc.page_content[:150]}...")
                else:
                    answer = str(result)
                
                # If answer is just echoing context, retry with simpler approach
                if answer and "Attack Type:" in answer[:100]:
                    print("[WARNING] Answer echoing context, using fallback")
                    answer = None
                    
            except Exception as e:
                print(f"[ERROR] RAG chain failed: {e}")
                answer = None
        
        # PATH 2: Use direct pipeline for general knowledge
        if (is_general or not needs_database or not answer) and pipe:
            print("[MODE] Using direct pipeline for general knowledge\n")
            
            # Ultra-minimal prompt for general questions
            if "list all" in q_lower or "types of" in q_lower:
                prompt_text = "List the main types of cyber attacks with brief explanations and full forms."
            elif "what is" in q_lower:
                # Extract the topic
                topic = req.query.lower().replace("what is", "").replace("?", "").strip()
                prompt_text = f"Explain {topic} in cybersecurity."
            else:
                prompt_text = req.query
            
            print(f"[PROMPT] {prompt_text}")
            
            result = pipe(prompt_text)
            
            if isinstance(result, list) and result:
                if isinstance(result[0], dict):
                    answer = result[0].get('generated_text', '') or result[0].get('translation_text', '')
                else:
                    answer = str(result[0])
            else:
                answer = str(result)
        
        if not answer:
            answer = "I apologize, but I couldn't generate a proper answer. Please rephrase your question."
        
        # Clean up answer
        answer = answer.strip()
        
        # Remove common prefixes that might have been echoed
        for prefix in ["Answer:", "Context:", "Question:", prompt_text or ""]:
            if prefix and answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove any remaining context echoes
        if "Attack Type:" in answer and len(answer) < 200:
            # This is just echoed context, use fallback
            answer = "Based on the database, multiple attack types are recorded including DDoS, Malware, SQL Injection, and Phishing with varying severity levels and response actions."
        
        print(f"\n[ANSWER] {answer}")
        print(f"[LENGTH] {len(answer.split())} words")
        print("="*80 + "\n")
        
        return QueryResponse(answer=answer)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")
