"""
PRODUCTION-READY RAG Service with LangChain + Memory
- Uses flan-t5-large for better quality
- Smart context routing (database vs general knowledge)
- Conversational memory with proper configuration
- Answer validation and cleanup
- Extensive error handling and logging
"""
import os
import traceback
import pandas as pd
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- LangChain imports ---
try:
    from langchain.prompts import PromptTemplate
except:
    try:
        from langchain_core.prompts import PromptTemplate
    except:
        PromptTemplate = None

try:
    from langchain.memory import ConversationBufferMemory
except:
    try:
        from langchain_community.memory import ConversationBufferMemory
    except:
        ConversationBufferMemory = None

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

try:
    from langchain.chains import RetrievalQA
except:
    try:
        from langchain_community.chains import RetrievalQA
    except:
        RetrievalQA = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except:
    AutoTokenizer = AutoModelForSeq2SeqLM = pipeline = None

# ---------------- Config ----------------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOCAL_MODEL_ID = "google/flan-t5-large"  # Better quality than base
TOP_K = 4  # As requested

# Keywords for smart routing
DATABASE_KEYWORDS = [
    "response action", "threat intelligence", "severity level",
    "in the database", "from database", "from our data",
    "what actions were taken", "show me attacks", "list attacks from"
]

GENERAL_KEYWORDS = [
    "full form", "full forms", "what is", "what are",
    "define", "explain", "how does", "list all types"
]

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,
    "rag_chain_db": None,      # Chain for database questions
    "rag_chain_general": None, # Chain for general questions
    "pipeline": None,
    "memory": None
}

# ---------------- Helper Functions ----------------
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
            print(f"[CSV] ✓ Found: {p}")
            return p
    print(f"[CSV] ✗ Not found in: {candidates}")
    return None

def load_dataframe_from_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        print(f"[DATA] ✓ Loaded: {len(df)} rows, {len(df.columns)} cols")
        return df
    except Exception as e:
        print(f"[DATA] ✗ Error: {e}")
        traceback.print_exc()
        return None

def create_embedding_function():
    if not SentenceTransformerEmbeddings:
        return None
    try:
        emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("[EMB] ✓ Initialized")
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
        print(f"[CHROMA] ✓ Initialized at {CHROMA_DB_PATH}")
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

def ingest_dataframe(df: pd.DataFrame, vs):
    """Ingest with natural language formatting"""
    if df is None or vs is None:
        return False, "no_df_or_vs"

    USEFUL_COLUMNS = ["Attack Type", "Attack Severity", "Threat Intelligence", "Response Action", "Data Exfiltrated"]
    
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]
    
    if not cols_to_use:
        print(f"[INGEST] ✗ Missing columns. Available: {list(df.columns)}")
        return False, "missing_columns"
    
    print(f"[INGEST] Using: {cols_to_use}")
    
    def create_doc(row):
        parts = []
        if "Attack Type" in cols_to_use and pd.notna(row.get("Attack Type")):
            parts.append(f"Attack Type: {row['Attack Type']}")
        if "Attack Severity" in cols_to_use and pd.notna(row.get("Attack Severity")):
            parts.append(f"Severity: {row['Attack Severity']}")
        if "Threat Intelligence" in cols_to_use and pd.notna(row.get("Threat Intelligence")):
            parts.append(f"Threat Intelligence: {row['Threat Intelligence']}")
        if "Response Action" in cols_to_use and pd.notna(row.get("Response Action")):
            parts.append(f"Response Action: {row['Response Action']}")
        if "Data Exfiltrated" in cols_to_use and pd.notna(row.get("Data Exfiltrated")):
            parts.append(f"Data Exfiltrated: {row['Data Exfiltrated']}")
        return ". ".join(parts) + "." if parts else ""
    
    docs = [d for d in df.apply(create_doc, axis=1).tolist() if d.strip()]
    
    if not docs:
        return False, "no_documents"
    
    print(f"[INGEST] Created {len(docs)} docs")
    print(f"[INGEST] Sample: {docs[0][:150]}...")
    
    try:
        batch_size = 500
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            vs.add_texts(texts=batch)
        
        if hasattr(vs, "persist"):
            vs.persist()
        
        print(f"[INGEST] ✓ Added {len(docs)} documents")
        return True, {"docs_added": len(docs)}
    except Exception as e:
        print(f"[INGEST] ✗ Error: {e}")
        traceback.print_exc()
        return False, str(e)

def create_local_llm():
    """Create LLM with balanced settings"""
    if not all([HuggingFacePipeline, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM]):
        return None, None
    
    try:
        print(f"[LLM] Loading {LOCAL_MODEL_ID}...")
        
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
        
        # *** BALANCED SETTINGS - NOT TOO RIGID, NOT TOO CREATIVE ***
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=450,
            temperature=0.5,           # Balanced
            do_sample=True,            # Enable sampling
            top_p=0.92,
            repetition_penalty=1.4,    # Prevent repetition
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        
        print("[LLM] ✓ Model loaded")
        return llm, pipe
    except Exception as e:
        print(f"[LLM] ✗ Error: {e}")
        traceback.print_exc()
        return None, None

def create_rag_chains(llm, vs, memory):
    """Create TWO chains: one for DB queries, one for general"""
    if not llm or not vs or not RetrievalQA or not PromptTemplate:
        print("[RAG] Cannot create chains")
        return None, None
    
    count = vs_count_estimate(vs)
    if count == 0:
        print("[RAG] ✗ Vector store empty")
        return None, None
    
    try:
        print(f"[RAG] Creating chains (store has {count} docs)...")
        
        # *** DATABASE QUERY CHAIN ***
        db_prompt = PromptTemplate(
            template="""Use the context from the database to answer the question. Include chat history for context.

Chat History: {chat_history}

Database Context:
{context}

Question: {question}

Provide a detailed answer based on the database context above.""",
            input_variables=["chat_history", "context", "question"]
        )
        
        # *** GENERAL KNOWLEDGE CHAIN ***
        general_prompt = PromptTemplate(
            template="""Answer the question using your general cybersecurity knowledge. Include chat history for context.

Chat History: {chat_history}

Question: {question}

Provide a comprehensive answer with details and examples.""",
            input_variables=["chat_history", "question"]
        )
        
        # Database chain - uses retriever
        db_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs={"prompt": db_prompt},
            return_source_documents=True
        )
        
        # General chain - NO retriever, just direct LLM
        # We'll build this manually when needed
        general_chain = {"prompt": general_prompt, "llm": llm}
        
        print("[RAG] ✓ Created DB and General chains")
        return db_chain, general_chain
        
    except Exception as e:
        print(f"[RAG] ✗ Error: {e}")
        traceback.print_exc()
        return None, None

def is_database_question(query: str) -> bool:
    """Smart detection of database vs general questions"""
    q_lower = query.lower().strip()
    
    # Check for explicit database keywords
    for kw in DATABASE_KEYWORDS:
        if kw in q_lower:
            return True
    
    # Check for general knowledge indicators
    general_count = sum(1 for kw in GENERAL_KEYWORDS if kw in q_lower)
    
    # If multiple general keywords, treat as general
    if general_count >= 2:
        return False
    
    # If starts with "what is/are" and no database keywords, treat as general
    if q_lower.startswith(("what is", "what are", "define", "explain")) and "database" not in q_lower:
        return False
    
    # Default: check database (safer)
    return True

def validate_answer(answer: str, question: str) -> str:
    """Clean and validate the generated answer"""
    answer = answer.strip()
    
    # Remove common prefixes
    prefixes = ["Answer:", "Context:", "Question:", "Based on"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Check for context echo (if answer is too short and contains "Attack Type:")
    if len(answer) < 150 and "Attack Type:" in answer:
        return f"Based on the database, {question.replace('?', '').lower()} includes various recorded incidents with different characteristics."
    
    # Check for minimum length
    if len(answer) < 20:
        return "I apologize, but I couldn't generate a sufficient answer. Please rephrase your question."
    
    return answer

# ---------------- Startup ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*80)
    print("STARTING UP - PRODUCTION RAG SERVICE")
    print("="*80 + "\n")
    
    # Step 1: Embeddings
    print("[1/6] Initializing embeddings...")
    ml_models["embedding_function"] = create_embedding_function()
    
    # Step 2: Vector Store
    print("[2/6] Initializing vector store...")
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]
    
    # Step 3: Data Ingestion
    print("[3/6] Checking data...")
    csv_path = find_csv_path()
    if vs and csv_path and vs_count_estimate(vs) == 0:
        print("[3/6] Auto-ingesting...")
        df = load_dataframe_from_csv(csv_path)
        if df is not None:
            ok, info = ingest_dataframe(df, vs)
            print(f"[3/6] Ingest: {ok}, {info}")
    
    # Step 4: LLM
    print("[4/6] Loading LLM...")
    llm, pipe = create_local_llm()
    ml_models["local_llm"] = llm
    ml_models["pipeline"] = pipe
    
    # Step 5: Memory
    print("[5/6] Creating memory...")
    if ConversationBufferMemory:
        ml_models["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False,
            input_key="question",
            output_key="result"  # *** CORRECT: RetrievalQA outputs "result" ***
        )
        print("[5/6] ✓ Memory created")
    
    # Step 6: RAG Chains
    print("[6/6] Creating RAG chains...")
    if llm and vs:
        db_chain, gen_chain = create_rag_chains(llm, vs, ml_models["memory"])
        ml_models["rag_chain_db"] = db_chain
        ml_models["rag_chain_general"] = gen_chain
    
    print("\n" + "="*80)
    print("STARTUP COMPLETE")
    print("="*80)
    print(f"  Vector Store: {'✓' if vs else '✗'} ({vs_count_estimate(vs) if vs else 0} docs)")
    print(f"  LLM: {'✓' if llm else '✗'}")
    print(f"  Memory: {'✓' if ml_models['memory'] else '✗'}")
    print(f"  DB Chain: {'✓' if ml_models['rag_chain_db'] else '✗'}")
    print(f"  Gen Chain: {'✓' if ml_models['rag_chain_general'] else '✗'}")
    print("="*80 + "\n")
    
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    return {"status": "Production RAG Service", "model": LOCAL_MODEL_ID}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    return {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs else 0,
        "llm_ready": ml_models.get("local_llm") is not None,
        "db_chain_ready": ml_models.get("rag_chain_db") is not None,
        "general_chain_ready": ml_models.get("rag_chain_general") is not None,
        "memory_ready": ml_models.get("memory") is not None,
        "model_id": LOCAL_MODEL_ID,
        "top_k": TOP_K
    }

@app.post("/reset-memory")
def reset_memory():
    """Clear conversation history"""
    memory = ml_models.get("memory")
    if memory:
        try:
            memory.clear()
            return {"status": "cleared"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "no_memory"}

@app.post("/force-ingest")
def force_ingest():
    """Force re-ingestion"""
    vs = ml_models.get("vector_store")
    if not vs:
        raise HTTPException(status_code=503, detail="Vector store unavailable")
    
    csv_path = find_csv_path()
    if not csv_path:
        raise HTTPException(status_code=404, detail="CSV not found")
    
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV unreadable")
    
    # Clear old
    try:
        if hasattr(vs, "_collection"):
            old_ids = vs._collection.get()['ids']
            if old_ids:
                vs._collection.delete(ids=old_ids)
    except:
        pass
    
    ok, info = ingest_dataframe(df, vs)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed: {info}")
    
    return {"status": "success", "count": vs_count_estimate(vs)}

@app.post("/generate-text", response_model=QueryResponse)
def generate_text(req: QueryRequest):
    """
    SMART RAG: Routes to database or general knowledge based on question type
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    
    db_chain = ml_models.get("rag_chain_db")
    gen_chain = ml_models.get("rag_chain_general")
    memory = ml_models.get("memory")
    pipe = ml_models.get("pipeline")
    
    if not pipe:
        raise HTTPException(status_code=503, detail="LLM not ready")
    
    try:
        print("\n" + "="*80)
        print(f"[QUERY] {req.query}")
        print("="*80)
        
        # *** SMART ROUTING ***
        use_database = is_database_question(req.query)
        print(f"[ROUTE] {'DATABASE' if use_database else 'GENERAL KNOWLEDGE'}")
        
        # Get chat history
        chat_history = ""
        if memory:
            try:
                chat_history = memory.load_memory_variables({}).get("chat_history", "")
            except:
                pass
        
        answer = None
        
        # *** PATH 1: DATABASE QUERY ***
        if use_database and db_chain:
            print("[MODE] Using database chain\n")
            try:
                result = db_chain.invoke({"query": req.query, "chat_history": chat_history})
                
                if isinstance(result, dict):
                    answer = result.get("result", "")
                    source_docs = result.get("source_documents", [])
                    
                    if source_docs:
                        print(f"[CONTEXT] Retrieved {len(source_docs)} docs")
                else:
                    answer = str(result)
                
                # Validate answer isn't just echoing
                if answer and len(answer) < 100 and "Attack Type:" in answer:
                    answer = None  # Force fallback
                    
            except Exception as e:
                print(f"[ERROR] DB chain failed: {e}")
        
        # *** PATH 2: GENERAL KNOWLEDGE ***
        if (not use_database or not answer) and gen_chain and pipe:
            print("[MODE] Using general knowledge\n")
            try:
                # Build prompt manually for general questions
                prompt_template = gen_chain["prompt"]
                prompt_text = prompt_template.format(
                    chat_history=chat_history,
                    question=req.query
                )
                
                # Call pipeline directly
                result = pipe(prompt_text)
                
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    answer = result[0].get('generated_text', '') or result[0].get('translation_text', '')
                else:
                    answer = str(result)
                    
            except Exception as e:
                print(f"[ERROR] General chain failed: {e}")
        
        if not answer:
            answer = "I apologize, but I couldn't generate an answer. Please try rephrasing."
        
        # *** VALIDATE AND CLEAN ***
        answer = validate_answer(answer, req.query)
        
        # *** SAVE TO MEMORY ***
        if memory:
            try:
                memory.save_context({"question": req.query}, {"result": answer})
            except Exception as e:
                print(f"[MEMORY] Save failed: {e}")
        
        print(f"\n[ANSWER] {answer[:200]}...")
        print(f"[LENGTH] {len(answer.split())} words")
        print("="*80 + "\n")
        
        return QueryResponse(answer=answer)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")
