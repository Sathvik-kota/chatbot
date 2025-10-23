"""
Robust RAG service with LOCAL MODEL support
Uses PromptTemplate to format a single string input for T5 models.
NOW WITH CONVERSATIONAL MEMORY.

Key Updates:
- MODEL UPGRADE: Switched from flan-t5-base to flan-t5-large.
  The 'base' model was not powerful enough to answer general questions
  and was hallucinating. 'large' should fix this.
- TOP_K set to 4 as requested.
- ingest_dataframe now formats CSV rows into natural language sentences.
- /generate-text endpoint now contains all-new conditional logic.
"""
import os
import traceback
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Try imports (be tolerant across environments) ---
try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None

try:
    from langchain.memory import ConversationBufferMemory
except Exception:
    try:
        from langchain_community.memory import ConversationBufferMemory
    except Exception:
        ConversationBufferMemory = None

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
# --- MODEL UPGRADE ---
# 'base' was not powerful enough. Using 'large' to get better general knowledge.
LOCAL_MODEL_ID = "google/flan-t5-large"

# --- USER REQUEST: Set TOP_K to 4 ---
TOP_K = 4

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,  # LangChain HuggingFacePipeline wrapper (if used)
    "rag_chain": None, # We will bypass this, but keep it as a potential fallback
    "memory": None, # Global memory object
    # optional direct pipeline store (not required)
    "hf_pipeline": None
}

# ---------------- Chat prompt setup ----------------
# --- NEW, SIMPLER PROMPT ---
# The complex logic is now in the Python code, not the prompt.
# This prompt just tells the model what info it has.
text_template = """You are a helpful cybersecurity expert assistant.
Answer the user's 'Question' using your general knowledge and the provided 'Chat History'.
If 'Context from database' is provided and relevant, use it to answer data-specific questions.

Chat History:
{chat_history}

Context from database:
{context}

Question: {question}

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
            try:
                from langchain.docstore.document import Document
            except Exception:
                from langchain_core.documents import Document
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
    """
    --- UPDATED FUNCTION ---
    Uses the user-provided column list to create natural language sentences
    for embedding, instead of just raw key:value pairs.
    """
    if df is None or vs is None:
        return False, "no_df_or_vs"

    # Use the text-heavy columns from the user's provided list
    # We'll skip Event ID, Timestamp, IPs for the *semantic text*
    USEFUL_COLUMNS = [
        "Attack Type",
        "Attack Severity",
        "Data Exfiltrated",
        "Threat Intelligence",
        "Response Action",
        "User Agent"
    ]
    
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    # Find which of our desired columns are *actually* in the CSV
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]
    
    if not cols_to_use:
        print("[INGEST] ERROR: Could not find expected text columns.")
        print(f"[INGEST] Available columns: {list(df.columns)}")
        return False, "missing_expected_columns"

    print(f"[INGEST] Using text columns for embedding: {cols_to_use}")

    def create_doc(row):
        # Create a natural language sentence from the row data
        parts = ["A cybersecurity event was recorded"]
        
        # Add key info
        if "Attack Type" in cols_to_use and pd.notna(row["Attack Type"]):
            parts.append(f"Attack Type: {row['Attack Type']}")
        if "Attack Severity" in cols_to_use and pd.notna(row["Attack Severity"]):
            parts.append(f"Severity: {row['Attack Severity']}")
        
        # Add other details
        if "Threat Intelligence" in cols_to_use and pd.notna(row["Threat Intelligence"]):
            parts.append(f"Threat Intelligence: {row['Threat Intelligence']}")
        if "Response Action" in cols_to_use and pd.notna(row["Response Action"]):
            parts.append(f"Response: {row['Response Action']}")
        if "Data Exfiltrated" in cols_to_use and pd.notna(row["Data Exfiltrated"]):
            parts.append(f"Data Exfiltrated: {row['Data Exfiltrated']}")
        if "User Agent" in cols_to_use and pd.notna(row["User Agent"]):
             parts.append(f"User Agent: {row['User Agent']}")

        # Join all parts with a comma, except the first one
        if len(parts) > 1:
            return parts[0] + ": " + ", ".join(parts[1:]) + "."
        else:
            return "" # Skip if no useful data

    docs = df.apply(create_doc, axis=1).tolist()
    docs = [d for d in docs if d.strip()]
    if not docs:
        print("[INGEST] ERROR: No documents were created from the DataFrame.")
        return False, "no_documents_created"

    print(f"[INGEST] Created {len(docs)} clean documents")
    print(f"[INGEST] Sample document: {docs[0][:200]}...")

    total_added = 0
    method = None
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
    if HuggingFacePipeline is None:
        print("[LLM] HuggingFacePipeline not available.")
        return None
    if pipeline is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        print("[LLM] transformers not available.")
        return None
    try:
        print(f"[LLM] Loading local model: {LOCAL_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        if "t5" in LOCAL_MODEL_ID.lower():
            model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
            task = "text2text-generation"
        else:
            model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_ID)
            task = "text-generation"

        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            min_length=40,
            temperature=0.3,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            early_stopping=False
        )

        # Keep a direct hf pipeline as fallback too
        ml_models["hf_pipeline"] = pipe

        try:
            llm = HuggingFacePipeline(pipeline=pipe)
            print(f"[LLM] Local model wrapped in HuggingFacePipeline: {LOCAL_MODEL_ID}")
            return llm
        except Exception:
            print("[LLM] Could not wrap pipeline in HuggingFacePipeline; returning raw pipeline as fallback.")
            return pipe

    except Exception as e:
        print(f"[LLM] Failed to load local model: {e}")
        traceback.print_exc()
        return None

def create_rag_chain(llm, vs, memory_obj=None):
    # This chain will only be used as a fallback if our new logic fails
    if llm is None or vs is None:
        print("[RAG] Cannot create chain: llm or vs is None")
        return None
    if RetrievalQA is None:
        print("[RAG] RetrievalQA not available")
        return None
    try:
        prompt_obj = None
        
        if prompt_obj is None and PromptTemplate is not None:
            try:
                # Update prompt to include chat_history
                prompt_obj = PromptTemplate(template=text_template, input_variables=["chat_history", "context", "question"])
                print("[RAG] Using PromptTemplate for chain with memory.")
            except Exception:
                traceback.print_exc()
                prompt_obj = None

        chain_type_kwargs = {}
        if prompt_obj is not None:
            chain_type_kwargs["prompt"] = prompt_obj

        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            # --- USER REQUEST: Use TOP_K (which is 4) ---
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
            memory=memory_obj  # Pass the memory object here
        )
        print(f"[RAG] RAG chain created successfully with prompt, memory, and TOP_K={TOP_K}")
        return rag
    except Exception as e:
        print(f"[RAG] Failed to create chain: {e}")
        traceback.print_exc()
        return None

# Helper to format prompt when we need a plain string prompt (fallback)
def build_fallback_prompt(chat_history: str, context: str, question: str) -> str:
    if PromptTemplate is not None:
         try:
             prompt_obj = PromptTemplate(template=text_template, input_variables=["chat_history", "context", "question"])
             return prompt_obj.format(chat_history=chat_history, context=context, question=question)
         except Exception:
             pass # Fall through to manual .format()
    try:
        # Fallback to direct string formatting
        return text_template.format(chat_history=chat_history, context=context, question=question)
    except Exception:
        # Absolute last resort
        return f"Chat History: {chat_history}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"


# Robust sender to LLM: handle various wrapper interfaces
def send_to_llm(llm_obj, prompt_text: str) -> str:
    try:
        # LangChain HuggingFacePipeline wrapper may have .invoke
        if hasattr(llm_obj, "invoke"):
            out = llm_obj.invoke(prompt_text)
            if isinstance(out, dict):
                return out.get("result") or out.get("generated_text") or str(out)
            return str(out)
        # LangChain wrappers may implement __call__
        if hasattr(llm_obj, "__call__"):
            out = llm_obj(prompt_text)
            # out may be dict or string
            if isinstance(out, dict):
                return out.get("result") or out.get("generated_text") or str(out)
            return str(out)
        # If it's a raw transformers pipeline (callable), call and extract
        if callable(llm_obj):
            out = llm_obj(prompt_text)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    # common keys: 'generated_text' or 'summary_text' or 'translation_text'
                    return first.get("generated_text") or first.get("translation_text") or first.get("summary_text") or str(first)
                return str(first)
            return str(out)
    except Exception:
        traceback.print_exc()
    # final fallback
    return ""

# ---------------- Lifespan & app ----------------
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
            cnt = vs_count_estimate(vs)
            print(f"[LIFESPAN] vector store estimated count: {cnt}")
            if cnt == 0:
                print("[LIFESPAN] Auto-ingesting CSV (may take time)...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None and len(df) > 0:
                    # --- Use new ingest function ---
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] ingest result:", ok, info)
        except Exception:
            traceback.print_exc()
    else:
        print("[LIFESPAN] Skipping auto-ingest (no vs or no csv)")

    print("[LIFESPAN] Loading local LLM (this may take a minute)...")
    ml_models["local_llm"] = create_local_llm()

    # Create global memory object
    if ConversationBufferMemory is not None:
        ml_models["memory"] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=False, # Return history as a string, best for T5
            input_key="question" # Explicitly tell memory what the input key is
        )
        print("[LIFESPAN] Global ConversationBufferMemory created.")
    else:
        print("[LIFESPAN] ConversationBufferMemory not available, chain will be stateless.")

    # We still create the RAG chain as a *backup*
    if ml_models["local_llm"] is not None and vs is not None:
        ml_models["rag_chain"] = create_rag_chain(
            ml_models["local_llm"], 
            vs, 
            ml_models["memory"] # Pass memory to chain
        )

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs is not None else 0,
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "memory_ready": ml_models.get("memory") is not None,
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
        "memory_ready": ml_models.get("memory") is not None,
        "model_id": LOCAL_MODEL_ID,
        "top_k": TOP_K, # Report the new TOP_K
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

    try:
        print("[INGEST] Clearing old data from vector store...")
        if hasattr(vs, "_collection"):
            vs._collection.delete(ids=vs._collection.get()['ids'])
            print("[INGEST] Old data cleared.")
        else:
            print("[INGEST] Could not automatically clear old data. Re-ingesting anyway.")
    except Exception:
        traceback.print_exc()

    # --- Use new ingest function ---
    ok, info = ingest_dataframe(df, vs, batch_docs=batch_size)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {info}")

    final_count = vs_count_estimate(vs)
    return {"status": "ingested", "method_info": info, "final_count": int(final_count)}

@app.get("/get-csv-columns")
def get_csv_columns():
    csv_path = find_csv_path()
    if csv_path is None:
        raise HTTPException(status_code=404, detail="CSV not found.")
    df = load_dataframe_from_csv(csv_path)
    if df is None:
        raise HTTPException(status_code=400, detail="CSV empty or unreadable.")
    # Return the exact columns from the file
    return {"columns": list(df.columns)}

@app.post("/reset-memory")
def reset_memory():
    """
    Clears the global conversation history.
    """
    memory = ml_models.get("memory")
    if memory is not None:
        try:
            memory.clear()
            print("[MEMORY] Global memory cleared.")
            return {"status": "memory_cleared"}
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to clear memory: {e}")
    return {"status": "no_memory_object_to_clear"}


@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    """
    --- NEW LOGIC ---
    This function now implements conditional logic to prevent
    context-copying for general questions.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    vs = ml_models.get("vector_store")
    local_llm = ml_models.get("local_llm") or ml_models.get("hf_pipeline")
    memory = ml_models.get("memory")

    if local_llm is None:
        raise HTTPException(status_code=503, detail="No local LLM available.")

    query_str_lower = req.query.strip().lower()
    context_text = ""
    retrieved_docs = []

    # --- NEW CONDITIONAL LOGIC ---
    
    # 1. Define general question triggers
    general_question_triggers = [
        "what is a ", "what is ", "what's a ", "what's ",
        "what are ", "what are ", "define ", "explain ", "how does"
    ]
    is_general_question = any(query_str_lower.startswith(trigger) for trigger in general_question_triggers)

    # 2. If it's NOT a general question, retrieve context.
    if not is_general_question:
        if vs is not None:
            try:
                print(f"[LOGIC] Data question detected. Retrieving TOP_K={TOP_K} docs...")
                retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
                
                if hasattr(retriever, "get_relevant_documents"):
                    retrieved_docs = retriever.get_relevant_documents(req.query)
                elif hasattr(retriever, "retrieve"):
                    retrieved_docs = retriever.retrieve(req.query)
                else:
                    retrieved_docs = retriever(req.query) if callable(retriever) else []
                
                if retrieved_docs:
                    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in retrieved_docs])
                    print(f"[LOGIC] Found {len(retrieved_docs)} relevant docs.")
                else:
                    context_text = "(No relevant data context found)"
            except Exception as e:
                traceback.print_exc()
                context_text = "(Error during context retrieval)"
        else:
            context_text = "(Vector store not available)"
    else:
        print("[LOGIC] General question detected. Forcing empty context.")
        context_text = "(No context needed for general question)"

    # 3. Load chat history
    chat_history = ""
    if memory is not None:
        try:
            chat_history = memory.load_memory_variables({}).get("chat_history", "")
        except Exception as e:
            print(f"[MEMORY] Error loading memory: {e}")

    # 4. Build the final prompt
    formatted_prompt = build_fallback_prompt(chat_history, context_text, req.query)

    # 5. Send to LLM
    generated = send_to_llm(local_llm, formatted_prompt)
    if not generated:
        raise HTTPException(status_code=500, detail="LLM produced no output.")

    # 6. Save to memory
    if memory is not None:
        try:
            memory.save_context({"question": req.query}, {"answer": generated})
            print("[MEMORY] Context saved to memory.")
        except Exception as e:
            print(f"[MEMORY] Error saving memory: {e}")

    generated = generated.strip()
    return QueryResponse(answer=generated)

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")

