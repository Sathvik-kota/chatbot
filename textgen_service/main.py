"""
Robust RAG service with OpenAI GPT-4o-mini
Uses ChatPromptTemplate and a ChatOpenAI model to provide
a robust, conversational RAG experience.

Key Updates:
- MODEL UPGRADE: Switched from flan-t5-large to gpt-4o-mini.
  This requires the OPENAI_API_KEY environment variable to be set.
- PROMPT: Replaced string PromptTemplate with ChatPromptTemplate
  to use System/Human roles.
- MEMORY: Updated ConversationBufferMemory to return_messages=True.
- LOGIC: Removed RetrievalQA chain. The /generate-text endpoint
  now manually builds the message list (with conditional context)
  and calls the LLM directly for more control and accuracy.
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
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except Exception:
    ChatPromptTemplate = None
    MessagesPlaceholder = None

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
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
except Exception:
    SystemMessage = None
    HumanMessage = None
    AIMessage = None


# ---------------- Config ----------------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
HARDCODED_CSV_PATH = "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv"
DEFAULT_CSV_BASENAME = "ai_cybersecurity_dataset-sampled-5k.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# --- MODEL UPGRADE ---
OPENAI_MODEL_NAME = "gpt-4o-mini"

# --- USER REQUEST: Set TOP_K to 4 ---
TOP_K = 4

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "llm": None,  # This will now be the ChatOpenAI model
    "memory": None, # Global memory object
}
chat_prompt_obj = None # Will be initialized in lifespan

# ---------------- Chat prompt setup ----------------
if ChatPromptTemplate and MessagesPlaceholder:
    chat_prompt_obj = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful cybersecurity expert assistant. Answer the user's 'Question' using your general knowledge and the provided 'Chat History'. If 'Context from database' is provided and relevant, use it to answer data-specific questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Context from database:\n{context}\n\nQuestion:\n{question}")
    ])
else:
    print("[PROMPT] CRITICAL: langchain_core.prompts not fully loaded. Prompts will fail.")

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

def create_llm():
    """
    Creates the ChatOpenAI LLM instance.
    Requires OPENAI_API_KEY to be set as an environment variable.
    """
    if ChatOpenAI is None:
        print("[LLM] langchain_openai not available.")
        return None
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] CRITICAL ERROR: OPENAI_API_KEY environment variable not set.")
        # We return None, and the lifespan/endpoints will handle this
        return None
        
    try:
        llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0.3)
        print(f"[LLM] OpenAI model initialized: {OPENAI_MODEL_NAME}")
        return llm
    except Exception as e:
        print(f"[LLM] Failed to load OpenAI model: {e}")
        traceback.print_exc()
        return None

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

    print("[LIFESPAN] Initializing OpenAI LLM...")
    ml_models["llm"] = create_llm()
    if ml_models["llm"] is None:
        print("[LIFESPAN] CRITICAL: LLM failed to initialize. Check OPENAI_API_KEY.")

    # Create global memory object
    if ConversationBufferMemory is not None:
        ml_models["memory"] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True, # MUST be True for Chat Models
            input_key="question"
        )
        print("[LIFESPAN] Global ConversationBufferMemory created (return_messages=True).")
    else:
        print("[LIFESPAN] ConversationBufferMemory not available, chain will be stateless.")

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs is not None else 0,
        "llm_ready": ml_models.get("llm") is not None,
        "rag_chain_ready": False, # We are using manual logic, not a chain
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
    return {"status": "Text Generation (RAG) Service running with OpenAI models."}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    count = vs_count_estimate(vs) if vs is not None else 0
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": count > 0,
        "vector_store_count": int(count),
        "llm_ready": ml_models.get("llm") is not None,
        "rag_chain_ready": False, # We are using manual logic
        "memory_ready": ml_models.get("memory") is not None,
        "model_id": OPENAI_MODEL_NAME,
        "top_k": TOP_K,
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
    This function now uses ChatOpenAI and builds a list of messages.
    It still uses the conditional logic to prevent context-copying.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    vs = ml_models.get("vector_store")
    llm = ml_models.get("llm")
    memory = ml_models.get("memory")

    if llm is None:
        raise HTTPException(status_code=503, detail="OpenAI LLM not available. Check server logs and OPENAI_API_KEY.")

    query_str_lower = req.query.strip().lower()
    context_text = ""
    retrieved_docs = []

    # --- CONDITIONAL LOGIC ---
    
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

    # 3. Load chat history (as messages)
    chat_history_messages = []
    if memory is not None:
        try:
            # This is how you get messages from ConversationBufferMemory
            chat_history_messages = memory.chat_memory.messages
        except Exception as e:
            print(f"[MEMORY] Error loading memory: {e}")

    # 4. Build the final prompt as a list of messages
    final_messages = []
    try:
        # Use the global ChatPromptTemplate
        final_prompt_value = chat_prompt_obj.invoke({
            "chat_history": chat_history_messages,
            "context": context_text,
            "question": req.query
        })
        final_messages = final_prompt_value.to_messages()
    except Exception as e:
        print(f"[PROMPT] Error formatting ChatPromptTemplate: {e}. Building manually.")
        # Manual fallback
        if SystemMessage:
            final_messages.append(SystemMessage(content="You are a helpful cybersecurity expert assistant. Answer the user's 'Question' using your general knowledge and the provided 'Chat History'. If 'Context from database' is provided and relevant, use it to answer data-specific questions."))
        
        final_messages.extend(chat_history_messages)
        
        if HumanMessage:
            final_messages.append(HumanMessage(content=f"Context from database:\n{context_text}\n\nQuestion:\n{req.query}"))

    if not final_messages:
        raise HTTPException(status_code=500, detail="Failed to build prompt messages.")

    # 5. Send to LLM
    try:
        print(f"[LLM] Invoking {OPENAI_MODEL_NAME} with {len(final_messages)} messages.")
        response = llm.invoke(final_messages)
        generated_text = response.content
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM failed to generate a response: {e}")

    if not generated_text:
        raise HTTPException(status_code=500, detail="LLM produced no output.")

    # 6. Save to memory
    if memory is not None:
        try:
            # We save the *actual* query and response
            memory.save_context({"question": req.query}, {"answer": generated_text})
            print("[MEMORY] Context saved to memory.")
        except Exception as e:
            print(f"[MEMORY] Error saving memory: {e}")

    generated_text = generated_text.strip()
    return QueryResponse(answer=generated_text)

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")

