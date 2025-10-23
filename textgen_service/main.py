"""
Robust RAG service with LOCAL MODEL support
FIXED: Memory system, response formatting, and T5-specific prompt format
"""

import os
import traceback
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Try imports (be tolerant across environments)
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
LOCAL_MODEL_ID = "google/flan-t5-large"
TOP_K = 4

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,
    "rag_chain": None,
    "memory": None,
    "hf_pipeline": None,
    "conversation_history": []  # Manual conversation tracking
}

# ---------------- CRITICAL FIX: T5-SPECIFIC PROMPT FORMAT ----------------
# T5 models need task prefix and specific formatting
# Format: "answer the question: context: <context> question: <question>"
text_template = """answer the question: context: {context} question: {question}"""

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

def _try_add_texts(vs, texts: List[str]):
    try:
        if hasattr(vs, "add_texts"):
            vs.add_texts(texts=texts)
            return True, "add_texts"
    except Exception:
        traceback.print_exc()
    try:
        if hasattr(vs, "add_documents"):
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
                ids = [f"doc_{i}" for i in range(len(texts))]
                col.add(documents=texts, metadatas=[{}]*len(texts), ids=ids)
            except Exception:
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

    USEFUL_COLUMNS = [
        "Attack Type",
        "Attack Severity",
        "Data Exfiltrated",
        "Threat Intelligence",
        "Response Action",
        "User Agent"
    ]

    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]

    if not cols_to_use:
        print("[INGEST] ERROR: Could not find expected text columns.")
        print(f"[INGEST] Available columns: {list(df.columns)}")
        return False, "missing_expected_columns"

    print(f"[INGEST] Using text columns for embedding: {cols_to_use}")

    def create_doc(row):
        parts = []
        
        if "Attack Type" in cols_to_use and pd.notna(row.get("Attack Type")):
            parts.append(f"Attack Type: {row['Attack Type']}")
        if "Attack Severity" in cols_to_use and pd.notna(row.get("Attack Severity")):
            parts.append(f"Attack Severity: {row['Attack Severity']}")
        if "Threat Intelligence" in cols_to_use and pd.notna(row.get("Threat Intelligence")):
            parts.append(f"Threat Intelligence: {row['Threat Intelligence']}")
        if "Response Action" in cols_to_use and pd.notna(row.get("Response Action")):
            parts.append(f"Response Action: {row['Response Action']}")
        if "Data Exfiltrated" in cols_to_use and pd.notna(row.get("Data Exfiltrated")):
            parts.append(f"Data Exfiltrated: {row['Data Exfiltrated']}")
        if "User Agent" in cols_to_use and pd.notna(row.get("User Agent")):
             parts.append(f"User Agent: {row['User Agent']}")

        if parts:
            return "Cybersecurity event: " + ", ".join(parts) + "."
        else:
            return ""

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

        # CRITICAL: Optimized parameters for T5 to follow context
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,  # Reduced for more focused answers
            min_new_tokens=10,
            do_sample=False,  # CRITICAL: Greedy decoding for factual accuracy
            temperature=1.0,  # Not used when do_sample=False
            num_beams=4,  # Beam search for better quality
            early_stopping=True,  # Stop when EOS is generated
            repetition_penalty=1.2,  # Prevent repetition
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )

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
    # NOTE: We won't use RetrievalQA with memory for T5
    # Instead, we'll manually handle retrieval + generation
    # This is because T5 needs specific prompt formatting
    return None

def build_t5_prompt(context: str, question: str, chat_history: str = "") -> str:
    """
    Build T5-specific prompt with conversation history incorporated into context
    T5 expects: "answer the question: context: <context> question: <question>"
    """
    # If there's chat history, incorporate it into the context
    if chat_history and chat_history.strip():
        full_context = f"Previous conversation:\n{chat_history}\n\nRelevant information:\n{context}"
    else:
        full_context = context
    
    # T5-specific format
    prompt = f"answer the question: context: {full_context} question: {question}"
    return prompt

def send_to_llm(llm_obj, prompt_text: str) -> str:
    """Send prompt to LLM and extract generated text only"""
    try:
        # For HuggingFacePipeline
        if hasattr(llm_obj, "invoke"):
            out = llm_obj.invoke(prompt_text)
            if isinstance(out, dict):
                return out.get("result") or out.get("generated_text") or str(out)
            return str(out)
        
        # For raw pipeline
        if hasattr(llm_obj, "__call__"):
            out = llm_obj(prompt_text)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    generated = first.get("generated_text", "")
                    # CRITICAL: Remove the input prompt from output
                    if generated.startswith(prompt_text):
                        generated = generated[len(prompt_text):].strip()
                    return generated
                return str(first)
            if isinstance(out, dict):
                return out.get("result") or out.get("generated_text") or str(out)
            return str(out)
        
        if callable(llm_obj):
            out = llm_obj(prompt_text)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    generated = first.get("generated_text", "")
                    # CRITICAL: Remove the input prompt from output
                    if generated.startswith(prompt_text):
                        generated = generated[len(prompt_text):].strip()
                    return generated
                return str(first)
            return str(out)
    except Exception:
        traceback.print_exc()
    return ""

def format_conversation_history(history: List[dict], max_turns: int = 3) -> str:
    """Format conversation history for inclusion in context"""
    if not history:
        return ""
    
    # Keep only last N turns to avoid context overflow
    recent_history = history[-max_turns:] if len(history) > max_turns else history
    
    formatted = []
    for turn in recent_history:
        formatted.append(f"User: {turn['question']}")
        formatted.append(f"Assistant: {turn['answer']}")
    
    return "\n".join(formatted)

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
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] ingest result:", ok, info)
        except Exception:
            traceback.print_exc()
    else:
        print("[LIFESPAN] Skipping auto-ingest (no vs or no csv)")

    print("[LIFESPAN] Loading local LLM (this may take a minute)...")
    ml_models["local_llm"] = create_local_llm()

    # Initialize conversation history list
    ml_models["conversation_history"] = []
    print("[LIFESPAN] Conversation history initialized.")

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": vs is not None,
        "vector_store_count": vs_count_estimate(vs) if vs is not None else 0,
        "local_llm_ready": ml_models.get("local_llm") is not None,
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
    history = ml_models.get("conversation_history", [])
    
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": count > 0,
        "vector_store_count": int(count),
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "conversation_turns": len(history),
        "model_id": LOCAL_MODEL_ID,
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
            ids_to_delete = vs._collection.get()['ids']
            if ids_to_delete:
                vs._collection.delete(ids=ids_to_delete)
                print("[INGEST] Old data cleared.")
            else:
                print("[INGEST] Collection was already empty.")
        else:
            print("[INGEST] Could not automatically clear old data. Re-ingesting anyway.")
    except Exception:
        traceback.print_exc()
        print("[INGEST] Error clearing data, re-ingesting may result in duplicates.")

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
    return {"columns": list(df.columns)}

@app.post("/reset-memory")
def reset_memory():
    ml_models["conversation_history"] = []
    print("[MEMORY] Conversation history cleared.")
    return {"status": "memory_cleared"}

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    vs = ml_models.get("vector_store")
    local_llm = ml_models.get("local_llm") or ml_models.get("hf_pipeline")
    conversation_history = ml_models.get("conversation_history", [])

    if local_llm is None:
        raise HTTPException(status_code=503, detail="No local LLM available.")

    # ---------------- STEP 1: Retrieve Relevant Context ----------------
    context_text = ""
    retrieved_docs = []
    
    if vs is not None:
        try:
            print(f"[RETRIEVAL] Retrieving TOP_K={TOP_K} docs for query: '{req.query}'")
            retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
            
            if hasattr(retriever, "get_relevant_documents"):
                retrieved_docs = retriever.get_relevant_documents(req.query)
            elif hasattr(retriever, "retrieve"):
                retrieved_docs = retriever.retrieve(req.query)
            else:
                retrieved_docs = retriever(req.query) if callable(retriever) else []
            
            if retrieved_docs:
                context_text = "\n".join([getattr(d, "page_content", str(d)) for d in retrieved_docs])
                print(f"[RETRIEVAL] Found {len(retrieved_docs)} relevant docs")
                print(f"[RETRIEVAL] Context sample: {context_text[:200]}...")
            else:
                context_text = "No relevant cybersecurity information found in database."
                print("[RETRIEVAL] No relevant documents found")
        except Exception as e:
            traceback.print_exc()
            context_text = "Error retrieving context from database."
            print(f"[RETRIEVAL] Error: {e}")
    else:
        context_text = "Vector store not available."

    # ---------------- STEP 2: Format Conversation History ----------------
    chat_history = format_conversation_history(conversation_history, max_turns=2)
    print(f"[HISTORY] Formatted chat history: '{chat_history[:200]}...'")

    # ---------------- STEP 3: Build T5-Specific Prompt ----------------
    formatted_prompt = build_t5_prompt(context_text, req.query, chat_history)
    print(f"[PROMPT] T5 prompt (first 300 chars): {formatted_prompt[:300]}...")

    # ---------------- STEP 4: Generate Answer ----------------
    generated = send_to_llm(local_llm, formatted_prompt)
    
    if not generated or not generated.strip():
        raise HTTPException(status_code=500, detail="LLM produced no output.")

    # ---------------- STEP 5: Clean Up Response ----------------
    generated = generated.strip()
    
    # Remove any remaining prompt artifacts
    prefixes_to_remove = [
        "answer the question:",
        "context:",
        "question:",
        "Answer:",
        "Response:",
        "Explanation:",
        "Summary:",
        req.query  # Remove if model echoed the question
    ]
    
    for prefix in prefixes_to_remove:
        if generated.lower().startswith(prefix.lower()):
            generated = generated[len(prefix):].strip()
            generated = generated.lstrip(':').strip()
    
    # Remove leading/trailing quotes
    generated = generated.strip('"\'')
    
    # If answer is too short or generic, try to extract from context
    if len(generated.split()) < 3 or generated.lower() in ["yes", "no", "unknown"]:
        print(f"[WARNING] Generated answer too short: '{generated}'")
        # For very short answers, we keep them as they might be correct
    
    print(f"[RESPONSE] Final answer: {generated}")

    # ---------------- STEP 6: Save to Conversation History ----------------
    conversation_history.append({
        "question": req.query,
        "answer": generated,
        "context": context_text[:500]  # Store limited context for reference
    })
    
    # Keep only last 10 turns to prevent memory overflow
    if len(conversation_history) > 10:
        conversation_history.pop(0)
    
    print(f"[MEMORY] Saved turn. Total turns: {len(conversation_history)}")

    return QueryResponse(answer=generated)

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")
