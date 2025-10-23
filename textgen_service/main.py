"""
Robust RAG service with LOCAL MODEL support
FIXED: Memory system and response formatting
This version correctly uses the RetrievalQA chain for all queries,
ensuring that memory is properly read and updated.
"""

import os
import traceback
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
# This is the path you provided for the dataset
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
    "hf_pipeline": None
}

# ---------------- IMPROVED PROMPT WITH MEMORY AWARENESS & PERSONA ----------------
text_template = """You are "CyberGuard Sentinel," a helpful and conversational cybersecurity expert assistant.
You are having a conversation with a user. Use the following pieces of information to answer the user's current question.

1. Previous Conversation (Chat History):
{chat_history}

2. Relevant Knowledge (Context from database):
{context}

3. Current Question:
{query}

Instructions:
- Answer the "Current Question" based on all the information provided.
- If the question is about the "Previous Conversation," use that information to answer.
- If the question is related to cybersecurity, use the "Relevant Knowledge" to provide a detailed answer.
- If the "Relevant Knowledge" is not relevant to the question, or if the question is a general knowledge question (e.g., "what is 2+2?"), answer the question directly.
- If you don't know the answer, just say "I'm not sure about that."
- Be friendly and conversational.

Helpful Answer:"""

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
        print(f"[EMB] Embedding initialized: {EMBEDDING_MODEL_NAME}")
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
        print(f"[CHROMA] Initialized with persist_directory: {CHROMA_DB_PATH}")
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
                # Create simple unique IDs for this batch
                start_id = vs_count_estimate(vs)
                ids = [f"doc_{start_id + i}" for i in range(len(texts))]
                col.add(documents=texts, metadatas=[{}]*len(texts), ids=ids)
            except Exception:
                col.add(documents=texts) # Fallback if IDs fail
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

    # Use the exact columns from the user's dataset
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

    # Map original column names for the row accessor
    col_map = {c: df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower}

    def create_doc(row):
        parts = ["A cybersecurity event was recorded"]
        
        if "Attack Type" in col_map and pd.notna(row[col_map["Attack Type"]]):
            parts.append(f"Attack Type: {row[col_map['Attack Type']]}")
        if "Attack Severity" in col_map and pd.notna(row[col_map["Attack Severity"]]):
            parts.append(f"Severity: {row[col_map['Attack Severity']]}")
        if "Threat Intelligence" in col_map and pd.notna(row[col_map["Threat Intelligence"]]):
            parts.append(f"Threat Intelligence: {row[col_map['Threat Intelligence']]}")
        if "Response Action" in col_map and pd.notna(row[col_map["Response Action"]]):
            parts.append(f"Response: {row[col_map['Response Action']]}")
        if "Data Exfiltrated" in col_map and pd.notna(row[col_map["Data Exfiltrated"]]):
            parts.append(f"Data Exfiltrated: {row[col_map['Data Exfiltrated']]}")
        if "User Agent" in col_map and pd.notna(row[col_map["User Agent"]]):
            parts.append(f"User Agent: {row[col_map['User Agent']]}")

        if len(parts) > 1:
            return parts[0] + ": " + ", ".join(parts[1:]) + "."
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
        print(f"[INGEST] Adding batch {i // batch_docs + 1} of {len(docs) // batch_docs + 1} (size: {len(batch)})")
        ok, method = _try_add_texts(vs, batch)
        if not ok:
            return False, f"batch_failed_at_{i}"
        total_added += len(batch)

    try:
        if hasattr(vs, "persist"):
            print("[INGEST] Persisting vector store...")
            vs.persist()
            print("[INGEST] Persist complete.")
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
            max_new_tokens=300,
            min_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            early_stopping=False,
            num_beams=3,
            length_penalty=1.0,
            no_repeat_ngram_size=2
        )

        ml_models["hf_pipeline"] = pipe

        try:
            llm = HuggingFacePipeline(pipeline=pipe)
            print(f"[LLM] Local model wrapped in HuggingFacePipeline: {LOCAL_MODEL_ID}")
            return llm
        except Exception:
            print("[LLM] Could not wrap pipeline in HuggingFacePipeline; returning raw pipeline as fallback.")
            return pipe # Return the raw pipeline, which the chain can often still use

    except Exception as e:
        print(f"[LLM] Failed to load local model: {e}")
        traceback.print_exc()
        return None

def create_rag_chain(llm, vs, memory_obj=None):
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
                # Use "query" to match the RetrievalQA input key
                prompt_obj = PromptTemplate(template=text_template, input_variables=["chat_history", "context", "query"])
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
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
            memory=memory_obj,
            input_key="query" # Explicitly tell the chain which input key is the question
        )
        print(f"[RAG] RAG chain created successfully with prompt, memory, and TOP_K={TOP_K}")
        return rag
    except Exception as e:
        print(f"[RAG] Failed to create chain: {e}")
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
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] ingest result:", ok, info)
                else:
                    print("[LIFESPAN] Failed to load or empty dataframe.")
            else:
                print("[LIFESPAN] Vector store already has data, skipping auto-ingest.")
        except Exception:
            traceback.print_exc()
    else:
        print("[LIFESPAN] Skipping auto-ingest (no vs or no csv)")

    print("[LIFESPAN] Loading local LLM (this may take a minute)...")
    ml_models["local_llm"] = create_local_llm()

    # Create global memory object with proper configuration
    if ConversationBufferMemory is not None:
        ml_models["memory"] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=False,
            input_key="query",  # Match RetrievalQA input
            output_key="result" # Match RetrievalQA output
        )
        print("[LIFESPAN] Global ConversationBufferMemory created (input='query', output='result').")
    else:
        print("[LIFESPAN] ConversationBufferMemory not available, chain will be stateless.")

    if ml_models["local_llm"] is not None and vs is not None:
        ml_models["rag_chain"] = create_rag_chain(
            ml_models["local_llm"], 
            vs, 
            ml_models["memory"]
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
    source_documents: Optional[List[dict]] = None

@app.get("/")
def root():
    return {"status": "CyberGuard Sentinel (RAG) Service running with LOCAL models."}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    count = vs_count_estimate(vs) if vs is not None else 0
    memory = ml_models.get("memory")
    memory_content = ""
    if memory is not None:
        try:
            memory_data = memory.load_memory_variables({})
            memory_content = memory_data.get("chat_history", "")
        except Exception:
            pass
            
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": count > 0,
        "vector_store_count": int(count),
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "memory_ready": ml_models.get("memory") is not None,
        "memory_content_preview": memory_content[:200] + "..." if len(memory_content) > 200 else memory_content,
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
        raise HTTPException(status_code=404, detail=f"CSV not found. Looked for {DEFAULT_CSV_BASENAME} and in {HARDCODED_CSV_PATH}")
    
    df = load_dataframe_from_csv(csv_path)
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="CSV empty or unreadable.")
    
    if sample_limit is not None:
        df = df.head(sample_limit)

    try:
        print("[INGEST] Clearing old data from vector store...")
        if hasattr(vs, "_collection"):
            count = vs._collection.count()
            if count > 0:
                # Fetch all IDs to delete. This can be slow for large dbs.
                ids_to_delete = vs._collection.get(limit=count, include=[])['ids']
                if ids_to_delete:
                    vs._collection.delete(ids=ids_to_delete)
                    print(f"[INGEST] Old data cleared ({len(ids_to_delete)} items).")
                else:
                    print("[INGEST] Collection reported count > 0 but get() returned no IDs.")
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
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_chain = ml_models.get("rag_chain")
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is not available. Check model and vector store status.")

    try:
        print(f"[LOGIC] Invoking RAG chain for query: '{req.query}'")
        
        # The RAG chain now automatically handles:
        # 1. Getting relevant docs from vector_store (for "context")
        # 2. Loading chat history from memory (for "chat_history")
        # 3. Formatting the prompt
        # 4. Calling the LLM
        # 5. Saving the new {query} and {result} to memory
        
        response = rag_chain.invoke({"query": req.query})
        
        generated = response.get("result", "").strip()
        source_docs = response.get("source_documents", [])
        
        print(f"[LOGIC] RAG chain returned. Answer (raw): '{generated[:100]}...'")
        
        if not generated:
            raise HTTPException(status_code=500, detail="LLM produced no output.")

        # Clean up the response
        prefixes_to_remove = ["Response:", "Answer:", "Explanation:", "Summary:", "Helpful Answer:"]
        for prefix in prefixes_to_remove:
            if generated.lower().startswith(prefix.lower()):
                generated = generated[len(prefix):].strip()
                break
        
        # Remove any leading/trailing quotes
        generated = generated.strip('"\'')
        
        print(f"[RESPONSE] Cleaned response: {generated}")

        # Convert Document objects to simple dicts for JSON response
        source_list = []
        for doc in source_docs:
            source_list.append({
                "content": getattr(doc, "page_content", str(doc)),
                "metadata": getattr(doc, "metadata", {})
            })
            
        # Memory is AUTOMATICALLY updated by the chain, no need for manual save_context

        return QueryResponse(answer=generated, source_documents=source_list)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during chain invocation: {e}")

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002 --reload")


