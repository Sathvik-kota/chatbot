"""
FINAL WORKING RAG SERVICE with PROPER CONVERSATIONAL MEMORY
Based on proven LangChain ConversationalRetrievalChain patterns
"""

import os
import traceback
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Try imports
try:
    from langchain.prompts import PromptTemplate
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    PromptTemplate = None
    ChatPromptTemplate = None

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
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    try:
        from langchain_community.chains import ConversationalRetrievalChain
    except Exception:
        ConversationalRetrievalChain = None

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
TOP_K = 5

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,
    "rag_chain": None,
    "chat_history": []  # Store as list of tuples (question, answer)
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
            attack_type = row['Attack Type']
            parts.append(f"This cybersecurity event involves a {attack_type} attack.")
        
        if "Attack Severity" in cols_to_use and pd.notna(row.get("Attack Severity")):
            parts.append(f"The severity level is {row['Attack Severity']}.")
        
        if "Threat Intelligence" in cols_to_use and pd.notna(row.get("Threat Intelligence")):
            parts.append(f"Threat intelligence: {row['Threat Intelligence']}")
        
        if "Response Action" in cols_to_use and pd.notna(row.get("Response Action")):
            parts.append(f"The response action taken was: {row['Response Action']}.")
        
        if "Data Exfiltrated" in cols_to_use and pd.notna(row.get("Data Exfiltrated")):
            parts.append(f"Data exfiltration occurred: {row['Data Exfiltrated']}.")
        
        if "User Agent" in cols_to_use and pd.notna(row.get("User Agent")):
            parts.append(f"User agent: {row['User Agent']}")

        if parts:
            return " ".join(parts)
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
    if pipeline is None or AutoTokenizer is None:
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
            max_length=512,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.3,
            num_beams=2
        )

        try:
            llm = HuggingFacePipeline(pipeline=pipe)
            print(f"[LLM] Local model wrapped: {LOCAL_MODEL_ID}")
            return llm
        except Exception:
            print("[LLM] Returning raw pipeline.")
            return pipe

    except Exception as e:
        print(f"[LLM] Failed to load local model: {e}")
        traceback.print_exc()
        return None

def create_conversational_chain(llm, vs):
    """Create ConversationalRetrievalChain - the PROPER way to handle conversational RAG"""
    if llm is None or vs is None or ConversationalRetrievalChain is None:
        print("[CHAIN] Cannot create conversational chain")
        return None
    
    try:
        # QA Prompt - for answering based on context
        qa_template = """You are a helpful cybersecurity assistant. Use the following context to answer the question. 
If you don't know the answer based on the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Helpful Answer:"""
        
        QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            verbose=True
        )
        
        print("[CHAIN] ConversationalRetrievalChain created successfully")
        return chain
        
    except Exception as e:
        print(f"[CHAIN] Error creating chain: {e}")
        traceback.print_exc()
        return None

# ---------------- Lifespan & app ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[LIFESPAN] starting up...")
    ml_models["embedding_function"] = create_embedding_function()
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    vs = ml_models["vector_store"]

    csv_path = find_csv_path()
    if csv_path:
        print(f"[LIFESPAN] Found CSV at: {csv_path}")
    else:
        print("[LIFESPAN] No CSV found.")

    if vs is not None and csv_path:
        try:
            cnt = vs_count_estimate(vs)
            print(f"[LIFESPAN] vector store count: {cnt}")
            if cnt == 0:
                print("[LIFESPAN] Auto-ingesting CSV...")
                df = load_dataframe_from_csv(csv_path)
                if df is not None and len(df) > 0:
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print(f"[LIFESPAN] ingest result: {ok}, {info}")
        except Exception:
            traceback.print_exc()

    print("[LIFESPAN] Loading local LLM...")
    ml_models["local_llm"] = create_local_llm()

    if ml_models["local_llm"] and vs:
        ml_models["rag_chain"] = create_conversational_chain(ml_models["local_llm"], vs)

    ml_models["chat_history"] = []

    print("[LIFESPAN] startup complete:", {
        "vector_store_ready": vs is not None,
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
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
    return {"status": "Conversational RAG Service running with LOCAL models."}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    count = vs_count_estimate(vs) if vs is not None else 0
    chat_history = ml_models.get("chat_history", [])
    
    return {
        "vector_store_ready": vs is not None,
        "vector_store_count": int(count),
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "conversation_turns": len(chat_history),
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
        if hasattr(vs, "_collection"):
            ids_to_delete = vs._collection.get()['ids']
            if ids_to_delete:
                vs._collection.delete(ids=ids_to_delete)
    except Exception:
        traceback.print_exc()

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
    ml_models["chat_history"] = []
    print("[MEMORY] Chat history cleared.")
    return {"status": "memory_cleared", "message": "Conversation history has been reset"}

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag_chain = ml_models.get("rag_chain")
    chat_history = ml_models.get("chat_history", [])

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not available.")

    try:
        print(f"\n[QUERY] User question: {req.query}")
        print(f"[HISTORY] Current history length: {len(chat_history)} turns")
        
        # ConversationalRetrievalChain expects chat_history as list of tuples
        result = rag_chain({
            "question": req.query,
            "chat_history": chat_history
        })
        
        answer = result.get("answer", "").strip()
        
        # Clean up answer
        if not answer:
            answer = "I'm sorry, I couldn't generate an answer based on the available information."
        
        # Remove any prompt artifacts
        for prefix in ["Helpful Answer:", "Answer:", "Response:", req.query]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                answer = answer.lstrip(':').strip()
        
        answer = answer.strip('"\'')
        
        print(f"[ANSWER] Generated: {answer}")
        
        # Update chat history - CRITICAL FORMAT: list of (question, answer) tuples
        chat_history.append((req.query, answer))
        
        # Keep only last 5 turns to avoid context overflow
        if len(chat_history) > 5:
            chat_history.pop(0)
        
        print(f"[HISTORY] Updated history length: {len(chat_history)} turns\n")
        
        return QueryResponse(answer=answer)
        
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002 --reload")
