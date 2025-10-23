"""
Robust RAG service with LOCAL MODEL support
FIXED: Memory system and response formatting
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
    "hf_pipeline": None
}

# ---------------- IMPROVED PROMPT WITH MEMORY AWARENESS ----------------
text_template = """You are a cybersecurity expert assistant having a conversation with a user.

Previous Conversation:
{chat_history}

Context from database:
{context}

Current Question: {question}

Provide a helpful, detailed answer. If the question references previous conversation, make sure to acknowledge it and connect your answer to what was discussed before.

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
        parts = ["A cybersecurity event was recorded"]
        
        if "Attack Type" in cols_to_use and pd.notna(row["Attack Type"]):
            parts.append(f"Attack Type: {row['Attack Type']}")
        if "Attack Severity" in cols_to_use and pd.notna(row["Attack Severity"]):
            parts.append(f"Severity: {row['Attack Severity']}")
        if "Threat Intelligence" in cols_to_use and pd.notna(row["Threat Intelligence"]):
            parts.append(f"Threat Intelligence: {row['Threat Intelligence']}")
        if "Response Action" in cols_to_use and pd.notna(row["Response Action"]):
            parts.append(f"Response: {row['Response Action']}")
        if "Data Exfiltrated" in cols_to_use and pd.notna(row["Data Exfiltrated"]):
            parts.append(f"Data Exfiltrated: {row['Data Exfiltrated']}")
        if "User Agent" in cols_to_use and pd.notna(row["User Agent"]):
             parts.append(f"User Agent: {row['User Agent']}")

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
            return pipe

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
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True,
            memory=memory_obj
        )
        print(f"[RAG] RAG chain created successfully with prompt, memory, and TOP_K={TOP_K}")
        return rag
    except Exception as e:
        print(f"[RAG] Failed to create chain: {e}")
        traceback.print_exc()
        return None

def build_fallback_prompt(chat_history: str, context: str, question: str) -> str:
    if PromptTemplate is not None:
         try:
            prompt_obj = PromptTemplate(template=text_template, input_variables=["chat_history", "context", "question"])
            return prompt_obj.format(chat_history=chat_history, context=context, question=question)
         except Exception:
            pass
    try:
        return text_template.format(chat_history=chat_history, context=context, question=question)
    except Exception:
        return f"Previous Conversation: {chat_history}\n\nContext: {context}\n\nCurrent Question: {question}\n\nAnswer:"

def send_to_llm(llm_obj, prompt_text: str) -> str:
    try:
        if hasattr(llm_obj, "invoke"):
            out = llm_obj.invoke(prompt_text)
            if isinstance(out, dict):
                return out.get("result") or out.get("generated_text") or str(out)
            return str(out)
        if hasattr(llm_obj, "__call__"):
            out = llm_obj(prompt_text)
            if isinstance(out, dict):
                return out.get("result") or out.get("generated_text") or str(out)
            return str(out)
        if callable(llm_obj):
            out = llm_obj(prompt_text)
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    return first.get("generated_text") or first.get("translation_text") or first.get("summary_text") or str(first)
                return str(first)
            return str(out)
    except Exception:
        traceback.print_exc()
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
                    ok, info = ingest_dataframe(df, vs, batch_docs=500)
                    print("[LIFESPAN] ingest result:", ok, info)
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
            input_key="question",
            output_key="answer"
        )
        print("[LIFESPAN] Global ConversationBufferMemory created.")
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

@app.get("/")
def root():
    return {"status": "Text Generation (RAG) Service running with LOCAL models."}

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
        "memory_content": memory_content[:200] + "..." if len(memory_content) > 200 else memory_content,
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

    vs = ml_models.get("vector_store")
    local_llm = ml_models.get("local_llm") or ml_models.get("hf_pipeline")
    memory = ml_models.get("memory")

    if local_llm is None:
        raise HTTPException(status_code=503, detail="No local LLM available.")

    context_text = ""
    retrieved_docs = []

    if vs is not None:
        try:
            print(f"[LOGIC] Retrieving TOP_K={TOP_K} docs for query...")
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
                context_text = "(No relevant data context found in database)"
        except Exception as e:
            traceback.print_exc()
            context_text = "(Error during context retrieval)"
    else:
        context_text = "(Vector store not available)"

    # Load and debug chat history
    chat_history = ""
    if memory is not None:
        try:
            memory_data = memory.load_memory_variables({})
            chat_history = memory_data.get("chat_history", "")
            print(f"[MEMORY] Loaded chat history: '{chat_history}'")
        except Exception as e:
            print(f"[MEMORY] Error loading memory: {e}")

    formatted_prompt = build_fallback_prompt(chat_history, context_text, req.query)
    print(f"[PROMPT] Sending prompt to LLM (first 250 chars): {formatted_prompt[:250]}...")

    generated = send_to_llm(local_llm, formatted_prompt)
    if not generated:
        raise HTTPException(status_code=500, detail="LLM produced no output.")

    # Clean up the response
    generated = generated.strip()
    
    # Remove common prefixes
    prefixes_to_remove = ["Response:", "Answer:", "Explanation:", "Summary:"]
    for prefix in prefixes_to_remove:
        if generated.lower().startswith(prefix.lower()):
            generated = generated[len(prefix):].strip()
            break
    
    # Remove any leading/trailing quotes
    generated = generated.strip('"\'')
    
    print(f"[RESPONSE] Cleaned response: {generated}")

    # Save to memory - use the exact format LangChain expects
    if memory is not None:
        try:
            # Save the new conversation turn
            memory.save_context({"question": req.query}, {"answer": generated})
            print(f"[MEMORY] Saved to memory - Question: '{req.query}', Answer: '{generated}'")
            
            # Verify it was saved
            memory_data = memory.load_memory_variables({})
            print(f"[MEMORY] Verification - Current history: '{memory_data.get('chat_history', '')}'")
        except Exception as e:
            print(f"[MEMORY] Error saving memory: {e}")
            traceback.print_exc()

    return QueryResponse(answer=generated)

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")