"""
Robust RAG service with LOCAL MODEL support - CHATPROMPT FINAL
This version ensures the ChatPromptTemplate (when available) is converted to a plain-string
prompt for models like FLAN-T5, while still using ChatPromptTemplate inside RetrievalQA
when LangChain supports it. The code is defensive across LangChain/transformers versions.
"""
import os
import traceback
import pandas as pd
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Try imports (be tolerant across environments) ---
try:
    from langchain.prompts import ChatPromptTemplate
except Exception:
    try:
        from langchain.prompts.chat import ChatPromptTemplate
    except Exception:
        ChatPromptTemplate = None

try:
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        from langchain_core.prompts import PromptTemplate
    except Exception:
        PromptTemplate = None

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
LOCAL_MODEL_ID = "google/flan-t5-base"
TOP_K = 10

# ---------------- Globals ----------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "local_llm": None,   # LangChain HuggingFacePipeline wrapper (if used)
    "rag_chain": None,
    "hf_pipeline": None  # raw transformers pipeline fallback
}

# ---------------- Chat prompt setup ----------------
chat_prompt_obj = None
text_template = """You are a cybersecurity expert assistant. Answer the question based ONLY on the provided context from the database.

Context from database:
{context}

Question: {question}

Instructions:
1. Carefully read ALL the context documents above
2. If the question asks to "list" or "what are the types", extract ALL UNIQUE items mentioned across all context documents
3. If the question asks to "explain" or "what is", provide a detailed 3-5 sentence explanation
4. For listing questions: Create a clear list or enumeration of unique items found in the context
5. For explanation questions: Provide comprehensive details with examples from the context
6. NEVER repeat the same item multiple times
7. If information is not in the context, say "I cannot find this information in the provided context."
8. Do NOT use general knowledge - use ONLY what's in the context above

Answer:"""

if ChatPromptTemplate is not None:
    try:
        chat_prompt_obj = ChatPromptTemplate.from_messages([
            ("system", "You are a cybersecurity expert. Use the provided context to answer accurately and clearly. If info is missing, say you cannot find it in the context."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])
    except Exception:
        # If this signature isn't available, we'll fallback later when constructing prompt text
        chat_prompt_obj = None

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

    USEFUL_COLUMNS = ["Attack Type", "Attack Severity", "Threat Intelligence", "Response Action"]
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]
    if not cols_to_use:
        print("[INGEST] ERROR: Could not find expected columns.")
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

def create_rag_chain(llm, vs):
    if llm is None or vs is None:
        print("[RAG] Cannot create chain: llm or vs is None")
        return None
    if RetrievalQA is None:
        print("[RAG] RetrievalQA not available")
        return None
    try:
        prompt_obj = None
        if chat_prompt_obj is not None:
            try:
                prompt_obj = chat_prompt_obj
                print("[RAG] Using ChatPromptTemplate object for chain.")
            except Exception:
                traceback.print_exc()
                prompt_obj = None

        if prompt_obj is None and PromptTemplate is not None:
            try:
                prompt_obj = PromptTemplate(template=text_template, input_variables=["context", "question"])
                print("[RAG] Using fallback PromptTemplate for chain.")
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
            return_source_documents=True
        )
        print("[RAG] RAG chain created successfully with prompt")
        return rag
    except Exception:
        traceback.print_exc()
        return None

# ----- ChatPrompt -> plain-string formatter (robust) -----
def format_chat_prompt_to_text(prompt_obj: Any, context: str, question: str) -> Optional[str]:
    """
    Convert various ChatPromptTemplate / PromptValue representations into a plain string
    acceptable by text-only models (like flan-t5).
    This tries multiple common APIs: .format, .format_prompt, .format_messages, .to_string, .to_text, etc.
    """
    if prompt_obj is None:
        return None

    try:
        # 1) Direct .format -> often returns a string
        try:
            formatted = prompt_obj.format(context=context, question=question)
            if isinstance(formatted, str) and formatted.strip():
                return formatted
        except Exception:
            formatted = None

        # 2) format_prompt -> may return a PromptValue-like object
        try:
            if hasattr(prompt_obj, "format_prompt"):
                pv = prompt_obj.format_prompt(context=context, question=question)
                # Try common extraction methods
                if isinstance(pv, str) and pv.strip():
                    return pv
                if hasattr(pv, "to_string"):
                    return pv.to_string()
                if hasattr(pv, "to_text"):
                    return pv.to_text()
                # If it's a list/dict of messages
                if hasattr(pv, "messages"):
                    msgs = getattr(pv, "messages")
                    if isinstance(msgs, (list, tuple)):
                        parts = []
                        for m in msgs:
                            # m may be a dict-like or object
                            if isinstance(m, dict):
                                role = m.get("role") or m.get("type") or ""
                                content = m.get("content") or m.get("text") or ""
                                parts.append(f"{role.upper()}: {content}")
                            else:
                                # fallback to str()
                                parts.append(str(m))
                        return "\n\n".join(parts)
        except Exception:
            pass

        # 3) If prompt_obj has messages or to_messages
        try:
            if hasattr(prompt_obj, "to_messages"):
                msgs = prompt_obj.to_messages(context=context, question=question)
                # msgs may be a list of Message objects or dicts
                parts = []
                if isinstance(msgs, (list, tuple)):
                    for m in msgs:
                        if isinstance(m, dict):
                            role = m.get("role", "")
                            content = m.get("content", "") or m.get("text", "")
                            parts.append(f"{role.upper()}: {content}")
                        else:
                            # try common attributes
                            content = getattr(m, "content", None) or getattr(m, "text", None) or str(m)
                            role = getattr(m, "role", "") or ""
                            parts.append(f"{role.upper()}: {content}")
                    return "\n\n".join(parts)
        except Exception:
            pass

        # 4) Fallback: try str() on the object after calling format (if possible)
        try:
            if hasattr(prompt_obj, "format"):
                maybe = prompt_obj.format(context=context, question=question)
                return str(maybe)
        except Exception:
            pass
    except Exception:
        traceback.print_exc()

    return None

# Robust sender to LLM: handle various wrapper interfaces
def send_to_llm(llm_obj, prompt_text: str) -> str:
    try:
        if hasattr(llm_obj, "invoke"):
            out = llm_obj.invoke(prompt_text)
            if isinstance(out, dict):
                # common fields
                for k in ("result", "generated_text", "text", "output_text", "answer"):
                    if k in out and out[k]:
                        return out[k]
                return str(out)
            return str(out)
        if hasattr(llm_obj, "__call__") and not isinstance(llm_obj, type):
            out = llm_obj(prompt_text)
            if isinstance(out, dict):
                for k in ("result", "generated_text", "text", "output_text", "answer"):
                    if k in out and out[k]:
                        return out[k]
                return str(out)
            # If it's a raw transformers pipeline result
            if isinstance(out, list) and out:
                first = out[0]
                if isinstance(first, dict):
                    for k in ("generated_text", "translation_text", "summary_text", "text"):
                        if k in first and first[k]:
                            return first[k]
                    return str(first)
                return str(first)
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
            try:
                vs._collection.delete(ids=vs._collection.get()['ids'])
                print("[INGEST] Old data cleared.")
            except Exception:
                print("[INGEST] Could not clear via _collection.delete; continuing.")
        else:
            print("[INGEST] Could not automatically clear old data. Re-ingesting anyway.")
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

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    """
    Generate answers using RAG - ensures retrieved context + chat prompt are passed to the LLM
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Prefer RAG chain if present (it encapsulates retrieval + prompt)
    rag = ml_models.get("rag_chain")
    if rag is not None:
        try:
            if hasattr(rag, "invoke"):
                res = rag.invoke({"query": req.query})
            else:
                res = rag({"query": req.query})
            if isinstance(res, dict):
                # support many possible return keys
                out_text = res.get("result") or res.get("output_text") or res.get("answer") or res.get("text") or res.get("generated_text")
                # If still None, try stringifying
                if out_text is None:
                    out_text = str(res)
            else:
                out_text = str(res)
            out_text = str(out_text).strip()
            return QueryResponse(answer=out_text)
        except Exception:
            traceback.print_exc()
            # fall through to manual retrieval + format + LLM flow

    # Manual flow: retrieve context and format prompt
    vs = ml_models.get("vector_store")
    local_llm = ml_models.get("local_llm") or ml_models.get("hf_pipeline")
    context_text = ""
    if vs is not None:
        try:
            retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
            # common retriever methods across versions:
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(req.query)
            elif hasattr(retriever, "retrieve"):
                docs = retriever.retrieve(req.query)
            else:
                docs = retriever(req.query) if callable(retriever) else []
            if docs:
                context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
        except Exception:
            traceback.print_exc()

    # Build formatted prompt text: prefer converting ChatPromptTemplate to plain text
    formatted_prompt = None
    if chat_prompt_obj is not None:
        try:
            formatted_prompt = format_chat_prompt_to_text(chat_prompt_obj, context_text, req.query)
        except Exception:
            traceback.print_exc()
            formatted_prompt = None

    # Fallback to plain text template if conversion failed
    if formatted_prompt is None:
        formatted_prompt = build_fallback_prompt(context_text, req.query)

    # Send to LLM
    if local_llm is None:
        raise HTTPException(status_code=503, detail="No local LLM available to generate answer.")

    # (optional) debug log of first part of prompt — helpful when debugging locally
    try:
        print("[PROMPT PREVIEW]", formatted_prompt[:1200].replace("\n", " \\n "))
    except Exception:
        pass

    generated = send_to_llm(local_llm, formatted_prompt)
    if not generated:
        raise HTTPException(status_code=500, detail="LLM produced no output.")

    generated = generated.strip()
    return QueryResponse(answer=generated)

if __name__ == "__main__":
    print("Run: uvicorn main:app --host 0.0.0.0 --port 8002")
