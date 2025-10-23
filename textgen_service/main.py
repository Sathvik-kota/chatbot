"""
Robust RAG service with LOCAL MODEL support - CHAT PROMPT INTEGRATION
Uses ChatPromptTemplate when available and falls back gracefully.
"""
import os
import traceback
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
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
HARDCODED_CSV_PATH = os.path.join(ROOT_DIR, "ai_cybersecurity_dataset-sampled-5k.csv")
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
    "hf_pipeline": None
}

# ---------------- Chat prompt setup ----------------
chat_prompt_obj = None
text_template = """You are a cybersecurity expert assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer using ONLY context.
- List all unique items if the question asks for types or lists.
- Provide 3-5 sentence explanation if question asks 'what is' or 'explain'.
- If info is missing, say 'I cannot find this information in the provided context.'

Answer:"""

if ChatPromptTemplate is not None:
    try:
        chat_prompt_obj = ChatPromptTemplate.from_messages([
            ("system", "You are a cybersecurity expert. Use the provided context to answer accurately. If info is missing, say you cannot find it."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}")
        ])
    except Exception:
        chat_prompt_obj = None

# ---------------- Helpers ----------------
def find_csv_path(basename: str = DEFAULT_CSV_BASENAME) -> Optional[str]:
    candidates = [HARDCODED_CSV_PATH,
                  os.path.join(DOCUMENTS_DIR, basename),
                  os.path.join(ROOT_DIR, basename),
                  os.path.join(os.getcwd(), basename),
                  basename]
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
        return None
    try:
        vs = LC_Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
        print("[CHROMA] Initialized Chroma vector store.")
        return vs
    except Exception:
        traceback.print_exc()
        return None

def vs_count_estimate(vs) -> int:
    if vs is None:
        return 0
    try:
        col = getattr(vs, "_collection", None)
        if col and hasattr(col, "count"):
            return int(col.count())
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
    return False, None

def ingest_dataframe(df: pd.DataFrame, vs, batch_docs: int = 500):
    if df is None or vs is None:
        return False, "no_df_or_vs"
    USEFUL_COLUMNS = ["Attack Type", "Attack Severity", "Threat Intelligence", "Response Action"]
    df_cols_lower = {col.lower(): col for col in df.columns}
    cols_to_use = [df_cols_lower[c.lower()] for c in USEFUL_COLUMNS if c.lower() in df_cols_lower]
    if not cols_to_use:
        return False, "missing_expected_columns"
    docs = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in cols_to_use if pd.notna(row[col]) and str(row[col]).strip()]
        doc = ". ".join(parts)
        if doc.strip():
            docs.append(doc)
    total_added = 0
    for i in range(0, len(docs), batch_docs):
        batch = docs[i:i+batch_docs]
        ok, _ = _try_add_texts(vs, batch)
        if not ok:
            return False, f"batch_failed_at_{i}"
        total_added += len(batch)
    try:
        if hasattr(vs, "persist"):
            vs.persist()
    except Exception:
        traceback.print_exc()
    return True, {"docs_added": total_added}

def create_local_llm():
    if HuggingFacePipeline is None or pipeline is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_ID)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=512, temperature=0.3, top_p=0.95, top_k=50, repetition_penalty=1.3)
        ml_models["hf_pipeline"] = pipe
        return HuggingFacePipeline(pipeline=pipe)
    except Exception:
        traceback.print_exc()
        return None

def create_rag_chain(llm, vs):
    if llm is None or vs is None or RetrievalQA is None:
        return None
    prompt_obj = chat_prompt_obj if chat_prompt_obj is not None else PromptTemplate(template=text_template, input_variables=["context", "question"])
    try:
        rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vs.as_retriever(search_kwargs={"k": TOP_K}),
            chain_type_kwargs={"prompt": prompt_obj},
            return_source_documents=True
        )
        return rag
    except Exception:
        traceback.print_exc()
        return None

def build_fallback_prompt(context: str, question: str) -> str:
    ctx = context.strip() if context else "(no context available)"
    return f"You are a cybersecurity expert.\nContext:\n{ctx}\nQuestion:\n{question}\nAnswer using only context."

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
                    return first.get("generated_text") or str(first)
                return str(first)
            return str(out)
    except Exception:
        traceback.print_exc()
    return ""

# ---------------- Lifespan & app ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["embedding_function"] = create_embedding_function()
    ml_models["vector_store"] = init_chroma(ml_models["embedding_function"])
    csv_path = find_csv_path()
    vs = ml_models["vector_store"]
    if vs and csv_path:
        df = load_dataframe_from_csv(csv_path)
        if df is not None and len(df) > 0:
            ingest_dataframe(df, vs)
    ml_models["local_llm"] = create_local_llm()
    if ml_models["local_llm"] and vs:
        ml_models["rag_chain"] = create_rag_chain(ml_models["local_llm"], vs)
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# ---------------- Request/Response Models ----------------
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    return {"status": "Text Generation (RAG) Service running with LOCAL models."}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    count = vs_count_estimate(vs) if vs else 0
    return {
        "vector_store_ready": vs is not None,
        "vector_store_has_data": count > 0,
        "vector_store_count": count,
        "local_llm_ready": ml_models.get("local_llm") is not None,
        "rag_chain_ready": ml_models.get("rag_chain") is not None,
        "model_id": LOCAL_MODEL_ID,
        "top_k": TOP_K,
        "csv_found": bool(find_csv_path())
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    rag = ml_models.get("rag_chain")
    if rag:
        try:
            if hasattr(rag, "invoke"):
                res = rag.invoke({"query": req.query})
            else:
                res = rag({"query": req.query})
            if isinstance(res, dict):
                out_text = res.get("result") or res.get("output_text") or str(res)
            else:
                out_text = str(res)
            return QueryResponse(answer=out_text.strip())
        except Exception:
            traceback.print_exc()

    vs = ml_models.get("vector_store")
    local_llm = ml_models.get("local_llm") or ml_models.get("hf_pipeline")
    context_text = ""
    if vs:
        try:
            retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
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

    if chat_prompt_obj:
        try:
            try:
                formatted_prompt = chat_prompt_obj.format(context=context_text, question=req.query)
            except Exception:
                fp = chat_prompt_obj.format_prompt(context=context_text, question=req.query)
                formatted_prompt = str(fp)
        except Exception:
            traceback.print_exc()
            formatted_prompt = None
    else:
        formatted_prompt = None

    if formatted_prompt is None:
        formatted_prompt = build_fallback_prompt(context_text, req.query)

    if local_llm is None:
        raise HTTPException(status_code=503, detail="No local LLM available.")

    generated = send_to_llm(local_llm, formatted_prompt)
    if not generated:
        raise HTTPException(status_code=500, detail="LLM produced no output.")

    return QueryResponse(answer=generated.strip())
