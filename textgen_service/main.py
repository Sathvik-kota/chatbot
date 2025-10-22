# textgen_service/main.py
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "ai_cybersecurity_dataset-sampled-5k.csv")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

ml_models = {}

def load_csv():
    """Load the 5k sampled cybersecurity CSV"""
    if not os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"Knowledge CSV not found: {KNOWLEDGE_FILE_PATH}")
        return None
    try:
        df = pd.read_csv(KNOWLEDGE_FILE_PATH)
        df.columns = df.columns.str.strip()
        print(f"Loaded CSV with {len(df)} rows, columns: {list(df.columns)}")
        return df
    except Exception as e:
        print("Failed to load CSV:", e)
        traceback.print_exc()
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up textgen service...")

    # 1) Initialize embedding
    try:
        emb_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = emb_fn
    except Exception:
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # 2) Initialize Chroma vector store
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=ml_models["embedding_function"])
        ml_models["vector_store"] = client
    except Exception:
        traceback.print_exc()
        try:
            client = Chroma(embedding_function=ml_models["embedding_function"])
            ml_models["vector_store"] = client
        except Exception:
            traceback.print_exc()
            ml_models["vector_store"] = None

    # 3) Ingest CSV if empty
    vs = ml_models.get("vector_store")
    if vs:
        try:
            docs_in_store = vs._collection.get(include=['documents'])['documents']
            if len(docs_in_store) == 0:
                df = load_csv()
                if df is not None:
                    # Convert each row to a simple string representation
                    docs = df.apply(lambda r: " | ".join([f"{c}: {r[c]}" for c in df.columns]), axis=1).tolist()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text("\n\n".join(docs))
                    vs.add_texts(texts=chunks)
                    vs.persist()
                    print(f"Ingested {len(docs)} docs -> {len(chunks)} chunks.")
        except Exception:
            traceback.print_exc()

    # 4) Initialize RAG chain
    ml_models["rag_chain"] = None
    if HF_TOKEN and vs:
        def try_create_llm(repo_id, task_type):
            try:
                llm = HuggingFaceHub(
                    repo_id=repo_id,
                    task=task_type,
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
                )
                rag = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vs.as_retriever()
                )
                return rag
            except Exception:
                traceback.print_exc()
                return None

        rag = try_create_llm(PRIMARY_REPO_ID, task_type="text-generation")
        if rag is None:
            rag = try_create_llm(FALLBACK_SMALL_REPO_ID, task_type="text2text-generation")
        ml_models["rag_chain"] = rag

    print("Startup complete.")
    yield
    ml_models.clear()
    print("Shutdown complete.")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Text Generation (RAG) Service running."}

@app.get("/status")
def status():
    return {
        "vector_store_ready": bool(ml_models.get("vector_store")),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN),
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not ml_models.get("rag_chain"):
        raise HTTPException(status_code=503, detail="RAG chain not available.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        out = ml_models["rag_chain"].run(req.query)
        return QueryResponse(answer=str(out) if out is not None else "No answer generated.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
