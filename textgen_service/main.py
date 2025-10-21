# textgen_service/main.py
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import chromadb

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
CSV_FILE_PATH = os.path.join(ROOT_DIR, "sampled_5k.csv")  # your uploaded CSV
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

# ---------- Globals ----------
ml_models = {}  # holds embedding, vector_store, rag_chain, db_client

# ---------- FastAPI lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LIFESPAN: starting up textgen service...")

    # 1) initialize embedding function
    try:
        emb_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = emb_fn
        print("LIFESPAN: embedding function initialized.")
    except Exception:
        print("LIFESPAN: failed to load embeddings.")
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # 2) initialize Chroma persistent client, fallback to in-memory if needed
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ml_models["db_client"] = client
        ml_models["vector_store"] = Chroma(
            client=client,
            collection_name="rag_collection",
            embedding_function=ml_models["embedding_function"],
        )
        print("LIFESPAN: Persistent Chroma initialized.")
    except Exception:
        print("LIFESPAN: Persistent Chroma failed, attempting in-memory fallback.")
        traceback.print_exc()
        try:
            client = chromadb.Client()  # in-memory
            ml_models["db_client"] = client
            ml_models["vector_store"] = Chroma(
                client=client,
                collection_name="rag_collection",
                embedding_function=ml_models["embedding_function"],
            )
            print("LIFESPAN: In-memory Chroma initialized.")
        except Exception:
            print("LIFESPAN: In-memory Chroma also failed.")
            traceback.print_exc()
            ml_models["vector_store"] = None

    # 3) ingest CSV into vector store
    try:
        vs = ml_models.get("vector_store")
        need_ingest = True
        try:
            if vs is not None and vs._collection.count() > 0:
                need_ingest = False
        except Exception:
            need_ingest = True

        if need_ingest:
            print("LIFESPAN: vector store empty or unknown, attempting ingestion...")
            if os.path.exists(CSV_FILE_PATH) and vs is not None:
                df = pd.read_csv(CSV_FILE_PATH)
                # Compose simple text documents from CSV columns
                docs = df.apply(
                    lambda r: f"Event at {r.get('Timestamp','N/A')}. Source: {r.get('Source IP','N/A')} -> Dest: {r.get('Destination IP','N/A')}. Protocol: {r.get('Protocol','N/A')}. Severity: {r.get('Severity','N/A')}. Type: {r.get('Event Type','N/A')}. Description: {r.get('Description','N/A')}.",
                    axis=1
                ).tolist()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_text("\n\n".join(docs))
                vs.add_texts(texts=chunks)
                print(f"LIFESPAN: ingested {len(docs)} documents (split -> {len(chunks)} chunks).")
            else:
                print("LIFESPAN: CSV not found or vector store not ready; skipping ingestion.")
        else:
            print("LIFESPAN: vector store already has data; skipping ingestion.")
    except Exception:
        print("LIFESPAN: ingestion failed.")
        traceback.print_exc()

    # 4) initialize RAG chain
    ml_models["rag_chain"] = None
    if not HF_TOKEN:
        print("LIFESPAN: HUGGINGFACEHUB_API_TOKEN not set; RAG chain unavailable.")
    else:
        if ml_models.get("vector_store") is None:
            print("LIFESPAN: vector store not available; cannot initialize RAG chain.")
        else:
            def try_create_llm(repo_id, task):
                try:
                    print(f"LIFESPAN: attempting to create HuggingFaceHub LLM for repo: {repo_id}")
                    llm = HuggingFaceHub(
                        repo_id=repo_id,
                        task=task,
                        huggingfacehub_api_token=HF_TOKEN,
                        model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
                    )
                    rag = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=ml_models["vector_store"].as_retriever()
                    )
                    print(f"LIFESPAN: LLM {repo_id} created successfully.")
                    return rag
                except Exception as e:
                    print(f"LIFESPAN: Failed to create LLM for {repo_id}: {e}")
                    traceback.print_exc()
                    return None

            rag = try_create_llm(PRIMARY_REPO_ID, task="text-generation")
            if rag is None:
                print("LIFESPAN: primary LLM failed. Trying small model fallback...")
                rag = try_create_llm(FALLBACK_SMALL_REPO_ID, task="text2text-generation")
                if rag is None:
                    print("LIFESPAN: small model fallback also failed; RAG chain unavailable.")
                else:
                    ml_models["rag_chain"] = rag
            else:
                ml_models["rag_chain"] = rag

    print("LIFESPAN: startup complete. Status:", {
        "vector_store_ready": bool(ml_models.get("vector_store")),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN),
    })
    yield
    ml_models.clear()
    print("LIFESPAN: shutdown complete.")

# ---------- FastAPI app ----------
app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "Text Generation (RAG) Service is running."}

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
        raise HTTPException(status_code=503, detail="RAG chain is not available. Check server configuration and API token.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        out = ml_models["rag_chain"].run(req.query)
        return QueryResponse(answer=str(out) if out is not None else "No answer generated.")
    except Exception as e:
        print("Generation error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
