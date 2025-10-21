# textgen_service/main.py
import os
import io
import zipfile
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import chromadb
import requests

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "cic-ids2018-sampled.csv")
DATASET_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2018/Dataset/MachineLearningCSV.zip"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# RAG primary repo (large). If fails to initialize, code will try a small model fallback.
PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

# ---------- Globals ----------
ml_models = {}  # holds embedding, vector_store, rag_chain, db_client

# ---------- Utilities ----------
def download_and_sample_dataset():
    """Download the zip and sample a CSV into KNOWLEDGE_FILE_PATH (if not already present)."""
    if os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"Knowledge file already present: {KNOWLEDGE_FILE_PATH}")
        return
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        print("Downloading dataset (this may take a while)...")
        r = requests.get(DATASET_URL, stream=True, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = next((n for n in z.namelist() if n.endswith(".csv") and z.getinfo(n).file_size > 0), None)
            if not csv_name:
                raise RuntimeError("No csv found in archive")
            print(f"Extracting {csv_name} from archive...")
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        sample_df.to_csv(KNOWLEDGE_FILE_PATH, index=False)
        print(f"Saved sampled CSV to {KNOWLEDGE_FILE_PATH}")
    except Exception as e:
        print("Failed to download/sample dataset:", e)
        traceback.print_exc()
        print("Continuing without a sampled knowledge file. You can add documents manually to ./documents")

# ---------- FastAPI lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LIFESPAN: starting up textgen service...")
    # 1) optionally create sampled dataset
    download_and_sample_dataset()

    # 2) initialize embedding function
    try:
        emb_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = emb_fn
        print("LIFESPAN: embedding function initialized.")
    except Exception:
        print("LIFESPAN: failed to load embeddings.")
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # 3) initialize Chroma persistent client, fallback to in-memory if needed
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

    # 4) ingest sampled CSV if vector store empty
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
            if os.path.exists(KNOWLEDGE_FILE_PATH) and vs is not None:
                df = pd.read_csv(KNOWLEDGE_FILE_PATH)
                required = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
                if all(c in df.columns for c in required):
                    docs = df.apply(
                        lambda r: (
                            f"A network traffic event. DstPort: {r['Dst Port']}. Protocol: {r['Protocol']}. "
                            f"FlowDuration: {r['Flow Duration']}. TotFwdPkts: {r['Tot Fwd Pkts']}. "
                            f"TotBwdPkts: {r['Tot Bwd Pkts']}. Label: {r['Label']}."
                        ),
                        axis=1
                    ).tolist()
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text("\n\n".join(docs))
                    vs.add_texts(texts=chunks)
                    print(f"LIFESPAN: ingested {len(docs)} documents (split -> {len(chunks)} chunks).")
                else:
                    print("LIFESPAN: sampled CSV missing required columns; skipping ingestion.")
            else:
                print("LIFESPAN: no sampled CSV found or vector store not ready; skipping ingestion.")
        else:
            print("LIFESPAN: vector store already has data; skipping ingestion.")
    except Exception:
        print("LIFESPAN: ingestion failed.")
        traceback.print_exc()

    # 5) initialize RAG chain: try primary model, fallback to small model if primary fails
    ml_models["rag_chain"] = None
    if not HF_TOKEN:
        print("LIFESPAN: HUGGINGFACEHUB_API_TOKEN not set; RAG chain unavailable.")
    else:
        if ml_models.get("vector_store") is None:
            print("LIFESPAN: vector store not available; cannot initialize RAG chain.")
        else:
            def try_create_llm(repo_id):
                try:
                    print(f"LIFESPAN: attempting to create HuggingFaceHub LLM for repo: {repo_id}")
                    llm = HuggingFaceHub(
                        repo_id=repo_id,
                        huggingfacehub_api_token=HF_TOKEN,
                        model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
                    )
                    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=ml_models["vector_store"].as_retriever())
                    print(f"LIFESPAN: LLM {repo_id} created successfully.")
                    return rag
                except Exception as e:
                    print(f"LIFESPAN: Failed to create LLM for {repo_id}: {e}")
                    traceback.print_exc()
                    return None

            # try primary
            rag = try_create_llm(PRIMARY_REPO_ID)
            if rag is None:
                print("LIFESPAN: primary LLM failed. Trying small model fallback (fast, public) ...")
                rag = try_create_llm(FALLBACK_SMALL_REPO_ID)
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
    # cleanup on shutdown
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
