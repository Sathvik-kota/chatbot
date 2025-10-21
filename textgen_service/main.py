import os
import pandas as pd
import traceback
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
import zipfile
import io

# --- Configuration ---
ml_models = {}
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DOCUMENTS_DIR = "./documents"
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "cic-ids2018-sampled.csv")
DATASET_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2018/Dataset/MachineLearningCSV.zip"  # Official source

def setup_knowledge_base():
    """
    Downloads & samples the dataset if not present at KNOWLEDGE_FILE_PATH.
    """
    if os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"✅ Knowledge base file found at {KNOWLEDGE_FILE_PATH}.")
        return

    print(f"⚠️ Knowledge base not found. Starting automatic download and sampling...")
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file_name = next((f for f in z.namelist() if f.endswith('.csv') and z.getinfo(f).file_size > 0), None)
            if not csv_file_name:
                raise RuntimeError("No CSV files found in the downloaded archive.")
            print(f"Extracting and reading '{csv_file_name}' from the archive...")
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f)

        df.columns = df.columns.str.strip()
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        sample_df.to_csv(KNOWLEDGE_FILE_PATH, index=False)
        print(f"✅ Sampled knowledge base saved to {KNOWLEDGE_FILE_PATH}")

    except Exception as e:
        print(f"❌ FAILED to automatically set up knowledge base: {e}")
        traceback.print_exc()
        print("Service will start with an empty DB. Add documents manually if needed.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up RAG models and vector store...")
    # 1) ensure dataset exists (sample)
    setup_knowledge_base()

    # 2) embeddings
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = embedding_function
        print("✅ Embedding function initialized.")
    except Exception as e:
        ml_models["embedding_function"] = None
        print("❌ Failed to initialize embeddings:", e)
        traceback.print_exc()

    # 3) Chroma persistent client + vector store
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ml_models["db_client"] = persistent_client
        ml_models["vector_store"] = Chroma(
            client=persistent_client,
            collection_name="rag_collection",
            embedding_function=ml_models["embedding_function"],
        )
        print("✅ Chroma vector store initialized.")
    except Exception as e:
        ml_models["vector_store"] = None
        print("❌ Failed to initialize Chroma vector store:", e)
        traceback.print_exc()

    # 4) Ingest if empty
    try:
        should_ingest = True
        try:
            if ml_models["vector_store"] is not None and ml_models["vector_store"]._collection.count() > 0:
                should_ingest = False
        except Exception:
            should_ingest = True

        if should_ingest:
            print("Database appears empty. Attempting ingestion from sampled file...")
            if os.path.exists(KNOWLEDGE_FILE_PATH) and ml_models.get("vector_store") is not None:
                df = pd.read_csv(KNOWLEDGE_FILE_PATH)
                required_columns = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
                if all(col in df.columns for col in required_columns):
                    documents = df.apply(
                        lambda row: (
                            f"A network traffic event was recorded. Destination port: {row['Dst Port']}. "
                            f"Protocol: {row['Protocol']}. Flow duration: {row['Flow Duration']} microseconds. "
                            f"Tot Fwd Pkts: {row['Tot Fwd Pkts']}. Tot Bwd Pkts: {row['Tot Bwd Pkts']}. "
                            f"Label: {row['Label']}."
                        ),
                        axis=1
                    ).tolist()

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    docs_split = text_splitter.split_text("\n\n".join(documents))
                    ml_models["vector_store"].add_texts(texts=docs_split)
                    print(f"✅ Ingested {len(documents)} records into the vector store.")
                else:
                    print("⚠️ Sampled dataset missing required columns. Skipping ingestion.")
            else:
                print("⚠️ No sampled knowledge file found or vector store not initialized. Skipping ingestion.")
        else:
            print("Vector store already contains data; skipping ingestion.")
    except Exception as e:
        print("❌ Error during ingestion:", e)
        traceback.print_exc()

    # 5) Initialize RAG chain
    if not HF_TOKEN:
        print("⚠️ HUGGINGFACEHUB_API_TOKEN is not set. RAG chain will be unavailable.")
        ml_models["rag_chain"] = None
    else:
        if ml_models.get("vector_store") is None:
            print("⚠️ Vector store unavailable; cannot create RAG chain.")
            ml_models["rag_chain"] = None
        else:
            try:
                print("🔐 Creating HuggingFaceHub LLM (this may contact HF and take time)...")
                llm = HuggingFaceHub(
                    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
                )
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ml_models["vector_store"].as_retriever()
                )
                ml_models["rag_chain"] = rag_chain
                print("✅ RAG chain initialized successfully.")
            except Exception as e:
                ml_models["rag_chain"] = None
                print("❌ Failed to initialize RAG chain:", e)
                traceback.print_exc()

    yield
    # cleanup
    ml_models.clear()
    print("✅ Shutdown complete. Models cleared.")

app = FastAPI(lifespan=lifespan)

# --- Request/response models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# --- Routes
@app.get("/")
def read_root():
    return {"status": "Text Generation (RAG) Service is running."}

@app.get("/status")
def status():
    return {
        "vector_store_ready": bool(ml_models.get("vector_store")),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN)
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    if not ml_models.get("rag_chain"):
        raise HTTPException(status_code=503, detail="RAG chain is not available. Check server configuration and API token.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        # .run returns a string answer
        result_str = ml_models["rag_chain"].run(request.query)
        if result_str is None:
            return QueryResponse(answer="Could not generate an answer.")
        return QueryResponse(answer=str(result_str))
    except Exception as e:
        print("❌ Error during generation:", e)
        traceback.print_exc()
        # Friendly fallback for clients
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {e}")
