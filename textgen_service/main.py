import os
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
import zipfile
import io
import traceback

# --- API Setup ---
ml_models = {}
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
DOCUMENTS_DIR = "./documents"
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "cic-ids2018-sampled.csv")
DATASET_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2018/Dataset/MachineLearningCSV.zip"  # Official source

def setup_knowledge_base():
    """
    Checks for the dataset, downloads and samples it if not present.
    """
    if os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"✅ Knowledge base file found at {KNOWLEDGE_FILE_PATH}.")
        return

    print(f"⚠️ Knowledge base not found. Starting automatic download and sampling...")
    print(f"Downloading from: {DATASET_URL}")
    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        response = requests.get(DATASET_URL, stream=True, timeout=60)
        response.raise_for_status()

        # Unzip and find a specific CSV file to use (e.g., first non-empty CSV)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file_name = next((f for f in z.namelist() if f.endswith('.csv') and z.getinfo(f).file_size > 0), None)
            if not csv_file_name:
                raise RuntimeError("No CSV files found in the downloaded archive.")
            print(f"Extracting and reading '{csv_file_name}' from the archive...")
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f)

        print(f"Successfully loaded dataframe with {len(df)} rows.")

        # Clean column names (often have leading spaces)
        df.columns = df.columns.str.strip()

        # Take a sample of 2000 records
        print("Sampling 2000 records from the dataset...")
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)

        # Save the sampled data
        sample_df.to_csv(KNOWLEDGE_FILE_PATH, index=False)
        print(f"✅ Sampled knowledge base saved to {KNOWLEDGE_FILE_PATH}")

    except Exception as e:
        print(f"❌❌❌ FAILED to automatically set up knowledge base: {e}")
        traceback.print_exc()
        print("The service will start with an empty database. Please add documents manually.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up and loading RAG models...")
    # --- AUTOMATED KNOWLEDGE BASE SETUP ---
    setup_knowledge_base()

    # Initialize embeddings
    try:
        embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = embedding_function
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # Initialize Chroma persistent client and vector store
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        persistent_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ml_models["db_client"] = persistent_client

        ml_models["vector_store"] = Chroma(
            client=persistent_client,
            collection_name="rag_collection",
            embedding_function=ml_models["embedding_function"],
        )
    except Exception as e:
        print(f"❌ Failed to initialize Chroma vector store: {e}")
        traceback.print_exc()
        ml_models["vector_store"] = None

    # Ingest data if DB is empty
    try:
        collection_count = 0
        try:
            collection_count = ml_models["vector_store"]._collection.count()
        except Exception:
            collection_count = 0

        if collection_count == 0:
            print("Database is empty. Attempting to ingest from file...")
            if os.path.exists(KNOWLEDGE_FILE_PATH):
                try:
                    df = pd.read_csv(KNOWLEDGE_FILE_PATH)
                    required_columns = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
                    if all(col in df.columns for col in required_columns):
                        documents = df.apply(
                            lambda row: f"A network traffic event was recorded. "
                                        f"Destination port: {row['Dst Port']}. "
                                        f"Protocol type: {row['Protocol']}. "
                                        f"Flow duration: {row['Flow Duration']} microseconds. "
                                        f"Total forward packets: {row['Tot Fwd Pkts']}. "
                                        f"Total backward packets: {row['Tot Bwd Pkts']}. "
                                        f"This activity was labeled as: {row['Label']}.",
                            axis=1
                        ).tolist()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        docs_split = text_splitter.split_text("\n\n".join(documents))
                        ml_models["vector_store"].add_texts(texts=docs_split)
                        print(f"✅ Successfully ingested {len(documents)} records from the dataset.")
                    else:
                        print(f"⚠️ WARNING: Dataset is missing required columns. Skipping ingestion.")
                except Exception as e:
                    print(f"❌ Error during file ingestion: {e}")
                    traceback.print_exc()
            else:
                print("⚠️ No knowledge file found to ingest from.")
        else:
            print("Database already contains data. Skipping file ingestion.")
    except Exception as e:
        print(f"❌ Error checking/ingesting collection: {e}")
        traceback.print_exc()

    # Initialize RAG chain only if HF token present and vector store available
    if not HF_TOKEN:
        print("⚠️ WARNING: HUGGINGFACEHUB_API_TOKEN not set. RAG service will be unavailable.")
        ml_models["rag_chain"] = None
    else:
        if ml_models.get("vector_store") is None:
            print("⚠️ WARNING: Vector store not available. RAG chain cannot be created.")
            ml_models["rag_chain"] = None
        else:
            try:
                repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                llm = HuggingFaceHub(
                    repo_id=repo_id,
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
                )
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ml_models["vector_store"].as_retriever()
                )
                ml_models["rag_chain"] = rag_chain
                print("✅ RAG chain models loaded successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize RAG chain: {e}")
                traceback.print_exc()
                ml_models["rag_chain"] = None

    yield
    ml_models.clear()
    print("✅ Models cleared. Shutting down.")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    return {"status": "Text Generation (RAG) Service is running."}

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(request: QueryRequest):
    if not ml_models.get("rag_chain"):
        raise HTTPException(status_code=503, detail="RAG chain is not available. Check server configuration and API token.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Use .run to get string output from RetrievalQA
        result_str = ml_models["rag_chain"].run(request.query)
        if result_str is None:
            return QueryResponse(answer="Could not generate an answer.")
        return QueryResponse(answer=str(result_str))
    except Exception as e:
        print("❌ Error during generation:")
        traceback.print_exc()
        # friendly fallback message if the remote HF client fails
        return QueryResponse(answer=f"RAG generation failed: {e}")
