import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# UPDATED IMPORTS - No more deprecation warnings!
try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
    USE_NEW_IMPORTS = True
except ImportError:
    # Fallback for older installations
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceHub as HuggingFaceEndpoint
    USE_NEW_IMPORTS = False

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "cic-ids2018-sampled-5k.csv")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# RAG models
PRIMARY_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FALLBACK_SMALL_REPO_ID = "google/flan-t5-small"

# ---------- Globals ----------
ml_models = {}

# ---------- Utilities ----------
def load_csv_and_sample():
    """Load the 5k CSV (already uploaded) and verify columns"""
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

# ---------- FastAPI lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LIFESPAN: starting up textgen service...")

    # 1) initialize embedding function
    try:
        emb_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = emb_fn
        print(f"LIFESPAN: embedding function initialized (using new imports: {USE_NEW_IMPORTS}).")
    except Exception as e:
        print("LIFESPAN: failed to initialize embeddings.")
        print(f"Error: {e}")
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # 2) initialize Chroma persistent client
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        client = Chroma(
            persist_directory=CHROMA_DB_PATH, 
            embedding_function=ml_models["embedding_function"]
        )
        ml_models["db_client"] = client
        ml_models["vector_store"] = client
        print("LIFESPAN: Persistent Chroma initialized.")
    except Exception as e:
        print("LIFESPAN: Chroma initialization failed, fallback to in-memory.")
        print(f"Error: {e}")
        traceback.print_exc()
        try:
            client = Chroma(embedding_function=ml_models["embedding_function"])
            ml_models["db_client"] = client
            ml_models["vector_store"] = client
            print("LIFESPAN: In-memory Chroma initialized.")
        except Exception as e2:
            print("LIFESPAN: Chroma fallback also failed.")
            print(f"Error: {e2}")
            traceback.print_exc()
            ml_models["vector_store"] = None

    # 3) ingest CSV into vector store if empty
    vs = ml_models.get("vector_store")
    if vs is not None:
        try:
            collection_count = vs._collection.count()
            
            if collection_count == 0:
                print("LIFESPAN: Vector store is empty, attempting ingestion...")
                df = load_csv_and_sample()
                if df is not None:
                    required_cols = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'Label']
                    if all(c in df.columns for c in required_cols):
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
                        print(f"LIFESPAN: ingested {len(docs)} docs -> {len(chunks)} chunks.")
                    else:
                        print(f"LIFESPAN: CSV missing required columns (need {required_cols}); skipping ingestion.")
                else:
                    print("LIFESPAN: CSV not loaded; skipping ingestion.")
            else:
                print(f"LIFESPAN: vector store already has {collection_count} items; skipping ingestion.")
        except Exception as e:
            print("LIFESPAN: vector store ingestion/check failed.")
            print(f"Error: {e}")
            traceback.print_exc()

    # 4) initialize RAG chain
    ml_models["rag_chain"] = None
    if not HF_TOKEN:
        print("LIFESPAN: HUGGINGFACEHUB_API_TOKEN not set; RAG chain unavailable.")
    else:
        if vs is None:
            print("LIFESPAN: vector store unavailable; cannot init RAG chain.")
        else:
            def try_create_llm(repo_id):
                try:
                    print(f"LIFESPAN: creating LLM for {repo_id}")
                    
                    # Use new HuggingFaceEndpoint
                    llm = HuggingFaceEndpoint(
                        repo_id=repo_id,
                        huggingfacehub_api_token=HF_TOKEN,
                        temperature=0.5,
                        max_new_tokens=1024
                    )
                    
                    rag = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vs.as_retriever(search_kwargs={"k": 3})
                    )
                    print(f"LIFESPAN: LLM {repo_id} initialized successfully.")
                    return rag
                except Exception as e:
                    print(f"LIFESPAN: Failed LLM {repo_id}: {e}")
                    traceback.print_exc()
                    return None

            rag = try_create_llm(PRIMARY_REPO_ID)
            if rag is None:
                print("LIFESPAN: Primary model failed, trying fallback...")
                rag = try_create_llm(FALLBACK_SMALL_REPO_ID)
            
            ml_models["rag_chain"] = rag

    print("LIFESPAN: startup complete. Status:", {
        "vector_store_ready": bool(vs),
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
    rag_chain = ml_models.get("rag_chain")
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG chain is not available.")
    
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        print(f"Invoking RAG chain with query: {req.query}")
        result = rag_chain.invoke({"query": req.query})
        
        answer = result.get("result", "No answer generated.")
        print(f"RAG chain generated answer: {answer}")
        
        return QueryResponse(answer=str(answer))

    except Exception as e:
        print("Generation error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
