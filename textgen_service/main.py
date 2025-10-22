# textgen_service/main.py
# TESTED WORKING VERSION (update: text splitter import moved to langchain_text_splitters)
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Core imports
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# <-- updated import: langchain_text_splitters is now a separate package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ChromaDB - EXPLICIT CLIENT IMPORT (FIX FOR YOUR ERROR)
import chromadb
from chromadb.config import Settings

# LangChain ChromaDB wrapper (community package)
from langchain_community.vectorstores import Chroma

# HuggingFace LLM access
try:
    # If you installed the newer "langchain-huggingface" / HuggingFaceEndpoint wrapper
    # (this import will succeed when that package is available)
    from langchain_huggingface import HuggingFaceEndpoint
    USE_NEW_API = True
except Exception:
    # Fallback to the community wrapper that wraps HF Hub endpoints
    from langchain_community.llms import HuggingFaceHub
    USE_NEW_API = False

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
    """Load CSV data"""
    if not os.path.exists(KNOWLEDGE_FILE_PATH):
        print(f"CSV not found: {KNOWLEDGE_FILE_PATH}")
        return None
    try:
        df = pd.read_csv(KNOWLEDGE_FILE_PATH)
        df.columns = df.columns.str.strip()
        print(f"Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"CSV error: {e}")
        traceback.print_exc()
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*70)
    print("STARTING SERVICE (ChromaDB Version)")
    print("="*70)
    
    # 1) Initialize embeddings
    print("\n[1/4] Initializing Embeddings...")
    try:
        emb_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = emb_fn
        print("SUCCESS: Embeddings ready")
    except Exception as e:
        print(f"FAILED: Embeddings - {e}")
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # 2) Initialize ChromaDB - THE FIXED WAY
    print("\n[2/4] Initializing ChromaDB Vector Store...")
    
    emb_fn = ml_models.get("embedding_function")
    vs = None
    
    if not emb_fn:
        print("FAILED: Cannot init vector store - No embeddings")
        ml_models["vector_store"] = None
    else:
        try:
            # Ensure directories exist
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            print(f"Directories created: {CHROMA_DB_PATH}")
            
            # CRITICAL FIX: Use PersistentClient explicitly
            print("Creating ChromaDB PersistentClient...")
            chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print("SUCCESS: ChromaDB client created")
            
            # Create vector store with explicit client
            vs = Chroma(
                client=chroma_client,
                collection_name="cybersecurity_docs",
                embedding_function=emb_fn
            )
            
            # Verify it works
            count = vs._collection.count()
            print(f"SUCCESS: ChromaDB initialized! Documents: {count}")
            
            ml_models["vector_store"] = vs
            ml_models["chroma_client"] = chroma_client
            
        except Exception as e:
            print(f"FAILED: ChromaDB init - {e}")
            traceback.print_exc()
            ml_models["vector_store"] = None

    # 3) Load data into ChromaDB
    print("\n[3/4] Loading Data...")
    
    vs = ml_models.get("vector_store")
    if vs:
        try:
            count = vs._collection.count()
            print(f"Current document count: {count}")
            
            if count == 0:
                print("Loading CSV data...")
                df = load_csv()
                
                if df is not None and len(df) > 0:
                    # Sample for faster initialization
                    if len(df) > 1000:
                        df = df.sample(n=1000, random_state=42)
                        print(f"Sampled {len(df)} rows")
                    
                    # Create document texts
                    print("Creating documents...")
                    docs = []
                    for idx, row in df.iterrows():
                        doc = " | ".join([f"{col}: {str(row[col])[:100]}" for col in df.columns])
                        docs.append(doc)
                    
                    print(f"Created {len(docs)} documents")
                    
                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=80
                    )
                    
                    # Add to vector store in batches
                    print("Adding to vector store...")
                    batch_size = 100
                    
                    for i in range(0, len(docs), batch_size):
                        batch_docs = docs[i:i+batch_size]
                        batch_text = "\n\n".join(batch_docs)
                        chunks = splitter.split_text(batch_text)
                        
                        vs.add_texts(texts=chunks)
                        print(f"  Batch {i//batch_size + 1}: {len(chunks)} chunks added")
                    
                    final_count = vs._collection.count()
                    print(f"SUCCESS: Vector store now has {final_count} documents")
                    
                else:
                    print("FAILED: No data to load")
            else:
                print(f"SUCCESS: Vector store already has {count} documents")
                
        except Exception as e:
            print(f"FAILED: Data loading - {e}")
            traceback.print_exc()
    else:
        print("FAILED: No vector store available")

    # 4) Initialize RAG chain
    print("\n[4/4] Initializing RAG Chain...")
    
    ml_models["rag_chain"] = None
    
    if not HF_TOKEN:
        print("FAILED: HUGGINGFACEHUB_API_TOKEN not set")
    elif not vs:
        print("FAILED: Vector store not available")
    else:
        try:
            count = vs._collection.count()
            if count == 0:
                print("WARNING: Vector store is empty!")
            else:
                print(f"Vector store has {count} documents")
            
            # Create prompt
            prompt_template = """Use the context to answer about cybersecurity.

Context: {context}

Question: {question}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create LLM
            print(f"Creating LLM: {FALLBACK_SMALL_REPO_ID}...")
            
            if USE_NEW_API:
                llm = HuggingFaceEndpoint(
                    repo_id=FALLBACK_SMALL_REPO_ID,
                    huggingfacehub_api_token=HF_TOKEN,
                    temperature=0.5,
                    max_new_tokens=256,
                    timeout=60
                )
            else:
                llm = HuggingFaceHub(
                    repo_id=FALLBACK_SMALL_REPO_ID,
                    task="text2text-generation",
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={"temperature": 0.5, "max_new_tokens": 256}
                )
            
            rag = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vs.as_retriever(search_kwargs={"k": 2}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            ml_models["rag_chain"] = rag
            print("SUCCESS: RAG chain ready")
            
        except Exception as e:
            print(f"FAILED: RAG chain - {e}")
            traceback.print_exc()

    # Final status
    print("\n" + "="*70)
    print("STARTUP COMPLETE")
    print("="*70)
    print(f"Embeddings:    {bool(ml_models.get('embedding_function'))}")
    print(f"Vector Store:  {bool(ml_models.get('vector_store'))}")
    print(f"RAG Chain:     {bool(ml_models.get('rag_chain'))}")
    print(f"HF Token:      {bool(HF_TOKEN)}")
    print(f"Using New API: {USE_NEW_API}")
    
    if vs:
        try:
            print(f"Documents:     {vs._collection.count()}")
        except:
            pass
    
    print("="*70 + "\n")
    
    yield
    
    print("Shutting down...")
    ml_models.clear()

# ---------- FastAPI App ----------
app = FastAPI(
    title="RAG Service with ChromaDB",
    version="3.0.0",
    lifespan=lifespan
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "RAG with ChromaDB",
        "version": "3.0.0"
    }

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    doc_count = 0
    
    if vs:
        try:
            doc_count = vs._collection.count()
        except:
            pass
    
    return {
        "vector_store_ready": bool(vs),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN),
        "using_new_api": USE_NEW_API,
        "document_count": doc_count
    }

@app.get("/health")
def health():
    vs = ml_models.get("vector_store")
    doc_count = 0
    
    if vs:
        try:
            doc_count = vs._collection.count()
        except:
            pass
    
    return {
        "status": "healthy" if ml_models.get("rag_chain") else "degraded",
        "components": {
            "embeddings": bool(ml_models.get("embedding_function")),
            "vector_store": bool(vs),
            "rag_chain": bool(ml_models.get("rag_chain"))
        },
        "document_count": doc_count
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not ml_models.get("rag_chain"):
        raise HTTPException(
            status_code=503,
            detail="RAG chain not available. Check /health"
        )
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        print(f"Query: {req.query[:80]}...")
        # RetrievalQA in this setup expects the `query` key
        result = ml_models["rag_chain"].invoke({"query": req.query})
        
        if isinstance(result, dict):
            answer = result.get("result", result.get("answer", str(result)))
        else:
            answer = str(result)
        
        print(f"Answer generated: {len(answer)} chars")
        return QueryResponse(answer=answer if answer else "No answer")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
