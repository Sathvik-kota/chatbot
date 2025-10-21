# textgen_service/main.py
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Import new HuggingFace API
try:
    from langchain_huggingface import HuggingFaceEndpoint
    USE_NEW_API = True
    print("✓ Using NEW HuggingFaceEndpoint API")
except ImportError:
    from langchain_community.llms import HuggingFaceHub
    USE_NEW_API = False
    print("⚠ Using deprecated HuggingFaceHub API - please install langchain-huggingface")

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
        print(f"❌ Knowledge CSV not found: {KNOWLEDGE_FILE_PATH}")
        return None
    try:
        df = pd.read_csv(KNOWLEDGE_FILE_PATH)
        df.columns = df.columns.str.strip()
        print(f"✓ Loaded CSV with {len(df)} rows, columns: {list(df.columns)[:5]}...")
        return df
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        traceback.print_exc()
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("="*60)
    print("🚀 Starting up textgen service...")
    print("="*60)

    # 1) Initialize embedding function
    try:
        print("📦 Initializing embedding function...")
        emb_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        ml_models["embedding_function"] = emb_fn
        print(f"✓ Embedding function initialized: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        traceback.print_exc()
        ml_models["embedding_function"] = None

    # 2) Initialize Chroma vector store (FIXED)
    vs = None
    if ml_models.get("embedding_function"):
        try:
            print("📊 Initializing Chroma vector store...")
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)

            # Create Chroma with proper settings
            vs = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=ml_models["embedding_function"],
                collection_name="cybersecurity_docs"
            )
            ml_models["vector_store"] = vs
            print(f"✓ Persistent Chroma initialized at: {CHROMA_DB_PATH}")

        except Exception as e:
            print(f"⚠ Persistent Chroma failed: {e}")
            print("Trying in-memory fallback...")
            try:
                vs = Chroma(
                    embedding_function=ml_models["embedding_function"],
                    collection_name="cybersecurity_docs"
                )
                ml_models["vector_store"] = vs
                print("✓ In-memory Chroma initialized")
            except Exception as e2:
                print(f"❌ Chroma initialization completely failed: {e2}")
                traceback.print_exc()
                ml_models["vector_store"] = None
    else:
        print("❌ Cannot initialize vector store without embeddings")
        ml_models["vector_store"] = None

    # 3) Ingest CSV if vector store is empty (FIXED)
    vs = ml_models.get("vector_store")
    if vs:
        try:
            print("📚 Checking vector store contents...")
            collection_data = vs._collection.get(include=['documents'])
            existing_docs = collection_data.get('documents', [])

            if len(existing_docs) == 0:
                print("📥 Vector store is empty. Loading data...")
                df = load_csv()

                if df is not None and len(df) > 0:
                    print(f"🔄 Processing {len(df)} rows...")

                    # Convert each row to text representation
                    docs = []
                    for idx, row in df.iterrows():
                        doc_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                        docs.append(doc_text)

                    # Split into chunks
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100,
                        length_function=len
                    )
                    chunks = splitter.split_text("\n\n".join(docs))

                    # Add to vector store
                    print(f"💾 Adding {len(chunks)} chunks to vector store...")
                    vs.add_texts(texts=chunks)
                    vs.persist()
                    print(f"✓ Successfully ingested {len(docs)} documents → {len(chunks)} chunks")
                else:
                    print("⚠ No data to ingest")
            else:
                print(f"✓ Vector store already contains {len(existing_docs)} documents")

        except Exception as e:
            print(f"❌ Vector store ingestion failed: {e}")
            traceback.print_exc()
    else:
        print("⚠ No vector store available for ingestion")

    # 4) Initialize RAG chain (FIXED for new API)
    ml_models["rag_chain"] = None

    if not HF_TOKEN:
        print("❌ HUGGINGFACEHUB_API_TOKEN not set")
    elif not vs:
        print("❌ Vector store unavailable - cannot create RAG chain")
    else:
        print("🤖 Initializing RAG chain...")

        # Create prompt template
        prompt_template = """Use the following cybersecurity context to answer the question accurately.
If you don't know the answer based on the context, say so clearly.

Context: {context}

Question: {question}

Detailed Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        def try_create_llm_new(repo_id):
            """Create LLM with new HuggingFaceEndpoint API"""
            try:
                print(f"  Trying {repo_id} with HuggingFaceEndpoint...")
                llm = HuggingFaceEndpoint(
                    repo_id=repo_id,
                    huggingfacehub_api_token=HF_TOKEN,
                    temperature=0.5,
                    max_new_tokens=1024,
                    timeout=120
                )

                rag = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vs.as_retriever(search_kwargs={"k": 3}),
                    chain_type_kwargs={"prompt": PROMPT},
                    return_source_documents=False
                )
                print(f"  ✓ Successfully initialized with {repo_id}")
                return rag
            except Exception as e:
                print(f"  ✗ Failed with {repo_id}: {str(e)[:100]}")
                return None

        def try_create_llm_old(repo_id, task_type):
            """Create LLM with old HuggingFaceHub API (deprecated)"""
            try:
                print(f"  Trying {repo_id} with HuggingFaceHub (deprecated)...")
                llm = HuggingFaceHub(
                    repo_id=repo_id,
                    task=task_type,
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={"temperature": 0.5, "max_new_tokens": 1024}
                )

                rag = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vs.as_retriever(search_kwargs={"k": 3})
                )
                print(f"  ✓ Successfully initialized with {repo_id} (old API)")
                return rag
            except Exception as e:
                print(f"  ✗ Failed with {repo_id}: {str(e)[:100]}")
                return None

        # Try to create RAG chain
        rag = None
        if USE_NEW_API:
            rag = try_create_llm_new(PRIMARY_REPO_ID)
            if not rag:
                print("  ⚠ Primary model failed, trying fallback...")
                rag = try_create_llm_new(FALLBACK_SMALL_REPO_ID)
        else:
            rag = try_create_llm_old(PRIMARY_REPO_ID, "text-generation")
            if not rag:
                print("  ⚠ Primary model failed, trying fallback...")
                rag = try_create_llm_old(FALLBACK_SMALL_REPO_ID, "text2text-generation")

        ml_models["rag_chain"] = rag

        if rag:
            print("✓ RAG chain initialized successfully")
        else:
            print("❌ All RAG chain initialization attempts failed")

    # Print final status
    print("="*60)
    print("📊 STARTUP COMPLETE - Service Status:")
    print("="*60)
    print(f"  Embeddings Ready:    {bool(ml_models.get('embedding_function'))}")
    print(f"  Vector Store Ready:  {bool(ml_models.get('vector_store'))}")
    print(f"  RAG Chain Ready:     {bool(ml_models.get('rag_chain'))}")
    print(f"  HF Token Set:        {bool(HF_TOKEN)}")
    print(f"  Using New API:       {USE_NEW_API}")
    print("="*60)

    yield

    print("🛑 Shutting down...")
    ml_models.clear()
    print("✓ Shutdown complete")

# ---------- FastAPI app ----------
app = FastAPI(
    title="Text Generation RAG Service",
    description="RAG service for cybersecurity question answering",
    version="2.0.0",
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
        "service": "Text Generation (RAG) Service",
        "version": "2.0.0"
    }

@app.get("/status")
def status():
    """Get service status"""
    return {
        "vector_store_ready": bool(ml_models.get("vector_store")),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN),
        "using_new_api": USE_NEW_API,
        "embedding_model": EMBEDDING_MODEL_NAME
    }

@app.get("/health")
def health():
    """Detailed health check"""
    vs = ml_models.get("vector_store")
    doc_count = 0

    if vs:
        try:
            collection_data = vs._collection.get(include=['documents'])
            doc_count = len(collection_data.get('documents', []))
        except:
            pass

    return {
        "status": "healthy" if ml_models.get("rag_chain") else "degraded",
        "service": "textgen",
        "components": {
            "embeddings": bool(ml_models.get("embedding_function")),
            "vector_store": bool(ml_models.get("vector_store")),
            "rag_chain": bool(ml_models.get("rag_chain"))
        },
        "vector_store_documents": doc_count
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    """Generate answer using RAG"""

    if not ml_models.get("rag_chain"):
        raise HTTPException(
            status_code=503,
            detail="RAG chain not available. Check /status endpoint."
        )

    if not req.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    try:
        print(f"\n🔍 Processing query: {req.query[:100]}...")

        # Use invoke method (compatible with both old and new APIs)
        result = ml_models["rag_chain"].invoke({"query": req.query})

        # Handle different response formats
        if isinstance(result, dict):
            answer = result.get("result", result.get("answer", str(result)))
        else:
            answer = str(result)

        print(f"✓ Generated answer ({len(answer)} chars)")

        return QueryResponse(
            answer=answer if answer else "No answer could be generated."
        )

    except Exception as e:
        print(f"❌ Generation error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

@app.get("/vector-store-info")
def vector_store_info():
    """Get vector store information"""
    vs = ml_models.get("vector_store")

    if not vs:
        return {"error": "Vector store not initialized"}

    try:
        collection_data = vs._collection.get(include=['documents', 'metadatas'])
        docs = collection_data.get('documents', [])

        return {
            "status": "ready",
            "document_count": len(docs),
            "sample_doc": docs[0][:200] + "..." if docs else None,
            "collection_name": vs._collection.name
        }
    except Exception as e:
        return {"error": str(e)}