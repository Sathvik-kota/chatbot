# textgen_service/main.py
# LangChain-first RAG service (uses langchain + langchain_huggingface + langchain_text_splitters + chromadb)
import os
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# LangChain imports (modern)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use official HuggingFace embeddings integration for LangChain
try:
    # preferred: langchain_huggingface integration
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # fallback: try langchain.embeddings (older/newer layouts vary)
    try:
        from langchain.embeddings import HuggingFaceEmbeddings  # some langchain releases expose this
    except Exception:
        HuggingFaceEmbeddings = None

# HuggingFace LLM (LangChain wrapper)
try:
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceHub
    USE_NEW_API = True
except Exception:
    # In case langchain_huggingface exposes only HuggingFaceHub or different names,
    # we'll import what we can at runtime below.
    try:
        from langchain_huggingface import HuggingFaceHub
        HuggingFaceEndpoint = None
        USE_NEW_API = False
    except Exception:
        HuggingFaceEndpoint = None
        HuggingFaceHub = None
        USE_NEW_API = False

# Chroma client (we'll use the langchain.wrapper for Chroma which uses chromadb under the hood)
import chromadb
from chromadb.config import Settings

# ---------- Configuration ----------
ROOT_DIR = os.path.dirname(__file__) or os.getcwd()
CHROMA_DB_PATH = os.path.join(ROOT_DIR, "chroma_db")
DOCUMENTS_DIR = os.path.join(ROOT_DIR, "documents")
KNOWLEDGE_FILE_PATH = os.path.join(DOCUMENTS_DIR, "ai_cybersecurity_dataset-sampled-5k.csv")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # model name for HuggingFaceEmbeddings
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

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
    print("STARTING SERVICE (LangChain + Chroma)")
    print("="*70)
    # 1) Embeddings (LangChain wrapper -> uses sentence-transformers under the hood)
    print("\n[1/4] Initializing embeddings via HuggingFaceEmbeddings...")
    emb = None
    if HuggingFaceEmbeddings is None:
        print("WARNING: HuggingFaceEmbeddings integration not available (import failed). Will try to use sentence-transformers directly later.")
    else:
        try:
            emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            ml_models["embedding"] = emb
            print("SUCCESS: LangChain HuggingFaceEmbeddings ready")
        except Exception as e:
            print(f"FAILED: HuggingFaceEmbeddings init - {e}")
            traceback.print_exc()
            ml_models["embedding"] = None

    # 2) Initialize chroma persistent client (langchain's Chroma wrapper uses chromadb)
    print("\n[2/4] Initializing ChromaDB persistent client and LangChain Chroma vector store...")
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(
            path=CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        ml_models["chroma_client"] = chroma_client
        print("SUCCESS: chromadb PersistentClient created")

        # Create or attach a collection using the langchain Chroma wrapper.
        # We'll construct the vectorstore with the embedding object if available.
        if ml_models.get("embedding"):
            # use LangChain's Chroma wrapper and pass embedding object
            vs = Chroma(
                collection_name="cybersecurity_docs",
                persist_directory=CHROMA_DB_PATH,
                embedding_function=ml_models["embedding"]
            )
        else:
            # Fallback: create Chroma wrapper without LangChain embeddings - we'll add raw embeddings if needed
            vs = Chroma(
                collection_name="cybersecurity_docs",
                persist_directory=CHROMA_DB_PATH
            )

        ml_models["vector_store"] = vs
        print("SUCCESS: LangChain Chroma vector store initialized")
    except Exception as e:
        print(f"FAILED: Chroma init - {e}")
        traceback.print_exc()
        ml_models["vector_store"] = None

    # 3) Load CSV into the Chroma vector store (if empty)
    print("\n[3/4] Loading data into vector store (if empty)...")
    vs = ml_models.get("vector_store")
    if not vs:
        print("SKIP: No vector store available")
    else:
        try:
            # Try to get document count via the wrapped collection
            try:
                cur_count = vs._collection.count()
            except Exception:
                # fallback (some langchain/chroma versions differ)
                cur_count = 0
            print(f"Current document count: {cur_count}")

            if cur_count == 0:
                df = load_csv()
                if df is not None and len(df) > 0:
                    if len(df) > 1000:
                        df = df.sample(n=1000, random_state=42)
                        print(f"Sampled {len(df)} rows")

                    # create raw texts
                    docs = []
                    for idx, row in df.iterrows():
                        doc = " | ".join([f"{col}: {str(row[col])[:200]}" for col in df.columns])
                        docs.append(doc)

                    # split with LangChain text splitter
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                    texts = []
                    for d in docs:
                        # create_documents returns Document objects in some versions; use split_text
                        try:
                            chunks = splitter.split_text(d)
                        except Exception:
                            # older/newer API name differences
                            try:
                                chunks = splitter.create_documents([d])
                                # extract page_content if Document objects
                                chunks = [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks]
                            except Exception:
                                # fallback naive split
                                chunks = [d[i:i+800] for i in range(0, len(d), 720)]
                        texts.extend(chunks)

                    print(f"Total chunks to add: {len(texts)}")

                    # Add to vector store via langchain Chroma wrapper
                    # If embedding object available, Chroma.from_texts will compute embeddings via it
                    try:
                        vs.add_texts(texts=texts)
                    except Exception:
                        # fallback: use from_documents flow
                        try:
                            from langchain.schema import Document
                            docs_for_store = [Document(page_content=t) for t in texts]
                            vs.add_documents(docs_for_store)
                        except Exception as ex:
                            print("Failed adding texts via both add_texts and add_documents:", ex)
                            raise

                    try:
                        print("Documents added. New count:", vs._collection.count())
                    except Exception:
                        print("Documents added.")
                else:
                    print("No CSV data to load.")
            else:
                print("Vector store already populated.")
        except Exception as e:
            print(f"FAILED: Data loading - {e}")
            traceback.print_exc()

    # 4) Initialize the RAG chain (RetrievalQA) using a small fallback LLM
    print("\n[4/4] Initializing retrieval chain (LLM + retriever)...")
    ml_models["rag_chain"] = None
    try:
        retriever = vs.as_retriever(search_kwargs={"k": 2})
        # Create prompt template
        prompt_template = """Use the context to answer about cybersecurity.

Context: {context}

Question: {question}

Answer:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # Create LLM via langchain_huggingface (preferred) or fallback to HFHub wrapper
        if HF_TOKEN and (HuggingFaceEndpoint is not None or HuggingFaceHub is not None):
            # Prefer the endpoint wrapper (if available)
            try:
                if HuggingFaceEndpoint is not None:
                    llm = HuggingFaceEndpoint(repo_id=FALLBACK_SMALL_REPO_ID, huggingfacehub_api_token=HF_TOKEN,
                                              temperature=0.5, max_new_tokens=256, timeout=60)
                else:
                    # older naming
                    llm = HuggingFaceHub(repo_id=FALLBACK_SMALL_REPO_ID, huggingfacehub_api_token=HF_TOKEN,
                                         task="text2text-generation", model_kwargs={"temperature": 0.5, "max_new_tokens": 256})
            except Exception as e:
                print("Warning: failed to instantiate HuggingFace-backed LLM via langchain_huggingface:", e)
                llm = None
        else:
            llm = None

        # If no HF token / LLM, fallback to a simple local small HF model via transformers (wrapped with LangChain)
        if llm is None:
            try:
                # Use langchain's wrapper for local transformers if available
                from langchain.llms import HuggingFaceHub as _HFLocalWrapper
                # try to instantiate a local wrapper (this may still download a model)
                llm = _HFLocalWrapper(repo_id=FALLBACK_SMALL_REPO_ID, model_kwargs={"temperature": 0.5, "max_new_tokens": 256})
            except Exception:
                # Last fallback: set llm to None; RetrievalQA.from_chain_type will raise if llm is None
                llm = None

        if llm is None:
            print("WARNING: No LLM available (HuggingFace token missing or local wrapper failed). RAG chain will not be created.")
        else:
            rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})
            ml_models["rag_chain"] = rag
            print("SUCCESS: RAG chain ready")
    except Exception as e:
        print(f"FAILED: building RAG chain - {e}")
        traceback.print_exc()
        ml_models["rag_chain"] = None

    print("\n" + "="*70)
    print("STARTUP COMPLETE")
    print("="*70)
    print(f"Embedding ready:   {bool(ml_models.get('embedding'))}")
    print(f"Vector store ready:{bool(ml_models.get('vector_store'))}")
    print(f"RAG chain ready:   {bool(ml_models.get('rag_chain'))}")
    print("="*70 + "\n")

    yield

    print("Shutting down... clearing models")
    ml_models.clear()


# ---------- FastAPI App ----------
app = FastAPI(title="RAG Service (LangChain)", version="1.0.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def root():
    return {"status": "running", "service": "RAG with LangChain", "version": "1.0.0"}

@app.get("/status")
def status():
    vs = ml_models.get("vector_store")
    doc_count = 0
    try:
        if vs:
            doc_count = vs._collection.count()
    except Exception:
        doc_count = 0
    return {
        "vector_store_ready": bool(vs),
        "rag_chain_ready": bool(ml_models.get("rag_chain")),
        "hf_token_set": bool(HF_TOKEN),
        "document_count": doc_count
    }

@app.post("/generate-text", response_model=QueryResponse)
async def generate_text(req: QueryRequest):
    if not ml_models.get("rag_chain"):
        raise HTTPException(status_code=503, detail="RAG chain not available. Check /status")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        print("Query:", req.query[:120])
        result = ml_models["rag_chain"].run({"query": req.query})
        # RetrievalQA returns a dict in many versions; try to safely extract
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or str(result)
        else:
            answer = str(result)
        print("Answer length:", len(answer))
        return QueryResponse(answer=answer or "No answer")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
