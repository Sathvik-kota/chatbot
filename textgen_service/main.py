import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import pandas as pd
import chromadb
from chromadb.config import Settings

# Initialize FastAPI
app = FastAPI(title="Cybersecurity RAG Service")

# ------------------ GLOBAL VARIABLES ------------------
ml_models = {
    "embedding_function": None,
    "vector_store": None,
    "llm": None,
}
DB_DIR = "./chroma_db"
DATA_FILE = "./data.csv"  # path to your CSV file

# ------------------ SETUP FUNCTIONS ------------------
def create_embedding_function():
    """Load sentence-transformer embeddings"""
    print("🔹 Loading embedding model...")
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def init_chroma(embedding_fn):
    """Initialize or load Chroma vector store"""
    print("🔹 Initializing Chroma DB...")
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.Client(Settings(persist_directory=DB_DIR))
    try:
        vs = Chroma(
            collection_name="cyber_docs",
            embedding_function=embedding_fn,
            persist_directory=DB_DIR,
        )
        print("✅ Chroma initialized")
        return vs
    except Exception as e:
        print("❌ Chroma init failed:", e)
        return None

def load_llm():
    """Load HuggingFace model"""
    print("🔹 Loading LLM...")
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.3, "max_length": 512},
        )
        print("✅ LLM loaded successfully")
        return llm
    except Exception as e:
        print("❌ LLM loading failed:", e)
        return None

def ingest_csv_to_chroma(vs, embedding_fn, file_path):
    """Read CSV and ingest into Chroma"""
    if not os.path.exists(file_path):
        print("❌ CSV not found at", file_path)
        return 0

    print("📄 Reading CSV:", file_path)
    df = pd.read_csv(file_path)
    all_text = []

    for col in df.columns:
        df[col] = df[col].astype(str)
    for i, row in df.iterrows():
        text = " ".join(row.values)
        all_text.append(text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = [Document(page_content=t) for t in all_text]
    split_docs = text_splitter.split_documents(docs)

    print(f"🧩 Ingesting {len(split_docs)} chunks into Chroma...")
    vs.add_documents(split_docs)
    vs.persist()
    print("✅ Ingestion complete.")
    return len(split_docs)

# ------------------ PROMPT TEMPLATE ------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cybersecurity expert. Use the given context to answer accurately, clearly, and concisely."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

# ------------------ FASTAPI LIFESPAN ------------------
@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    try:
        embedding_fn = create_embedding_function()
        ml_models["embedding_function"] = embedding_fn
        vs = init_chroma(embedding_fn)
        ml_models["vector_store"] = vs
        llm = load_llm()
        ml_models["llm"] = llm
    except Exception as e:
        traceback.print_exc()
        print("❌ Startup initialization failed:", e)

# ------------------ ROUTES ------------------

@app.get("/status")
async def status():
    vs = ml_models.get("vector_store")
    llm = ml_models.get("llm")
    embed = ml_models.get("embedding_function")
    count = 0
    if vs:
        try:
            count = len(vs.get()["ids"])
        except Exception:
            count = 0
    return {
        "vector_store_ready": bool(vs),
        "vector_store_count": count,
        "llm_ready": bool(llm),
        "embedding_ready": bool(embed),
        "model_id": "google/flan-t5-base",
        "top_k": 6,
    }

@app.post("/force-ingest")
async def force_ingest():
    vs = ml_models.get("vector_store")
    embedding_fn = ml_models.get("embedding_function")

    if not vs or not embedding_fn:
        raise HTTPException(status_code=503, detail="Vectorstore or embedding not ready")

    try:
        count = ingest_csv_to_chroma(vs, embedding_fn, DATA_FILE)
        return {"status": "ok", "ingested_chunks": count}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-text")
async def generate_text(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()

    if not query:
        raise HTTPException(status_code=400, detail="Missing query")

    llm = ml_models.get("llm")
    vs = ml_models.get("vector_store")

    if not llm or not ml_models.get("embedding_function"):
        raise HTTPException(status_code=503, detail="LLM or embedding not ready")

    # Retrieve context
    context_text = ""
    if vs:
        retriever = vs.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(query)
        if docs:
            context_text = "\n\n".join([d.page_content for d in docs])
    else:
        context_text = "No context available."

    # Format prompt
    formatted_prompt = prompt.format(context=context_text, question=query)

    # Generate answer
    try:
        response = llm.invoke(formatted_prompt)
    except Exception as e:
        print("LLM generation error:", e)
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return {"answer": response}

# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
