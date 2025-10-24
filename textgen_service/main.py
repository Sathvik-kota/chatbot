"""
CyberGuard AI - Conversational RAG Service
This script implements the complete, stateful architecture for a conversational
chatbot as designed in the research report. It uses ConversationalRetrievalChain
to correctly manage dialogue history, retrieve relevant context, and handle both
domain-specific and general knowledge questions.
"""

import os
import logging
import traceback
import pandas as pd
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- LangChain and Model Imports ---
try:
    from langchain_community.llms import LlamaCpp
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain_core.prompts import PromptTemplate
    from langchain.docstore.document import Document
except ImportError as e:
    print(f"Required libraries are missing: {e}. Please run 'pip install langchain langchain_community faiss-cpu sentence-transformers llama-cpp-python pandas fastapi uvicorn'")
    exit(1)

# --- Configuration ---
# Setup logging for better diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths and Model Settings ---
# IMPORTANT: Update these paths to match your environment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# Model path is set to a local 'models' folder
MODEL_PATH = os.getenv("MODEL_PATH", "./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf") 

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(ROOT_DIR, "vectorstores/cyber_faiss"))
CYBER_CSV_PATH = os.getenv("CYBER_CSV_PATH", "/content/project/textgen_service/ai_cybersecurity_dataset-sampled-5k.csv")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- Global In-Memory Stores ---
# These dictionaries will hold the loaded models and active conversation chains.
ml_models: Dict[str, Any] = {}

# --- Data Ingestion Logic ---
def ingest_cybersecurity_csv(csv_path: str, vector_store_path: str, embedding_model) -> FAISS:
    """
    One-time process to load the cybersecurity CSV, process it into documents,
    and create a FAISS vector store.
    """
    logging.info(f"Starting data ingestion from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        # Standardize column names for easier access (e.g., "Attack Type" -> "attack_type")
        df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
        logging.info(f"Successfully loaded CSV with {len(df)} rows. Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at {csv_path}. The service cannot build its knowledge base.")
        raise
    except Exception as e:
        logging.error(f"FATAL: Failed to load or parse CSV file: {e}")
        raise

    # Define which columns are most useful for creating document context
    USEFUL_COLUMNS = [
        "attack_type", "attack_severity", "data_exfiltrated", 
        "threat_intelligence", "response_action", "user_agent"
    ]
    
    documents: List[Document] = [] 
    for _, row in df.iterrows():
        # Construct a descriptive sentence for each event. This creates better context for the RAG system.
        content_parts = ["A cybersecurity event was recorded."]
        for col in USEFUL_COLUMNS:
            if col in df.columns and pd.notna(row[col]):
                # Format the column name back to a readable format for the LLM
                col_name_readable = col.replace('_', ' ').title()
                content_parts.append(f"{col_name_readable}: {row[col]}.")
        
        page_content = " ".join(content_parts)
        
        # Create a LangChain Document object
        documents.append(Document(
            page_content=page_content,
            metadata={
                "source_csv": os.path.basename(csv_path),
                "event_id": row.get("event_id", "N/A"),
                "timestamp": row.get("timestamp", "N/A")
            }
        ))

    if not documents:
        logging.error("FATAL: No documents were created from the CSV. Check the CSV content and column names.")
        raise ValueError("No documents to ingest.")

    logging.info(f"Created {len(documents)} documents. Now creating vector store...")
    
    # Create and save the FAISS vector store
    db = FAISS.from_documents(documents, embedding_model)
    os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
    db.save_local(vector_store_path)
    logging.info(f"Vector store successfully created and saved at {vector_store_path}")
    return db

# --- Core Logic: Conversational Chain ---

# This prompt is designed to rephrase a follow-up question into a standalone one,
# which is crucial for accurate document retrieval.
_template = """
Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
The standalone question should be in the same language as the follow-up question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# This prompt guides the LLM on how to answer. It's explicitly told to use the
# retrieved context if available, but to fall back to its general knowledge if not.
# This makes the chatbot versatile.
qa_template = """
You are "CyberGuard AI", a helpful and knowledgeable Cybersecurity Expert Assistant.
Your goal is to provide accurate and helpful answers based on the information provided.

Use the provided context from cybersecurity event logs to answer the question.
If the context is empty or not relevant to the question, answer the question using your general knowledge.

When answering, maintain a professional and helpful tone. If you don't know the answer, state that clearly.

Context:
{context}

Chat History:
{chat_history}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate.from_template(qa_template)

def create_conversational_chain(vector_store_retriever) -> ConversationalRetrievalChain:
    """
    Creates and returns a stateful ConversationalRetrievalChain.
    """
    logging.info("Creating a new conversational chain instance.")
    
    # Correctly configured memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",    # Tells memory the human's input is in the "question" key
        output_key="answer"      # Tells memory the AI's output is in the "answer" key
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=ml_models["llm"],
        retriever=vector_store_retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False
    )
    return chain

# --- FastAPI Application Lifespan (Startup and Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup logic: loading models and building the vector store if needed.
    """
    logging.info("Application startup sequence initiated...")
    
    # This dictionary will store active conversation chains, mapped by session_id.
    # This is the core of our application's state management.
    ml_models["conversation_chains"] = {}
    
    try:
        # 1. Load Embedding Model
        logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        ml_models["embeddings"] = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Change to 'cuda' for GPU
        )

        # 2. Load or Create Vector Store
        if os.path.exists(VECTOR_STORE_PATH):
            logging.info(f"Loading existing vector store from: {VECTOR_STORE_PATH}")
            ml_models["vector_store"] = FAISS.load_local(
                VECTOR_STORE_PATH, 
                ml_models["embeddings"], 
                allow_dangerous_deserialization=True
            )
        else:
            logging.warning(f"Vector store not found at {VECTOR_STORE_PATH}. Triggering one-time ingestion...")
            ml_models["vector_store"] = ingest_cybersecurity_csv(
                CYBER_CSV_PATH, 
                VECTOR_STORE_PATH, 
                ml_models["embeddings"]
            )

        # 3. Load LLM
        logging.info(f"Loading LLM from: {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"LLM model not found at {MODEL_PATH}. Please download the model file.")
        ml_models["llm"] = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, n_batch=512, n_ctx=4096, f16_kv=True, verbose=False
        )
        
        logging.info("--- CyberGuard AI is online and ready. ---")
    
    except Exception as e:
        logging.error(f"FATAL ERROR during startup: {e}")
        logging.error(traceback.format_exc())
        # The application cannot run without its core components.
        exit(1)

    yield # The application is now running

    # --- Shutdown Logic ---
    logging.info("Application shutdown sequence initiated...")
    ml_models.clear()
    logging.info("--- CyberGuard AI is offline. ---")


# --- FastAPI Application and Endpoints ---

app = FastAPI(
    title="CyberGuard AI - Conversational RAG Service",
    description="A stateful chatbot for cybersecurity and general knowledge questions.",
    lifespan=lifespan
)

# Pydantic models for type-safe API requests and responses
class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    source_documents: List[dict]

@app.get("/", tags=["Root"])
def root():
    return {"status": "CyberGuard AI is running."}

@app.get("/status", tags=["Status"])
def status():
    return {
        "llm_loaded": ml_models.get("llm") is not None,
        "vector_store_loaded": ml_models.get("vector_store") is not None,
        "active_conversations": len(ml_models.get("conversation_chains", {})),
    }

@app.post("/chat", response_model=ChatResponse, tags=["Conversation"])
async def chat(request: ChatRequest):
    """
    Handles a chat request, maintaining conversation state using session_id.
    """
    logging.info(f"Received chat request for session_id: {request.session_id}")
    
    if not request.question or not request.session_id:
        raise HTTPException(status_code=400, detail="Both 'question' and 'session_id' are required.")

    conversation_chains = ml_models["conversation_chains"]
    
    # Get or create the conversational chain for the given session_id
    if request.session_id not in conversation_chains:
        logging.info(f"Creating new conversation chain for session_id: {request.session_id}")
        retriever = ml_models["vector_store"].as_retriever(search_kwargs={"k": 3})
        conversation_chains[request.session_id] = create_conversational_chain(retriever)
    
    qa_chain = conversation_chains[request.session_id]

    try:
        # Asynchronously invoke the chain to get the answer
        result = await qa_chain.ainvoke({"question": request.question})
        
        # Format source documents for a clean API response
        sources = [] # Initialize empty list
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

        return ChatResponse(
            answer=result["answer"].strip(),
            source_documents=sources
        )

    except Exception as e:
        logging.error(f"Error during chain invocation for session {request.session_id}: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during chain invocation: {e}")

@app.post("/reset", tags=["Conversation"])
async def reset(session_id: str):
    """
    Resets and clears the conversation history for a given session_id.
    """
    if session_id in ml_models["conversation_chains"]:
        del ml_models["conversation_chains"][session_id]
        logging.info(f"Conversation history for session_id '{session_id}' has been reset.")
        return {"message": f"Conversation for session '{session_id}' reset successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

if __name__ == "__main__":
    import uvicorn
    # To run this script:
    # 1. Make sure you have a GGUF model file at the location specified in MODEL_PATH.
    # 2. Make sure your cybersecurity CSV is at the location specified in CYBER_CSV_PATH.
    # 3. Run from your terminal: uvicorn main:app --host 0.0.0.0 --port 8002
    uvicorn.run(app, host="0.0.0.0", port=8002)

