from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import pipeline

app = FastAPI(title="Text Generation (RAG-lite) Service")

# Use a small text2text model (CPU friendly for demos)
generator = pipeline("text2text-generation", model="google/flan-t5-small")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status":"textgen running"}

@app.post("/generate")
def generate_text(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    # For real RAG you'd retrieve docs first; here we directly ask the model
    out = generator(q, max_length=200)
    answer = out[0].get("generated_text", out[0].get("text", ""))
    return {"answer": answer}
