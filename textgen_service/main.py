# textgen_service/main.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Text Generation Service")

# small CPU-friendly model for demo
generator = pipeline("text2text-generation", model="google/flan-t5-small")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "textgen running"}

@app.post("/generate")
def generate_text(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    try:
        out = generator(q, max_length=200)
        answer = out[0].get("generated_text") or out[0].get("text") or ""
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# alias for compatibility
@app.post("/generate-text")
def generate_text_alias(req: QueryRequest):
    return generate_text(req)
