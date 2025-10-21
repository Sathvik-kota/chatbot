# imggen_service/main.py
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Image Generation Service")

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Use a hosted HF model — change if you prefer another
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

class ImgReq(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status": "imggen running"}

@app.post("/generate-image")
def generate_image(req: ImgReq):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Hugging Face token not configured on server.")
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Hugging Face request failed: {e}")

    # If HF returned an image payload directly
    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code == 200 and content_type.startswith("image/"):
        return Response(content=resp.content, media_type=content_type)

    # If HF returned JSON error or text, forward it
    try:
        err = resp.json()
    except Exception:
        err = resp.text
    raise HTTPException(status_code=502, detail={"status_code": resp.status_code, "error": err})
