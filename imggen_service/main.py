import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # set this in Colab or env
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

app = FastAPI(title="Image Generation Service")

class ImgReq(BaseModel):
    prompt: str

@app.get("/")
def root():
    return {"status":"imggen running"}

@app.post("/generate-image")
def generate_image(req: ImgReq):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF token not configured in environment")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt empty")

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": req.prompt}
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=str(e))

    if resp.status_code == 200 and resp.headers.get("Content-Type","").startswith("image/"):
        return Response(content=resp.content, media_type=resp.headers["Content-Type"])
    else:
        # return HF error details
        detail = resp.text
        raise HTTPException(status_code=502, detail=f"Hugging Face returned error: {detail}")
