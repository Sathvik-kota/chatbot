import io
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Segmentation (fallback) Service")

@app.get("/")
def root():
    return {"status":"segmentation running (kmeans fallback)"}

@app.post("/segment-image/")
async def segment_image(prompt: str = Form(...), image: UploadFile = File(...)):
    # simple validation
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_np = np.array(img)
    h,w,_ = img_np.shape

    # Resize small to speed up
    scale = 600 / max(h,w) if max(h,w) > 600 else 1.0
    if scale != 1.0:
        img_small = cv2.resize(img_np, (int(w*scale), int(h*scale)))
    else:
        img_small = img_np.copy()

    # KMeans color segmentation on small image
    Z = img_small.reshape((-1,3)).astype(np.float32)
    K = 3  # number of clusters (you can change or expose)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()].reshape(img_small.shape)

    # Choose the cluster to highlight — simple heuristic: largest cluster
    labels, counts = np.unique(label, return_counts=True)
    largest_label = labels[np.argmax(counts)]
    mask_small = (label.flatten() == largest_label).reshape(img_small.shape[0], img_small.shape[1])

    # Upscale mask to original size if resized
    if scale != 1.0:
        mask = cv2.resize(mask_small.astype(np.uint8)*255, (w,h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = (mask_small.astype(np.uint8))*255

    # Create overlay
    overlay = img_np.copy()
    colored = np.zeros_like(overlay)
    colored[mask>0] = (0, 0, 255)  # red mask
    alpha = 0.6
    overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

    # Return image
    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
