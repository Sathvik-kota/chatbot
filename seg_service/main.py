# seg_service/main.py
import io
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI(title="Segmentation (fallback) Service")

@app.get("/")
def root():
    return {"status": "segmentation running (kmeans fallback)"}

@app.post("/segment-image/")
async def segment_image(prompt: str = Form(...), image: UploadFile = File(...)):
    # Validate
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    img_np = np.array(img)
    h, w, _ = img_np.shape

    # Resize small for speed
    scale = 600 / max(h, w) if max(h, w) > 600 else 1.0
    if scale != 1.0:
        img_small = cv2.resize(img_np, (int(w * scale), int(h * scale)))
    else:
        img_small = img_np.copy()

    # KMeans segmentation
    try:
        Z = img_small.reshape((-1, 3)).astype(np.float32)
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        label = label.flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KMeans failed: {e}")

    # Choose largest cluster
    labels, counts = np.unique(label, return_counts=True)
    largest_label = labels[np.argmax(counts)]
    mask_small = (label == largest_label).astype(np.uint8).reshape(img_small.shape[0], img_small.shape[1])

    if scale != 1.0:
        mask = cv2.resize(mask_small * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_small * 255

    overlay = img_np.copy()
    colored = np.zeros_like(overlay)
    colored[mask > 0] = (0, 0, 255)
    alpha = 0.6
    overlay = cv2.addWeighted(overlay, 1.0, colored, alpha, 0)

    pil = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
