import os
import sys
import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import wget
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import hydra
from hydra import initialize_config_dir
import traceback
import requests
from io import BytesIO

# -------------------------------
# Paths & sys.path
# -------------------------------
# Ensure this file can be run from seg_service directory.
ROOT_DIR = os.path.dirname(__file__)
SAM2_REPO = os.path.join(ROOT_DIR, "segment-anything-2")

# add sam2 repo to path so "import sam2" works
sys.path.append(os.path.abspath(SAM2_REPO))

# -------------------------------
# Imports from sam2
# -------------------------------
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:
    print("❌ Could not import sam2. Ensure 'segment-anything-2/sam2' exists and contains __init__.py")
    traceback.print_exc()
    build_sam2 = None
    SAM2ImagePredictor = None

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    try:
        print(f"GPU: {torch.cuda.get_device_name()} | Mem: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception:
        pass

# -------------------------------
# Checkpoint helper
# -------------------------------
def download_if_missing(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"✓ {os.path.basename(dest)} already exists")
        return
    print(f"Downloading {os.path.basename(dest)} to {dest} ...")
    wget.download(url, dest)
    print(f"\n✓ Downloaded {os.path.basename(dest)}")

# download/checkpoint path inside seg_service
SAM2_CHECKPOINT = os.path.join(ROOT_DIR, "checkpoints", "sam2.1_hiera_large.pt")
download_if_missing(
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    SAM2_CHECKPOINT
)

# -------------------------------
# TextDrivenSegmenter (core logic preserved)
# -------------------------------
class TextDrivenSegmenter:
    def __init__(self, device):
        if build_sam2 is None:
            raise RuntimeError("SAM2 components are not available (sam2 import failed).")

        self.device = device
        self._verify_checkpoints()

        # Initialize SAM2
        print("🔄 Initializing SAM2...")
        try:
            # clear previous hydra state if any
            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()

            config_dir = os.path.join(SAM2_REPO, "sam2", "configs")
            if not os.path.isdir(config_dir):
                raise FileNotFoundError(f"SAM2 config directory not found: {config_dir}")

            # initialize hydra config dir (keeps same behavior as original)
            initialize_config_dir(config_dir=config_dir)
            config_name = "sam2.1/sam2.1_hiera_l"
            sam2_model = build_sam2(config_name, SAM2_CHECKPOINT, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 initialized successfully!")
        except Exception as e:
            print("❌ SAM2 initialization failed:")
            traceback.print_exc()
            raise RuntimeError(f"SAM2 initialization failed: {e}")

        # Initialize CLIPSeg (same model as original)
        print("🔄 Initializing CLIPSeg...")
        try:
            model_name = "CIDAS/clipseg-rd64-refined"
            self.clip_processor = CLIPSegProcessor.from_pretrained(model_name)
            self.clip_model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
            print("✅ CLIPSeg initialized successfully!")
        except Exception as e:
            print("❌ CLIPSeg initialization failed:")
            traceback.print_exc()
            raise RuntimeError(f"CLIPSeg initialization failed: {e}")

        print("🎉 TextDrivenSegmenter ready!")

    def _verify_checkpoints(self):
        required = [SAM2_CHECKPOINT]
        for p in required:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing checkpoint: {p}")
        print("✅ All checkpoint files verified")

    def _load_image(self, image_input):
        # support url / local path / PIL.Image
        if isinstance(image_input, str) and image_input.startswith("http"):
            r = requests.get(image_input)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        elif isinstance(image_input, str):
            return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError("Invalid image input type; pass a URL, path, or PIL.Image")

    def process_image(self, image_input, text_prompt, clip_threshold=0.5):
        """
        Core pipeline (preserves your original multi-object behavior):
        1) CLIPSeg -> heatmaps per phrase
        2) threshold -> contours -> largest contour -> bounding box seed
        3) SAM2.predict(box) to refine mask
        4) collect masks, boxes, scores, phrases
        """
        img = self._load_image(image_input)
        prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]
        if len(prompts) == 0:
            raise ValueError("No prompt provided")

        # Prepare inputs via CLIPSeg processor (preserve original behavior)
        inputs = self.clip_processor(
            text=prompts,
            images=[img] * len(prompts),
            padding="max_length",
            return_tensors="pt"
        )

        # Move tensors to device (fix that preserves logic)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        heatmaps = torch.sigmoid(outputs.logits)  # [num_prompts, H', W']
        img_np = np.array(img)
        self.sam2_predictor.set_image(img_np)

        all_masks = []
        all_scores = []
        final_boxes = []
        final_phrases = []

        for i, heatmap in enumerate(heatmaps):
            # resize to image resolution
            heatmap_resized = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(img.height, img.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            binary_mask = (heatmap_resized > clip_threshold).cpu().numpy().astype(np.uint8)

            # find contours and get largest region (same as original)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                # no object found for this phrase
                # preserve original behavior (skip)
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            sam_box_prompt = np.array([x, y, x + w, y + h])

            # refine with SAM2 (preserve return structure: masks, scores, _)
            refined_masks, scores, _ = self.sam2_predictor.predict(box=sam_box_prompt, multimask_output=False)

            # store results
            all_masks.append(refined_masks[0])
            # scores may be None or scalar; keep as float if present
            all_scores.append(float(scores[0]) if (scores is not None and len(scores) > 0) else 0.0)
            final_boxes.append(sam_box_prompt)
            final_phrases.append(prompts[i])

        if not all_masks:
            # preserve original behavior -> return None to indicate no detection
            return None

        return {
            "image": img_np,
            "boxes": np.array(final_boxes),
            "masks": np.array(all_masks),
            "phrases": final_phrases,
            "scores": np.array(all_scores)
        }

# -------------------------------
# Initialize segmenter (but via FastAPI lifespan below)
# -------------------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting segmentation service (lifespan)...")
    try:
        ml_models["segmenter"] = TextDrivenSegmenter(device)
        print("✅ Segmentation models loaded.")
    except Exception:
        print("❌ Failed to load segmentation models:")
        traceback.print_exc()
        # Raise so uvicorn shows the error and service doesn't silently run without models
        raise RuntimeError("Segmentation service failed to start.")
    yield
    ml_models.clear()
    print("✅ Service shutdown complete.")

app = FastAPI(lifespan=lifespan)

# -------------------------------
# Visualization helper (same style as original)
# -------------------------------
def apply_masks_and_visualize(image_np, masks):
    overlay = image_np.copy()
    for mask in masks:
        color = np.random.randint(50, 256, 3)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)
    return overlay

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def read_root():
    return {"status": "Text-based Image Segmentation Service is running."}

@app.post("/segment-image/")
async def segment_image(prompt: str = Form(...), image: UploadFile = File(...)):
    if "segmenter" not in ml_models:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Provided file is not an image.")
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = ml_models["segmenter"].process_image(pil_image, prompt)
        if results is None:
            raise HTTPException(status_code=404, detail="No object detected for given prompt.")
        final_image_np = apply_masks_and_visualize(results["image"], results["masks"])
        final_image_pil = Image.fromarray(final_image_np)
        img_byte_arr = io.BytesIO()
        final_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")

# -------------------------------
# optional: utility endpoint to return JSON details (boxes/phrases) instead of image
# -------------------------------
@app.post("/segment-json/")
async def segment_json(prompt: str = Form(...), image: UploadFile = File(...)):
    """
    Returns JSON with boxes, phrases, and scores (and masks as RLE or shapes if needed).
    Useful for programmatic consumers.
    """
    if "segmenter" not in ml_models:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Provided file is not an image.")
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = ml_models["segmenter"].process_image(pil_image, prompt)
        if results is None:
            raise HTTPException(status_code=404, detail="No object detected for given prompt.")
        # convert numpy arrays to lists for JSON
        boxes = results["boxes"].tolist()
        phrases = results["phrases"]
        scores = results["scores"].tolist() if results.get("scores") is not None else []
        # For masks, return simple bounding-box + mask-sum (heavy mask data avoided).
        mask_summaries = [int(mask.sum()) for mask in results["masks"]]
        return {"boxes": boxes, "phrases": phrases, "scores": scores, "mask_pixel_counts": mask_summaries}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")
