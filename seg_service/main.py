import os
import sys
import io
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import wget
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import hydra
from hydra import initialize_config_dir
import traceback

# -------------------------------
# Add SAM2 repo to path
# -------------------------------
# Assumes 'segment-anything-2' is a subdirectory in the same folder as this script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "segment-anything-2")))

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("❌ Could not import SAM2. Make sure 'segment-anything-2' directory exists and is accessible.")
    build_sam2 = None
    SAM2ImagePredictor = None

# -------------------------------
# Text-Driven Segmenter Class
# -------------------------------
class TextDrivenSegmenter:
    def __init__(self, device):
        if build_sam2 is None:
            raise RuntimeError("SAM2 components are not available. Check import paths.")

        self.device = device
        self._verify_and_download_checkpoints()

        # --- Initialize SAM2 ---
        sam2_checkpoint = os.path.join(os.path.dirname(__file__), "checkpoints", "sam2.1_hiera_large.pt")
        print("🔄 Initializing SAM2...")
        try:
            if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.core.global_hydra.GlobalHydra.instance().clear()

            config_dir = os.path.join(os.path.dirname(__file__), "segment-anything-2", "sam2", "configs")
            if not os.path.isdir(config_dir):
                raise FileNotFoundError(f"SAM2 config directory not found: {config_dir}")

            initialize_config_dir(config_dir=config_dir, version_base=None)
            config_name = "sam2.1/sam2.1_hiera_l"
            sam2_model = build_sam2(config_name, sam2_checkpoint, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("✅ SAM2 initialized successfully!")
        except Exception as e:
            print(f"❌ SAM2 initialization failed: {e}")
            traceback.print_exc()
            raise RuntimeError(f"SAM2 initialization failed: {e}")

        # --- Initialize CLIPSeg ---
        print("🔄 Initializing CLIPSeg...")
        try:
            model_name = "CIDAS/clipseg-rd64-refined"
            self.clip_processor = CLIPSegProcessor.from_pretrained(model_name)
            self.clip_model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(self.device)
            print("✅ CLIPSeg initialized successfully!")
        except Exception as e:
            print(f"❌ CLIPSeg initialization failed: {e}")
            traceback.print_exc()
            raise RuntimeError(f"CLIPSeg initialization failed: {e}")

        print("🎉 TextDrivenSegmenter ready!")

    def _verify_and_download_checkpoints(self):
        checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        sam2_dest = os.path.join(checkpoints_dir, "sam2.1_hiera_large.pt")
        if not os.path.exists(sam2_dest):
            print(f"Downloading SAM2 checkpoint to {sam2_dest}...")
            wget.download(
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
                sam2_dest
            )
            print("\n✅ Download complete.")
        else:
            print("✅ SAM2 checkpoint already exists.")

    def process_image(self, image_input: Image.Image, text_prompt: str, clip_threshold=0.5):
        img = image_input.convert("RGB")
        prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]
        if not prompts:
            print("⚠️ No valid prompts found in the input text.")
            return None

        inputs = self.clip_processor(
            text=prompts,
            images=[img] * len(prompts),
            padding="max_length",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        heatmaps = torch.sigmoid(outputs.logits)
        img_np = np.array(img)
        self.sam2_predictor.set_image(img_np)

        all_masks, all_scores, final_boxes, final_phrases = [], [], [], []

        for i, heatmap in enumerate(heatmaps):
            heatmap_resized = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(img.height, img.width),
                mode='bilinear',
                align_corners=False
            ).squeeze()

            binary_mask = (heatmap_resized > clip_threshold).cpu().numpy().astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                print(f"⚠️ No object found by CLIPSeg for prompt: '{prompts[i]}'")
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            sam_box_prompt = np.array([x, y, x + w, y + h])

            refined_masks, scores, _ = self.sam2_predictor.predict(
                box=sam_box_prompt,
                multimask_output=False
            )
            all_masks.append(refined_masks[0])
            all_scores.append(scores[0])
            final_boxes.append(sam_box_prompt)
            final_phrases.append(prompts[i])

        if not all_masks:
            print("⚠️ Mask generation failed for all prompts.")
            return None

        print(f"🎯 Generated {len(all_masks)} refined masks for objects: {final_phrases}")
        return {
            "image": img_np,
            "boxes": np.array(final_boxes),
            "masks": np.array(all_masks),
            "phrases": final_phrases,
            "scores": np.array(all_scores)
        }

# -------------------------------
# FastAPI Setup
# -------------------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting segmentation service...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ml_models["segmenter"] = TextDrivenSegmenter(device)
        print(f"✅ Segmentation models loaded on device: {device}")
    except Exception:
        print("❌ Failed to load segmentation models:")
        traceback.print_exc()
        raise RuntimeError("Segmentation service failed to start.")
    yield
    ml_models.clear()
    print("✅ Service shutdown complete.")

app = FastAPI(lifespan=lifespan)

# -------------------------------
# Helper functions for Visualization
# -------------------------------
def draw_boxes(image_np, boxes, phrases):
    """Draws only the bounding boxes and labels on an image."""
    overlay = image_np.copy()
    for box, phrase in zip(boxes, phrases):
        x0, y0, x1, y1 = box
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
        text_size, _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        cv2.rectangle(overlay, (x0, y0 - text_h - 5), (x0 + text_w, y0 - 5), (0, 255, 0), -1)
        cv2.putText(overlay, phrase, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return overlay

def draw_masks_and_boxes(image_np, masks, boxes, phrases):
    """Draws masks, bounding boxes, and labels on an image."""
    overlay = image_np.copy()
    for mask in masks:
        color = np.random.randint(50, 256, 3)
        colored_mask = np.zeros_like(overlay)
        colored_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
    # Draw boxes on top of masks
    overlay = draw_boxes(overlay, boxes, phrases)
    return overlay

def create_comparison_visualization(results):
    """Creates a 3-panel comparison image: Original, Bounding Boxes, and Final Masks."""
    image_np = results["image"]
    masks = results["masks"]
    boxes = results["boxes"]
    phrases = results["phrases"]

    # 1. Create each panel
    panel1 = image_np.copy()
    panel2 = draw_boxes(image_np, boxes, phrases)
    panel3 = draw_masks_and_boxes(image_np, masks, boxes, phrases)

    # 2. Add titles to each panel
    h, w, _ = image_np.shape
    header_height = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    panels = [panel1, panel2, panel3]
    titles = ["1. Original Image", "2. Region Seeds (from CLIPSeg)", "3. Final Refined Masks (by SAM2)"]
    titled_panels = []

    for i, panel in enumerate(panels):
        # Create a new image with space for the header
        new_panel = np.ones((h + header_height, w, 3), dtype=np.uint8) * 255
        new_panel[header_height:h + header_height, :] = panel
        
        # Add the title text
        text_size, _ = cv2.getTextSize(titles[i], font, font_scale, font_thickness)
        text_x = (w - text_size[0]) // 2
        text_y = (header_height + text_size[1]) // 2
        cv2.putText(new_panel, titles[i], (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        titled_panels.append(new_panel)

    # 3. Combine the panels horizontally
    comparison_image = np.hstack(titled_panels)
    return comparison_image

# -------------------------------
# Routes
# -------------------------------
@app.get("/")
def read_root():
    return {"status": "Text-based Image Segmentation Service is running."}

@app.post("/segment-image/")
async def segment_image(prompt: str = Form(...), image: UploadFile = File(...)):
    if "segmenter" not in ml_models:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded or is initializing.")
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Provided file is not an image.")
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        results = ml_models["segmenter"].process_image(pil_image, prompt)

        if results is None:
            raise HTTPException(status_code=404, detail="No object detected for the given prompt.")

        # --- UPDATED: Use the new 3-panel comparison visualization ---
        final_image_np = create_comparison_visualization(results)
        final_image_pil = Image.fromarray(final_image_np)

        img_byte_arr = io.BytesIO()
        final_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
        print(f"❌ An error occurred during segmentation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Segmentation error: {str(e)}")

