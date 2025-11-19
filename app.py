import io
import base64
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)
from transformers.image_utils import load_image

MODEL_ID = "openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MM-Grounding-DINO model {MODEL_ID} on {DEVICE}...")

# IMPORTANT: trust_remote_code=True so Transformers can load the custom mm-grounding-dino classes
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
).to(DEVICE)
model.eval()

app = FastAPI(title="MM-Grounding-DINO Region Proposals")


class DetectRequest(BaseModel):
    image_b64: str           # base64-encoded PNG/JPEG of one page
    prompts: List[str]       # e.g. ["goods description table", "HS code column"]
    box_threshold: float = 0.3


class Detection(BaseModel):
    label: str
    score: float
    box_xyxy: List[float]    # [x1, y1, x2, y2] in pixels
    image_width: int
    image_height: int


class DetectResponse(BaseModel):
    detections: List[Detection]


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    # Decode image
    image_bytes = base64.b64decode(req.image_b64)
    image = load_image(io.BytesIO(image_bytes))

    # MM-Grounding-DINO expects list-of-lists for text
    text_labels = [req.prompts]

    inputs = processor(
        images=image,
        text=text_labels,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    processed = processor.post_process_grounded_object_detection(
        outputs,
        threshold=req.box_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    detections = []
    for box, score, label in zip(
        processed["boxes"], processed["scores"], processed["labels"]
    ):
        detections.append(
            Detection(
                label=label,
                score=float(score),
                box_xyxy=[float(x) for x in box],
                image_width=image.width,
                image_height=image.height,
            )
        )

    return DetectResponse(detections=detections)
