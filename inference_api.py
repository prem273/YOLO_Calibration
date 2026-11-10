r"""Local FastAPI backend for raw-confidence YOLOv8n and YOLO11n comparison.

Run from the repository root with:
    .\.venv\Scripts\python.exe -m uvicorn inference_api:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MODEL_WEIGHTS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
}

app = FastAPI(
    title="YOLO Raw Confidence Comparison API",
    description="Runs YOLOv8n and YOLO11n on an uploaded image without confidence calibration.",
    version="1.0.0",
)

_models: dict[str, YOLO] = {}
_model_lock = threading.Lock()


def get_model(model_key: str) -> YOLO:
    """Load a checkpoint once. Ultralytics downloads official weights if missing."""
    if model_key not in MODEL_WEIGHTS:
        raise ValueError(f"Unsupported model key: {model_key}")
    if model_key not in _models:
        with _model_lock:
            if model_key not in _models:
                _models[model_key] = YOLO(MODEL_WEIGHTS[model_key])
    return _models[model_key]


def decode_image(data: bytes) -> np.ndarray:
    """Validate a supported image and return an OpenCV BGR array."""
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Please upload an image under 12 MB.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format. Use JPG, PNG, or WEBP.")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions are too large to process safely.")
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="That file is not a valid supported image.") from exc

    array = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="The image could not be decoded.")
    return decoded


def run_one_model(model_key: str, image_bgr: np.ndarray, confidence_threshold: float) -> dict[str, Any]:
    """Infer and return detections, annotated image, and measured prediction time."""
    try:
        model = get_model(model_key)
        start = time.perf_counter()
        result = model.predict(source=image_bgr, conf=confidence_threshold, imgsz=640, verbose=False)[0]
        inference_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:
        # Keep the API message useful but avoid dumping a full traceback into the browser.
        raise HTTPException(
            status_code=500,
            detail=(
                f"{MODEL_WEIGHTS[model_key]} failed to load or run ({type(exc).__name__}). "
                "Check the VS Code terminal for details and confirm that the first-run weight download completed."
            ),
        ) from exc

    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy_values = boxes.xyxy.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_names = result.names
        for coordinates, class_value, confidence_value in zip(xyxy_values, class_values, confidence_values):
            class_id = int(class_value)
            confidence = float(confidence_value)
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(class_names.get(class_id, class_id)),
                    "confidence": round(confidence, 6),
                    "confidence_percent": round(confidence * 100.0, 2),
                    "box_xyxy": [round(float(coord), 2) for coord in coordinates],
                }
            )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    annotated_bgr = result.plot()
    ok, encoded = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the annotated result image.")

    return {
        "model": model_key,
        "weights": MODEL_WEIGHTS[model_key],
        "inference_ms": round(inference_ms, 2),
        "detection_count": len(detections),
        "annotated_image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
        "detections": detections,
        "confidence_note": "Raw, uncalibrated detector confidence. It is not a guaranteed probability of correctness.",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Return backend status without forcing model downloads."""
    return {
        "status": "ok",
        "models": {
            key: {"weights": weight, "loaded": key in _models}
            for key, weight in MODEL_WEIGHTS.items()
        },
    }


@app.post("/api/predict")
async def predict_both(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
) -> dict[str, Any]:
    """Run both nano detectors on the same image using raw confidence scores."""
    if not 0.05 <= confidence <= 0.90:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.05 and 0.90.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    image_bgr = decode_image(data)
    results = [
        run_one_model("yolov8n", image_bgr, confidence),
        run_one_model("yolo11n", image_bgr, confidence),
    ]
    return {
        "filename": Path(file.filename or "uploaded-image").name,
        "image_width": int(image_bgr.shape[1]),
        "image_height": int(image_bgr.shape[0]),
        "calibration": "none",
        "method_label": "Raw confidence (uncalibrated baseline)",
        "results": results,
    }


@app.get("/live-detection")
def live_detection_page() -> FileResponse:
    return FileResponse(WEB_DIR / "live-detection.html")


# Serve the pre-existing dashboard and its assets at the end, after API/page routes.
# The dashboard's root index.html, app.js, styles.css, and data.js remain unchanged.
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="project-site")
