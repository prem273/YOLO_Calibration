"""FastAPI backend for raw and post-hoc calibrated YOLOv8n / YOLO11n inference."""
from __future__ import annotations
import json

import base64
import io
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
CALIBRATION_DIR = ROOT / "calibration_artifacts"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}
MODEL_WEIGHTS = {"yolov8n": "yolov8n.pt", "yolo11n": "yolo11n.pt"}
METHOD_LABELS = {"none": "None (Raw Confidence)", "platt": "Platt Scaling", "isotonic": "Isotonic Regression", "loc_isotonic": "Localization-Aware Isotonic"}
ANALYSIS_MODE_LABELS = {
    "standard": "Standard Detection / Score Calibration",
    "crc": "Conformal Risk Control (CRC-style)",
    "mc_dropout": "Monte Carlo Dropout (experimental)",
}
CRC_POLICY_PATH = CALIBRATION_DIR / "crc_risk_thresholds.json"
MC_DROPOUT_P = 0.10
MC_DEFAULT_PASSES = 6
MC_INFERENCE_CONF = 0.05
MC_MATCH_IOU = 0.50
_mc_lock = threading.Lock()

app = FastAPI(
    title="YOLO Calibration Comparison API",
    description="Compare raw and post-hoc calibrated confidence for YOLOv8n and YOLO11n.",
    version="1.1.0",
)
_models: dict[str, YOLO] = {}
_calibrators: dict[tuple[str, str], dict[str, Any]] = {}
_model_lock = threading.Lock()
_calibrator_lock = threading.Lock()


def get_model(model_key: str) -> YOLO:
    if model_key not in MODEL_WEIGHTS:
        raise ValueError(f"Unsupported model key: {model_key}")
    if model_key not in _models:
        with _model_lock:
            if model_key not in _models:
                _models[model_key] = YOLO(MODEL_WEIGHTS[model_key])
    return _models[model_key]


def get_calibrator(model_key: str, method: str) -> dict[str, Any]:
    key = (model_key, method)
    if key in _calibrators:
        return _calibrators[key]
    path = CALIBRATION_DIR / f"{model_key}_{method}.joblib"
    if not path.is_file():
        raise HTTPException(
            status_code=409,
            detail=f"{METHOD_LABELS.get(method, method)} is not fitted for {model_key}. For localization-aware isotonic run: python fit_localization_isotonic.py --data-dir data/coco --max-images 500 (pilot), or --max-images 5000 for the full split.",
        )
    with _calibrator_lock:
        if key not in _calibrators:
            try:
                artifact = joblib.load(path)
                if artifact.get("model_key") != model_key or artifact.get("method") != method or "estimator" not in artifact:
                    raise ValueError("Calibration artifact metadata did not match the requested model/method.")
                _calibrators[key] = artifact
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Could not load calibration artifact {path.name}: {type(exc).__name__}.") from exc
    return _calibrators[key]


def decode_image(data: bytes) -> np.ndarray:
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
    image_bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="The image could not be decoded.")
    return image_bgr


def calibrate_score(model_key: str, method: str, raw_score: float) -> float:
    if method == "none":
        return raw_score
    artifact = get_calibrator(model_key, method)
    estimator = artifact["estimator"]
    if method == "platt":
        score = float(estimator.predict_proba(np.asarray([[raw_score]], dtype=float))[0, 1])
    elif method in ("isotonic", "loc_isotonic"):
        score = float(estimator.predict(np.asarray([raw_score], dtype=float))[0])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown calibration method: {method}")
    return float(np.clip(score, 0.0, 1.0))


def _box_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
    return (
        boxes.xyxy.detach().cpu().numpy().astype(float),
        boxes.conf.detach().cpu().numpy().astype(float),
        boxes.cls.detach().cpu().numpy().astype(int),
    )


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _annotate(image_bgr: np.ndarray, detections: list[dict[str, Any]], mode: str) -> str:
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    thickness = max(1, round(min(height, width) / 450))
    font_scale = max(0.45, min(height, width) / 1200)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box_xyxy"]]
        x1 = max(0, min(width - 1, x1)); x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1)); y2 = max(0, min(height - 1, y2))
        name = det["class_name"]
        raw = float(det["raw_confidence"])
        if mode == "mc_dropout":
            label = f"{name} raw {raw:.2f} persist {det['mc_presence_frequency']:.0%}"
        elif mode == "crc":
            label = f"{name} {raw:.2f} CRC"
        elif det.get("localization_quality_score") is not None:
            label = f"{name} locQ {float(det['localization_quality_score']):.2f} (raw {raw:.2f})"
        elif det.get("calibrated_confidence") is not None:
            label = f"{name} {float(det['calibrated_confidence']):.2f} (raw {raw:.2f})"
        else:
            label = f"{name} {raw:.2f}"
        color = (92, 215, 153)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        label_y = max(text_h + 4, y1)
        cv2.rectangle(annotated, (x1, label_y - text_h - 6), (min(width - 1, x1 + text_w + 6), label_y + 2), color, -1)
        cv2.putText(annotated, label, (x1 + 3, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (8, 18, 30), thickness, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the annotated result image.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


class _MCDropoutPreHook:
    """Experimental post-hoc dropout hook at Detect-head feature inputs."""
    def __init__(self, probability: float):
        self.probability = probability
        self.calls = 0

    def __call__(self, module: Any, args: tuple[Any, ...]) -> Any:
        if not args:
            return None
        features = args[0]
        if not isinstance(features, (list, tuple)):
            return None
        self.calls += 1
        dropped = [F.dropout(feature, p=self.probability, training=True) for feature in features]
        return (type(features)(dropped) if isinstance(features, tuple) else dropped, *args[1:])


def _find_detect_head(model: YOLO) -> Any:
    predictor_model = getattr(getattr(model, "predictor", None), "model", None)
    if predictor_model is None:
        raise HTTPException(status_code=409, detail="YOLO inference predictor is not initialized; retry the request.")
    heads = [module for module in predictor_model.modules() if module.__class__.__name__ == "Detect"]
    if not heads:
        raise HTTPException(status_code=409, detail="This checkpoint has no compatible Detect head for the experimental dropout probe.")
    return heads[-1]


def _run_mc_dropout(model_key: str, image_bgr: np.ndarray, confidence_threshold: float, mc_passes: int) -> dict[str, Any]:
    if not 2 <= mc_passes <= 12:
        raise HTTPException(status_code=400, detail="MC Dropout passes must be between 2 and 12.")
    try:
        model = get_model(model_key)
        with _mc_lock:
            start = time.perf_counter()
            base_result = model.predict(source=image_bgr, conf=0.001, imgsz=640, iou=0.7, max_det=300, verbose=False)[0]
            base_ms = (time.perf_counter() - start) * 1000.0
            head = _find_detect_head(model)
            hook = _MCDropoutPreHook(MC_DROPOUT_P)
            handle = head.register_forward_pre_hook(hook)
            stochastic_results = []
            stochastic_ms = 0.0
            try:
                for _ in range(mc_passes):
                    pass_start = time.perf_counter()
                    result = model.predict(source=image_bgr, conf=MC_INFERENCE_CONF, imgsz=640, iou=0.7, max_det=300, verbose=False)[0]
                    stochastic_ms += (time.perf_counter() - pass_start) * 1000.0
                    stochastic_results.append(result)
            finally:
                handle.remove()
        if hook.calls == 0:
            raise HTTPException(status_code=500, detail="MC Dropout hook did not run; this model's inference graph is incompatible.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MC Dropout failed for {MODEL_WEIGHTS[model_key]} ({type(exc).__name__}). Check the Ubuntu terminal for details.") from exc

    base_boxes, base_scores, base_classes = _box_arrays(base_result)
    per_pass = [_box_arrays(result) for result in stochastic_results]
    detections: list[dict[str, Any]] = []
    for box, raw_score, class_id in zip(base_boxes, base_scores, base_classes):
        raw = float(np.clip(raw_score, 0.0, 1.0))
        if raw < confidence_threshold:
            continue
        pass_conf = np.zeros(mc_passes, dtype=float)
        pass_iou = np.zeros(mc_passes, dtype=float)
        for pass_idx, (pb, pc, pk) in enumerate(per_pass):
            same_class = np.where(pk == int(class_id))[0]
            if len(same_class) == 0:
                continue
            overlaps = [(_iou_xyxy(box, pb[j]), int(j)) for j in same_class]
            best_iou, best_idx = max(overlaps, key=lambda pair: pair[0])
            if best_iou >= MC_MATCH_IOU:
                pass_conf[pass_idx] = float(np.clip(pc[best_idx], 0.0, 1.0))
                pass_iou[pass_idx] = float(best_iou)
        freq = float(np.mean(pass_conf > 0.0))
        mc_mean = float(np.mean(pass_conf))
        mc_std = float(np.std(pass_conf))
        mc_iou = float(np.mean(pass_iou))
        detections.append({
            "class_id": int(class_id), "class_name": str(base_result.names.get(int(class_id), class_id)),
            "raw_confidence": round(raw, 6), "raw_confidence_percent": round(raw * 100.0, 2),
            "calibrated_confidence": None, "calibrated_confidence_percent": None,
            "localization_quality_score": None, "localization_quality_percent": None,
            "display_score": round(raw, 6), "display_score_percent": round(raw * 100.0, 2),
            "display_confidence": round(raw, 6), "display_confidence_percent": round(raw * 100.0, 2),
            "confidence": round(raw, 6), "confidence_percent": round(raw * 100.0, 2),
            "box_xyxy": [round(float(v), 2) for v in box],
            "mc_presence_frequency": round(freq, 6), "mc_presence_frequency_percent": round(freq * 100.0, 2),
            "mc_mean_confidence": round(mc_mean, 6), "mc_confidence_std": round(mc_std, 6),
            "mc_mean_matched_iou": round(mc_iou, 6),
        })
    detections.sort(key=lambda item: item["raw_confidence"], reverse=True)
    note = (f"Experimental post-hoc stochastic dropout probe: p={MC_DROPOUT_P:.2f}, T={mc_passes}. "
            "Dropout is injected into Detect-head feature inputs of pretrained models that were not trained with dropout. "
            "Presence frequency and score variability are exploratory uncertainty signals, not calibrated probabilities or a validated Bayesian posterior.")
    return {
        "model": model_key, "weights": MODEL_WEIGHTS[model_key], "calibration_method": "none", "analysis_mode": "mc_dropout",
        "score_semantics": "raw_confidence_plus_experimental_mc_statistics",
        "inference_ms": round(base_ms + stochastic_ms, 2), "base_inference_ms": round(base_ms, 2),
        "mc_inference_ms": round(stochastic_ms, 2), "mc_passes": mc_passes, "mc_dropout_probability": MC_DROPOUT_P,
        "detection_count": len(detections), "annotated_image": _annotate(image_bgr, detections, "mc_dropout"),
        "detections": detections, "confidence_note": note,
    }


def run_one_model(model_key: str, image_bgr: np.ndarray, confidence_threshold: float, method: str,
                  analysis_mode: str = "standard", crc_config: dict[str, Any] | None = None,
                  risk_target: float | None = None, mc_passes: int = MC_DEFAULT_PASSES) -> dict[str, Any]:
    if analysis_mode == "mc_dropout":
        return _run_mc_dropout(model_key, image_bgr, confidence_threshold, mc_passes)
    try:
        model = get_model(model_key)
        start = time.perf_counter()
        result = model.predict(source=image_bgr, conf=0.001, imgsz=640, max_det=300, verbose=False)[0]
        inference_ms = (time.perf_counter() - start) * 1000.0
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{MODEL_WEIGHTS[model_key]} failed to load or run ({type(exc).__name__}). Check the Ubuntu terminal and model-weight download.") from exc

    is_crc = analysis_mode == "crc"
    score_threshold = float(crc_config["threshold"]) if is_crc and crc_config else confidence_threshold
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy_values = boxes.xyxy.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_names = result.names
        for coordinates, class_value, raw_value in zip(xyxy_values, class_values, confidence_values):
            class_id = int(class_value)
            raw = float(np.clip(raw_value, 0.0, 1.0))
            display_score = raw if is_crc else calibrate_score(model_key, method, raw)
            if display_score < score_threshold:
                continue
            class_name = str(class_names.get(class_id, class_id))
            box = [round(float(v), 2) for v in coordinates]
            detections.append({
                "class_id": class_id, "class_name": class_name,
                "raw_confidence": round(raw, 6), "raw_confidence_percent": round(raw * 100.0, 2),
                "calibrated_confidence": round(display_score, 6) if (not is_crc and method not in ("none", "loc_isotonic")) else None,
                "calibrated_confidence_percent": round(display_score * 100.0, 2) if (not is_crc and method not in ("none", "loc_isotonic")) else None,
                "localization_quality_score": round(display_score, 6) if method == "loc_isotonic" and not is_crc else None,
                "localization_quality_percent": round(display_score * 100.0, 2) if method == "loc_isotonic" and not is_crc else None,
                "display_score": round(display_score, 6), "display_score_percent": round(display_score * 100.0, 2),
                "display_confidence": round(display_score, 6), "display_confidence_percent": round(display_score * 100.0, 2),
                "confidence": round(display_score, 6), "confidence_percent": round(display_score * 100.0, 2),
                "box_xyxy": box,
            })
    detections.sort(key=lambda item: item["display_score"], reverse=True)
    if is_crc:
        policy = {
            "risk_target": float(risk_target), "threshold": float(crc_config["threshold"]),
            "empirical_calibration_risk": crc_config.get("empirical_risk"),
            "risk_upper_bound": crc_config.get("risk_upper_bound"),
            "calibration_images": crc_config.get("calibration_images"),
            "delta": crc_config.get("delta"), "calibration_coverage": crc_config.get("retention_fraction"),
            "risk_definition": "Per-image false-positive occurrence: fraction of calibration images with at least one incorrect retained detection.",
        }
        note = (f"CRC-style risk threshold {score_threshold:.2f} selected for target per-image false-positive occurrence risk {float(risk_target):.0%}. "
                f"Calibration upper bound: {float(crc_config.get('risk_upper_bound', 0)):.3f}. "
                "This finite-sample bound assumes exchangeable calibration and future images; it is not a guarantee under arbitrary distribution shift. Pilot policy is for development only.")
    elif method == "none":
        note = "Raw detector score; not guaranteed to equal the probability that the detection is correct."
        policy = None
    elif method == "loc_isotonic":
        note = "Localization-aware score fitted to correctness × IoU (IoU for matched same-class detections at IoU >= 0.50; otherwise 0). It is a quality target, not a plain probability-of-correctness score. Pilot artifacts may not transfer to other image distributions."
        policy = None
    else:
        note = "Post-hoc calibrated estimate of detection correctness under same-class IoU >= 0.50 matching on COCO val2017. Performance may differ on other image distributions."
        policy = None
    return {
        "model": model_key, "weights": MODEL_WEIGHTS[model_key], "calibration_method": "none" if is_crc else method,
        "analysis_mode": analysis_mode, "score_semantics": "raw_confidence_crc_filtered" if is_crc else ("localization_quality" if method == "loc_isotonic" else ("raw_confidence" if method == "none" else "estimated_detection_correctness")),
        "inference_ms": round(inference_ms, 2), "detection_count": len(detections),
        "annotated_image": _annotate(image_bgr, detections, "crc" if is_crc else "standard"),
        "detections": detections, "confidence_note": note, "risk_control": policy,
    }

def _load_crc_policy() -> dict[str, Any] | None:
    if not CRC_POLICY_PATH.is_file():
        return None
    try:
        payload = json.loads(CRC_POLICY_PATH.read_text(encoding="utf-8"))
        if payload.get("models") and payload.get("risk_targets"):
            return payload
    except Exception:
        return None
    return None


@app.get("/api/health")
def health() -> dict[str, Any]:
    policy = _load_crc_policy()
    target_availability: dict[str, bool] = {}
    if policy:
        for target in policy.get("risk_targets", []):
            t_key = f"{float(target):.2f}"
            target_availability[t_key] = all(
                bool(policy.get("models", {}).get(model_key, {}).get("thresholds", {}).get(t_key, {}).get("certified", False))
                for model_key in MODEL_WEIGHTS
            )
    return {
        "status": "ok",
        "models": {key: {"weights": weight, "loaded": key in _models} for key, weight in MODEL_WEIGHTS.items()},
        "calibrators": {
            method: {"available": all((CALIBRATION_DIR / f"{model_key}_{method}.joblib").is_file() for model_key in MODEL_WEIGHTS)}
            for method in ("platt", "isotonic", "loc_isotonic")
        },
        "analysis_modes": {
            "standard": {"available": True},
            "crc": {"available": bool(policy) and any(target_availability.values()), "targets": target_availability,
                    "pilot": bool(policy.get("pilot", True)) if policy else True},
            "mc_dropout": {"available": True, "experimental": True, "default_passes": MC_DEFAULT_PASSES,
                           "dropout_probability": MC_DROPOUT_P,
                           "note": "Post-hoc dropout injected at Detect-head inputs; pretrained models were not trained with dropout."},
        },
    }


@app.post("/api/predict")
async def predict_both(
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    calibration: str = Form("none"),
    analysis_mode: str = Form("standard"),
    risk_target: float = Form(0.25),
    mc_passes: int = Form(MC_DEFAULT_PASSES),
) -> dict[str, Any]:
    if analysis_mode not in ANALYSIS_MODE_LABELS:
        raise HTTPException(status_code=400, detail="Choose Standard Detection, CRC, or MC Dropout.")
    if analysis_mode != "crc" and not 0.05 <= confidence <= 0.90:
        raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.05 and 0.90.")
    if analysis_mode == "standard":
        if calibration not in METHOD_LABELS:
            raise HTTPException(status_code=400, detail="Choose a supported score-calibration method.")
        if calibration != "none":
            for model_key in MODEL_WEIGHTS:
                get_calibrator(model_key, calibration)
    if analysis_mode == "crc":
        policy = _load_crc_policy()
        if not policy:
            raise HTTPException(status_code=409, detail="CRC policy not fitted yet. Run: python fit_crc.py --data-dir data/coco --max-images 500 --device cpu")
        target_key = f"{risk_target:.2f}"
        if target_key not in {f"{float(v):.2f}" for v in policy.get("risk_targets", [])}:
            raise HTTPException(status_code=400, detail="Choose a risk target listed by /api/health.")
        for model_key in MODEL_WEIGHTS:
            cfg = policy.get("models", {}).get(model_key, {}).get("thresholds", {}).get(target_key)
            if not cfg or not cfg.get("certified", False):
                raise HTTPException(status_code=409, detail=f"No CRC threshold met target risk {risk_target:.0%} for {model_key} on this pilot. Try a higher target risk or fit with more calibration images.")
    if analysis_mode == "mc_dropout" and not 2 <= mc_passes <= 12:
        raise HTTPException(status_code=400, detail="MC Dropout passes must be between 2 and 12.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    image_bgr = decode_image(data)
    results = []
    for model_key in MODEL_WEIGHTS:
        crc_config = None
        if analysis_mode == "crc":
            crc_config = policy["models"][model_key]["thresholds"][f"{risk_target:.2f}"]
        results.append(run_one_model(
            model_key, image_bgr, confidence, calibration if analysis_mode == "standard" else "none",
            analysis_mode=analysis_mode, crc_config=crc_config, risk_target=risk_target, mc_passes=mc_passes,
        ))
    response: dict[str, Any] = {
        "filename": Path(file.filename or "uploaded-image").name,
        "image_width": int(image_bgr.shape[1]), "image_height": int(image_bgr.shape[0]),
        "analysis_mode": analysis_mode, "analysis_label": ANALYSIS_MODE_LABELS[analysis_mode], "results": results,
    }
    if analysis_mode == "standard":
        response["calibration"] = calibration
        response["method_label"] = METHOD_LABELS[calibration]
    elif analysis_mode == "crc":
        response["risk_target"] = risk_target
        response["analysis_note"] = "CRC-style per-image false-positive occurrence control. The target is a risk target, not a calibrated confidence probability. Results depend on the calibration data distribution."
    else:
        response["mc_passes"] = mc_passes
        response["analysis_note"] = "Experimental MC Dropout probe. Dropout is injected post-hoc into a pretrained Detect head not trained with dropout; use the statistics as exploratory uncertainty indicators only."
    return response

@app.get("/live-detection")
def live_detection_page() -> FileResponse:
    return FileResponse(WEB_DIR / "live-detection.html")


app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="project-site")
