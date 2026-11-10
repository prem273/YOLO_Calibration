#!/usr/bin/env python3
"""Fit real post-hoc confidence calibrators for YOLOv8n and YOLO11n on COCO val2017.

Each prediction is labelled correct if it matches an unmatched ground-truth object of the
same COCO class at IoU >= 0.50. Matching is greedy in descending raw-confidence order.
This is a transparent, practical calibration protocol; it is not the full COCO AP evaluator.

Default split for 5,000 images: 2,000 fit / 1,000 validation / 2,000 test (seed 42).
For a faster pilot use --max-images 500, producing 200 / 100 / 200 images. Pilot metrics
are demonstrative only and should not be reported as the full research benchmark.

Outputs to calibration_artifacts/ (do not commit the COCO images or model weights).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from ultralytics import YOLO

MODEL_SPECS = {
    "yolov8n": "yolov8n.pt",
    "yolo11n": "yolo11n.pt",
}
IOU_THRESHOLD = 0.50
PREDICTION_CONFIDENCE_FLOOR = 0.001


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_coco(data_dir: Path) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    ann_path = data_dir / "annotations" / "instances_val2017.json"
    image_dir = data_dir / "val2017"
    if not ann_path.exists():
        candidates = list(data_dir.rglob("instances_val2017.json"))
        if candidates:
            ann_path = candidates[0]
    if not image_dir.exists():
        candidates = [p for p in data_dir.rglob("val2017") if p.is_dir()]
        candidates = [p for p in candidates if any(p.glob("*.jpg"))]
        if candidates:
            image_dir = candidates[0]
    if not ann_path.exists() or not image_dir.exists():
        raise FileNotFoundError(
            f"Expected COCO annotations and images under {data_dir}. "
            "Expected annotations/instances_val2017.json and val2017/*.jpg."
        )
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    category_names = {int(c["id"]): str(c["name"]).strip().lower() for c in payload["categories"]}
    anns_by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in payload["annotations"]:
        # Crowd regions are excluded from this simplified correctness definition.
        if int(ann.get("iscrowd", 0)):
            continue
        x, y, w, h = map(float, ann["bbox"])
        anns_by_id[int(ann["image_id"])].append({
            "class_name": category_names.get(int(ann["category_id"]), ""),
            "box_xyxy": [x, y, x + w, y + h],
        })
    available = []
    for rec in payload["images"]:
        path = image_dir / rec["file_name"]
        if path.is_file():
            available.append({"id": int(rec["id"]), "path": path, "file_name": rec["file_name"]})
    if len(available) < 20:
        raise RuntimeError(f"Only {len(available)} annotated images found in {image_dir}.")
    return sorted(available, key=lambda x: x["file_name"]), anns_by_id, category_names


def label_predictions(result: Any, gts: list[dict[str, Any]], model_names: dict[int, str]) -> tuple[list[float], list[int]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    cls_ids = boxes.cls.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    preds = []
    for box, cls_id, score in zip(xyxy, cls_ids, scores):
        preds.append({
            "box_xyxy": [float(x) for x in box],
            "class_name": str(model_names.get(int(cls_id), str(int(cls_id)))).strip().lower(),
            "score": float(score),
        })
    preds.sort(key=lambda p: p["score"], reverse=True)
    used_gt: set[int] = set()
    out_scores: list[float] = []
    out_labels: list[int] = []
    for pred in preds:
        best_iou = 0.0
        best_index = -1
        for idx, gt in enumerate(gts):
            if idx in used_gt or gt["class_name"] != pred["class_name"]:
                continue
            overlap = iou_xyxy(pred["box_xyxy"], gt["box_xyxy"])
            if overlap > best_iou:
                best_iou, best_index = overlap, idx
        correct = best_index >= 0 and best_iou >= IOU_THRESHOLD
        if correct:
            used_gt.add(best_index)
        out_scores.append(float(np.clip(pred["score"], 0.0, 1.0)))
        out_labels.append(1 if correct else 0)
    return out_scores, out_labels


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    if len(scores) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(scores)
    error = 0.0
    for i in range(bins):
        mask = (scores >= edges[i]) & ((scores <= edges[i + 1]) if i == bins - 1 else (scores < edges[i + 1]))
        if mask.any():
            error += mask.sum() / total * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return float(error)


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    if len(scores) == 0:
        return {"detections": 0, "ece_15_bins": float("nan"), "brier": float("nan"), "mean_score": float("nan"), "empirical_accuracy": float("nan")}
    return {
        "detections": int(len(scores)),
        "ece_15_bins": round(ece(scores, labels), 6),
        "brier": round(float(np.mean((scores - labels) ** 2)), 6),
        "mean_score": round(float(scores.mean()), 6),
        "empirical_accuracy": round(float(labels.mean()), 6),
    }


def apply_estimator(method: str, estimator: Any, scores: np.ndarray) -> np.ndarray:
    if method == "platt":
        return np.asarray(estimator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=float)
    return np.asarray(estimator.predict(scores), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt and isotonic confidence calibration for two YOLO nano detectors.")
    parser.add_argument("--data-dir", default="data/coco", help="COCO val2017 root (default: data/coco)")
    parser.add_argument("--max-images", type=int, default=500, help="Images to use; start with 500 pilot, use 5000 for full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--output-dir", default="calibration_artifacts")
    args = parser.parse_args()
    if args.max_images < 100:
        raise SystemExit("Use at least 100 images for a pilot; recommended 500 or 5000.")

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_images, anns_by_id, _ = load_coco(data_dir)
    rng = random.Random(args.seed)
    chosen = rng.sample(all_images, min(args.max_images, len(all_images)))
    rng.shuffle(chosen)
    n = len(chosen)
    n_fit = int(n * 0.40)
    n_val = int(n * 0.20)
    splits: dict[str, list[dict[str, Any]]] = {
        "fit": chosen[:n_fit],
        "validation": chosen[n_fit:n_fit + n_val],
        "test": chosen[n_fit + n_val:],
    }
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit("The chosen number of images creates an empty split.")

    print(f"COCO images available: {len(all_images)}")
    print(f"Selected {n} images; split: fit={len(splits['fit'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    print("Matching: same class and IoU >= 0.50; each ground-truth box can match at most one prediction.")
    if n < 5000:
        print("PILOT MODE: use these artifacts for interface testing, not as full-benchmark research results.")

    report: dict[str, Any] = {
        "protocol": "COCO val2017 detection calibration; greedy class-aware IoU matching",
        "iou_threshold": IOU_THRESHOLD,
        "confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "seed": args.seed,
        "selected_images": n,
        "split_images": {k: len(v) for k, v in splits.items()},
        "pilot": n < 5000,
        "models": {},
    }

    for model_key, weights in MODEL_SPECS.items():
        print(f"\nLoading {weights} ...", flush=True)
        model = YOLO(weights)
        names = {int(k): str(v) for k, v in model.names.items()} if isinstance(model.names, dict) else {i: str(v) for i, v in enumerate(model.names)}
        split_scores: dict[str, list[float]] = {k: [] for k in splits}
        split_labels: dict[str, list[int]] = {k: [] for k in splits}
        total_start = time.perf_counter()
        for split_name, records in splits.items():
            print(f"[{model_key}] inferencing {split_name}: {len(records)} images", flush=True)
            # Batch source paths to avoid repeatedly initializing the prediction pipeline.
            for start_idx in range(0, len(records), max(1, args.batch_size)):
                batch = records[start_idx:start_idx + max(1, args.batch_size)]
                results = model.predict(
                    source=[str(rec["path"]) for rec in batch],
                    conf=PREDICTION_CONFIDENCE_FLOOR,
                    imgsz=640,
                    device=args.device,
                    batch=max(1, args.batch_size),
                    max_det=300,
                    verbose=False,
                    stream=False,
                )
                for rec, result in zip(batch, results):
                    sc, lab = label_predictions(result, anns_by_id.get(rec["id"], []), names)
                    split_scores[split_name].extend(sc)
                    split_labels[split_name].extend(lab)
                done = min(start_idx + len(batch), len(records))
                if done == len(records) or done % 100 == 0:
                    print(f"  {done}/{len(records)} images", flush=True)
        elapsed = time.perf_counter() - total_start
        fit_scores = np.asarray(split_scores["fit"], dtype=np.float64)
        fit_labels = np.asarray(split_labels["fit"], dtype=np.int64)
        if len(fit_scores) < 50 or len(np.unique(fit_labels)) < 2:
            raise RuntimeError(f"Not enough varied fit detections for {model_key}: {len(fit_scores)} detections, labels={np.unique(fit_labels)}")

        platt = LogisticRegression(solver="lbfgs", max_iter=2000)
        platt.fit(fit_scores.reshape(-1, 1), fit_labels)
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        isotonic.fit(fit_scores, fit_labels)
        metadata = {
            "model": model_key,
            "weights": weights,
            "fit_images": len(splits["fit"]),
            "fit_detections": int(len(fit_scores)),
            "iou_threshold": IOU_THRESHOLD,
            "seed": args.seed,
            "pilot": n < 5000,
            "label_definition": "1 if same-class prediction greedily matches an unmatched COCO ground-truth box at IoU >= 0.50, otherwise 0",
        }
        for method, estimator in (("platt", platt), ("isotonic", isotonic)):
            artifact_path = out_dir / f"{model_key}_{method}.joblib"
            joblib.dump({"model_key": model_key, "method": method, "estimator": estimator, "metadata": metadata}, artifact_path)
            print(f"Saved {artifact_path}")

        model_report: dict[str, Any] = {"metadata": metadata, "elapsed_seconds": round(elapsed, 2), "splits": {}}
        for split_name in splits:
            score_arr = np.asarray(split_scores[split_name], dtype=np.float64)
            label_arr = np.asarray(split_labels[split_name], dtype=np.int64)
            part: dict[str, Any] = {"raw": metrics(score_arr, label_arr)}
            if len(score_arr):
                part["platt"] = metrics(apply_estimator("platt", platt, score_arr), label_arr)
                part["isotonic"] = metrics(apply_estimator("isotonic", isotonic, score_arr), label_arr)
            else:
                part["platt"] = metrics(score_arr, label_arr)
                part["isotonic"] = metrics(score_arr, label_arr)
            model_report["splits"][split_name] = part
            print(f"[{model_key} / {split_name}] {json.dumps(part, sort_keys=True)}", flush=True)
        report["models"][model_key] = model_report

    report_path = out_dir / "calibration_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nCalibration fitting complete. Metrics report: {report_path}")
    print("The frontend can now offer Platt and Isotonic options for both models.")
    if report["pilot"]:
        print("Reminder: pilot artifacts are for testing the UI; run --max-images 5000 before reporting research results.")


if __name__ == "__main__":
    main()
