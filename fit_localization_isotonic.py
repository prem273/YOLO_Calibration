#!/usr/bin/env python3
"""Fit Localization-Aware Isotonic calibration for YOLOv8n and YOLO11n.

Per prediction, greedily match the highest-confidence predictions to one unused
same-class COCO ground-truth box. The target is IoU if a match has IoU >= 0.50,
otherwise zero. This corresponds to detection correctness multiplied by IoU.
The fitted score therefore estimates a localization-aware quality target; it is
NOT merely a probability-of-correctness score.

Default pilot split for --max-images 500: 200 fit / 100 validation / 200 test.
Use --max-images 5000 for the full split. Pilot output is for UI verification and
must not be presented as the final research benchmark.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from ultralytics import YOLO

from fit_calibrators import (
    MODEL_SPECS,
    IOU_THRESHOLD,
    PREDICTION_CONFIDENCE_FLOOR,
    ece,
    iou_xyxy,
    load_coco,
)


def label_localization_targets(result: Any, gts: list[dict[str, Any]], model_names: dict[int, str]) -> tuple[list[float], list[float]]:
    """Return raw scores and `correct * IoU` targets using greedy class-aware matching."""
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
    out_targets: list[float] = []
    for pred in preds:
        best_iou = 0.0
        best_index = -1
        for idx, gt in enumerate(gts):
            if idx in used_gt or gt["class_name"] != pred["class_name"]:
                continue
            overlap = iou_xyxy(pred["box_xyxy"], gt["box_xyxy"])
            if overlap > best_iou:
                best_iou, best_index = overlap, idx
        matched = best_index >= 0 and best_iou >= IOU_THRESHOLD
        if matched:
            used_gt.add(best_index)
        out_scores.append(float(np.clip(pred["score"], 0.0, 1.0)))
        out_targets.append(float(np.clip(best_iou, 0.0, 1.0)) if matched else 0.0)
    return out_scores, out_targets


def localization_metrics(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    if len(scores) == 0:
        return {"detections": 0, "localization_ece_15_bins": 0.0, "localization_brier": 0.0,
                "mean_predicted_score": 0.0, "mean_localization_target": 0.0}
    return {
        "detections": int(len(scores)),
        "localization_ece_15_bins": round(float(ece(scores, targets, bins=15)), 6),
        "localization_brier": round(float(np.mean((scores - targets) ** 2)), 6),
        "mean_predicted_score": round(float(scores.mean()), 6),
        "mean_localization_target": round(float(targets.mean()), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit localization-aware isotonic mapping for YOLOv8n and YOLO11n.")
    parser.add_argument("--data-dir", default="data/coco", help="COCO val2017 root")
    parser.add_argument("--max-images", type=int, default=500, help="Use 500 for pilot, 5000 for full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0")
    parser.add_argument("--batch-size", type=int, default=8)
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
    splits = {
        "fit": chosen[:n_fit],
        "validation": chosen[n_fit:n_fit + n_val],
        "test": chosen[n_fit + n_val:],
    }
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit("The chosen number of images creates an empty split.")

    print(f"COCO images available: {len(all_images)}")
    print(f"Selected {n} images; split: fit={len(splits['fit'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    print("Target: matched detection IoU (only if same-class IoU >= 0.50); unmatched/duplicate/incorrect predictions target 0.")
    if n < 5000:
        print("PILOT MODE: use these artifacts for interface testing, not as full-benchmark research results.")

    report: dict[str, Any] = {
        "protocol": "Localization-aware calibration on COCO val2017 with greedy same-class IoU matching",
        "target_definition": "correctness * IoU: IoU for matched same-class prediction with IoU >= 0.50; otherwise 0",
        "iou_match_threshold": IOU_THRESHOLD,
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
        raw_by_split: dict[str, list[float]] = {k: [] for k in splits}
        target_by_split: dict[str, list[float]] = {k: [] for k in splits}
        start_time = time.perf_counter()
        for split_name, records in splits.items():
            print(f"[{model_key}] inferencing {split_name}: {len(records)} images", flush=True)
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
                names = {int(k): str(v) for k, v in model.names.items()} if isinstance(model.names, dict) else {i: str(v) for i, v in enumerate(model.names)}
                for rec, result in zip(batch, results):
                    sc, targets = label_localization_targets(result, anns_by_id.get(rec["id"], []), names)
                    raw_by_split[split_name].extend(sc)
                    target_by_split[split_name].extend(targets)
                done = min(start_idx + len(batch), len(records))
                if done == len(records) or done % 100 == 0:
                    print(f"  {done}/{len(records)} images", flush=True)

        fit_scores = np.asarray(raw_by_split["fit"], dtype=np.float64)
        fit_targets = np.asarray(target_by_split["fit"], dtype=np.float64)
        if len(fit_scores) < 50 or len(np.unique(fit_targets)) < 2:
            raise RuntimeError(f"Not enough varied localization targets for {model_key}: {len(fit_scores)} detections, targets={len(np.unique(fit_targets))}")
        estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        estimator.fit(fit_scores, fit_targets)
        metadata = {
            "model": model_key,
            "weights": weights,
            "fit_images": len(splits["fit"]),
            "fit_detections": int(len(fit_scores)),
            "iou_match_threshold": IOU_THRESHOLD,
            "seed": args.seed,
            "pilot": n < 5000,
            "target_definition": "correctness * IoU: IoU if greedy same-class match has IoU >= 0.50, else 0",
        }
        artifact_path = out_dir / f"{model_key}_loc_isotonic.joblib"
        joblib.dump({"model_key": model_key, "method": "loc_isotonic", "estimator": estimator, "metadata": metadata}, artifact_path)
        print(f"Saved {artifact_path}")

        model_report: dict[str, Any] = {"metadata": metadata, "elapsed_seconds": round(time.perf_counter() - start_time, 2), "splits": {}}
        for split_name in splits:
            raw_scores = np.asarray(raw_by_split[split_name], dtype=np.float64)
            targets = np.asarray(target_by_split[split_name], dtype=np.float64)
            part: dict[str, Any] = {"raw_vs_localization_target": localization_metrics(raw_scores, targets)}
            if len(raw_scores):
                calibrated = np.clip(estimator.predict(raw_scores), 0.0, 1.0)
                part["loc_isotonic_vs_localization_target"] = localization_metrics(calibrated, targets)
            else:
                part["loc_isotonic_vs_localization_target"] = localization_metrics(raw_scores, targets)
            model_report["splits"][split_name] = part
            print(f"[{model_key} / {split_name}] {json.dumps(part, sort_keys=True)}", flush=True)
        report["models"][model_key] = model_report

    report_path = out_dir / "localization_calibration_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nLocalization-aware isotonic fitting complete. Metrics report: {report_path}")
    print("Use held-out test metrics for evaluation; pilot artifacts are for UI testing, not final research claims." if report["pilot"] else "Review held-out test metrics before making research claims.")


if __name__ == "__main__":
    main()
