#!/usr/bin/env python3
"""Fit a finite-sample risk-control threshold for YOLOv8n and YOLO11n.

Loss per calibration image is 1 if at least one retained detection is incorrect
(same-class IoU < 0.50 or unmatched) and 0 otherwise. This bounded loss is
monotone as the confidence threshold is lowered, so a CRC-style threshold can
be selected from a fixed grid using a one-sided Hoeffding bound. Multiple
predeclared risk targets are handled by a union-bound correction.

This is a simple finite-sample risk-control implementation, not a claim that it
reproduces every detail of canonical Conformal Risk Control. Its risk guarantee
assumes exchangeability of calibration and future images and does not cover
arbitrary distribution shift. Pilot mode is for UI development only.
"""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
from typing import Any
import numpy as np
from ultralytics import YOLO
from fit_calibrators import MODEL_SPECS, PREDICTION_CONFIDENCE_FLOOR, IOU_THRESHOLD, load_coco, label_predictions

RISK_TARGETS = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.05, 1.00, 0.05)] + [1.01]
DEFAULT_DELTA = 0.05


def image_loss(scores: list[float], labels: list[int], threshold: float) -> float:
    keep = [i for i, score in enumerate(scores) if score >= threshold]
    if not keep:
        return 0.0
    return float(any(labels[i] == 0 for i in keep))


def retention_fraction(records: list[dict[str, Any]], threshold: float) -> float:
    total = sum(len(rec["scores"]) for rec in records)
    kept = sum(sum(score >= threshold for score in rec["scores"]) for rec in records)
    return float(kept / total) if total else 0.0


def summarize(records: list[dict[str, Any]], threshold: float) -> dict[str, float | int]:
    losses = [image_loss(r["scores"], r["labels"], threshold) for r in records]
    kept = sum(sum(score >= threshold for score in r["scores"]) for r in records)
    total = sum(len(r["scores"]) for r in records)
    return {
        "images": len(records),
        "empirical_per_image_false_positive_risk": round(float(np.mean(losses)) if losses else 0.0, 6),
        "images_with_at_least_one_false_positive": int(sum(losses)),
        "retained_detections": int(kept),
        "all_candidate_detections": int(total),
        "retention_fraction": round(float(kept / total) if total else 0.0, 6),
        "mean_retained_detections_per_image": round(float(kept / len(records)) if records else 0.0, 6),
    }


def infer_records(model: YOLO, records: list[dict[str, Any]], anns_by_id: dict[int, list[dict[str, Any]]], device: str, batch_size: int, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    names = {int(k): str(v) for k, v in model.names.items()} if isinstance(model.names, dict) else {i: str(v) for i, v in enumerate(model.names)}
    for start in range(0, len(records), max(1, batch_size)):
        batch = records[start:start + max(1, batch_size)]
        results = model.predict(source=[str(r["path"]) for r in batch], conf=PREDICTION_CONFIDENCE_FLOOR,
                                imgsz=640, device=device, batch=max(1, batch_size), max_det=300, verbose=False, stream=False)
        for rec, result in zip(batch, results):
            scores, labels = label_predictions(result, anns_by_id.get(rec["id"], []), names)
            output.append({"image_id": rec["id"], "file_name": rec["file_name"], "scores": scores, "labels": labels})
        done = min(start + len(batch), len(records))
        print(f"  [{split}] {done}/{len(records)} images", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit CRC-style risk-control thresholds for YOLOv8n and YOLO11n.")
    parser.add_argument("--data-dir", default="data/coco", help="COCO val2017 root")
    parser.add_argument("--max-images", type=int, default=500, help="500 for pilot; 5000 for the full protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu or Ultralytics device such as 0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="Failure probability for the risk upper bound")
    parser.add_argument("--output-dir", default="calibration_artifacts")
    args = parser.parse_args()
    if args.max_images < 100:
        raise SystemExit("Use at least 100 images; recommended 500 for UI pilot or 5000 for evaluation.")
    if not 0 < args.delta < 1:
        raise SystemExit("--delta must lie strictly between 0 and 1.")

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    all_images, anns_by_id, _ = load_coco(data_dir)
    rng = random.Random(args.seed)
    chosen = rng.sample(all_images, min(args.max_images, len(all_images)))
    rng.shuffle(chosen)
    n = len(chosen); n_fit = int(n * .40); n_val = int(n * .20)
    splits = {"fit": chosen[:n_fit], "validation": chosen[n_fit:n_fit+n_val], "test": chosen[n_fit+n_val:]}
    print(f"COCO images available: {len(all_images)}")
    print(f"Selected {n} images; split: fit={len(splits['fit'])}, validation={len(splits['validation'])}, test={len(splits['test'])}")
    print("Risk target: per-image event that at least one retained detection is incorrect (same-class IoU < 0.50 or unmatched).")
    if n < 5000: print("PILOT MODE: artifacts are for UI testing, not final research claims.")

    # Union-bound correction makes the predeclared risk-target choices simultaneous.
    radius = math.sqrt(math.log(len(RISK_TARGETS) / args.delta) / (2 * len(splits["fit"])))
    report: dict[str, Any] = {
        "method": "CRC-style finite-sample risk control via monotone bounded image-level loss and Hoeffding upper bound",
        "risk_definition": "Per-image false-positive occurrence: 1 if any retained detection is incorrect (same-class IoU < 0.50 or unmatched), else 0.",
        "assumptions": ["Calibration and future images are exchangeable", "Same-class greedy matching at IoU >= 0.50 defines prediction correctness", "The threshold grid and risk target grid were fixed before evaluation"],
        "iou_threshold": IOU_THRESHOLD, "confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
        "seed": args.seed, "selected_images": n, "split_images": {k: len(v) for k,v in splits.items()},
        "risk_targets": RISK_TARGETS, "threshold_grid": THRESHOLDS, "delta": args.delta,
        "risk_bound_radius": round(radius, 6), "simultaneous_targets_correction": len(RISK_TARGETS),
        "pilot": n < 5000, "models": {},
    }

    for model_key, weights in MODEL_SPECS.items():
        print(f"\nLoading {weights} ...", flush=True)
        model = YOLO(weights)
        model_records = {}
        start_time = time.perf_counter()
        for split_name, recs in splits.items():
            print(f"[{model_key}] inferencing {split_name}: {len(recs)} images", flush=True)
            model_records[split_name] = infer_records(model, recs, anns_by_id, args.device, args.batch_size, split_name)
        fit_recs = model_records["fit"]
        thresholds_for_model = {}
        for target in RISK_TARGETS:
            best = None
            for threshold in THRESHOLDS:  # ascending => maximum retention is first feasible threshold
                stats = summarize(fit_recs, threshold)
                upper = min(1.0, float(stats["empirical_per_image_false_positive_risk"]) + radius)
                if upper <= target:
                    best = {"certified": True, "threshold": float(threshold),
                            "risk_target": float(target), "empirical_risk": float(stats["empirical_per_image_false_positive_risk"]),
                            "risk_upper_bound": float(upper), "bound_radius": float(radius),
                            "calibration_images": len(fit_recs), "delta": args.delta,
                            "retention_fraction": float(stats["retention_fraction"]),
                            "mean_retained_detections_per_image": float(stats["mean_retained_detections_per_image"]),
                            "images_with_at_least_one_false_positive": int(stats["images_with_at_least_one_false_positive"]),
                            "method": report["method"]}
                    break
            if best is None:
                best = {"certified": False, "threshold": None, "risk_target": float(target),
                        "risk_upper_bound_floor": float(radius), "calibration_images": len(fit_recs), "delta": args.delta,
                        "reason": "No threshold on the fixed grid met the requested risk target under the simultaneous risk bound."}
            else:
                # Evaluate the selected threshold on untouched splits; these values are diagnostic only.
                for split_name in ("validation", "test"):
                    best[f"{split_name}_metrics"] = summarize(model_records[split_name], float(best["threshold"]))
            thresholds_for_model[f"{target:.2f}"] = best
            tag = f"{model_key} target={target:.2f}"
            if best["certified"]:
                print(f"{tag}: threshold={best['threshold']:.2f} upper_bound={best['risk_upper_bound']:.3f} retention={best['retention_fraction']:.3f} test-risk={best['test_metrics']['empirical_per_image_false_positive_risk']:.3f}", flush=True)
            else:
                print(f"{tag}: NOT FEASIBLE (minimum bound radius={radius:.3f})", flush=True)
        report["models"][model_key] = {"weights": weights, "elapsed_seconds": round(time.perf_counter()-start_time, 2), "thresholds": thresholds_for_model}

    out_path = output_dir / "crc_risk_thresholds.json"
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nCRC-style risk-control fitting complete: {out_path}")
    print("Use calibration upper bounds for the selected threshold; held-out test risk is a diagnostic estimate, not the bound.")
    print("Pilot artifacts are for interface development; run --max-images 5000 and audit assumptions before research claims." if report["pilot"] else "Full-size outputs saved. Review held-out test metrics and assumptions before reporting.")

if __name__ == "__main__":
    main()
