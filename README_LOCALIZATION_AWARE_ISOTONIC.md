# Localization-Aware Isotonic add-on

This patch adds **Localization-Aware Isotonic** to the existing live detector. It is distinct from ordinary Isotonic Regression.

## Target definition
For each predicted box, use greedy same-class matching to an unused COCO ground-truth box. A matched prediction with IoU >= 0.50 receives target `IoU`; otherwise the target is `0`. This is the defined `correctness × IoU` target. The resulting output should be interpreted as a localization-aware quality score, not a plain probability of correctness.

## Run in Ubuntu/WSL
From the project root with `(yolo-calibration)` activated:

```bash
python fit_localization_isotonic.py --data-dir data/coco --max-images 500 --device cpu
```

The pilot uses 200 fit / 100 validation / 200 test images. It writes `calibration_artifacts/yolov8n_loc_isotonic.joblib`, `calibration_artifacts/yolo11n_loc_isotonic.joblib`, and `calibration_artifacts/localization_calibration_metrics.json`. These artifacts are for UI testing only; use `--max-images 5000` for the full split before research claims.

Restart the backend after fitting:

```bash
python -m uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Refresh `http://localhost:8000/live-detection` with Ctrl+F5. The option is enabled only when both fitted artifacts exist.
