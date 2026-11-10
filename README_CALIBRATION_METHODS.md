# YOLO Calibration Methods Add-on

This add-on adds genuine post-hoc confidence calibration for the live YOLOv8n / YOLO11n comparison page. It includes `None (Raw Confidence)`, Platt Scaling, and Isotonic Regression. Calibration choices remain disabled in the page until both models have fitted artifacts.

## Important definition

The fitting script labels a detection as correct when it greedily matches an *unmatched* COCO ground-truth box of the same category at IoU >= 0.50. This is a clear object-detection correctness target, but it is not the full COCO AP evaluation protocol. The calibrator estimates this detection-correctness probability and can behave differently under distribution shift.

## 1. Install the additional dependencies

From Ubuntu / WSL, in the repository root:

```bash
source ~/venvs/yolo-calibration/bin/activate
cd "/mnt/c/Users/PREM KUMAR JHA/Downloads/YOLO_Calibration"
python -m pip install -r requirements-inference.txt
```

## 2. Fit pilot calibrators first

```bash
python fit_calibrators.py --data-dir data/coco --max-images 500 --device cpu
```

This uses a deterministic image-level split of 200 fit / 100 validation / 200 test images. It runs both models and may take a while on CPU. These pilot artifacts are for exercising the UI, not for claiming full benchmark results.

## 3. Fit the full 5,000-image split

When the pilot works, run:

```bash
python fit_calibrators.py --data-dir data/coco --max-images 5000 --device cpu
```

This uses 2,000 fit / 1,000 validation / 2,000 test images. It makes two model passes over the selected images. Do not run the full fit concurrently with a second YOLO server on the same computer.

Artifacts are written to `calibration_artifacts/`:

- `yolov8n_platt.joblib`, `yolo11n_platt.joblib`
- `yolov8n_isotonic.joblib`, `yolo11n_isotonic.joblib`
- `calibration_metrics.json` with raw / Platt / isotonic ECE, Brier score, and empirical correctness for each split

The script uses raw confidence >= 0.001 from Ultralytics, then applies matching labels using same-class IoU >= 0.50. It prints progress by model, split, and images processed.

## 4. Start the app

Stop any existing server with Ctrl+C, then run:

```bash
python -m uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/live-detection> and refresh. Platt and Isotonic should be selectable once artifacts for both models exist. The threshold slider applies to the score selected by the current method. The API still uses the underlying YOLO detections; calibration changes scores, not model weights.

## Safety and reproducibility

- The script does not alter YOLO checkpoints or the original research result files.
- COCO images, YOLO `.pt` checkpoints, and the local `calibration_artifacts/` folder should not be blindly committed. The `.joblib` artifacts are reproducible from the script + dataset, and the JSON metrics report can be reviewed before sharing.
- `run_calibration.py` in the original repository prints prewritten benchmark summary statements after loading weights; it does not fit the calibrators described here. Use `fit_calibrators.py` for the actual fitting operation.
