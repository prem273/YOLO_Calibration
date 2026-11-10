# Live YOLO Raw-Confidence Detector Add-on

This add-on is designed to be copied into the **root of the cloned `YOLO_Calibration` repository**. It adds a separate `/live-detection` page and a FastAPI endpoint while leaving existing research-dashboard files untouched.

## Files

- `inference_api.py` — FastAPI backend for YOLOv8n and YOLO11n
- `web/live-detection.html` — separate upload and comparison page
- `web/live-detection.css` — styles for the new page
- `web/live-detection.js` — browser upload, API call, and results rendering

## Install the inference dependencies

From the repository root, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```

## Run locally (Windows / PowerShell)

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Then open:

- Existing research dashboard: http://127.0.0.1:8000/
- New live detector: http://127.0.0.1:8000/live-detection
- API health check: http://127.0.0.1:8000/api/health

On first inference, Ultralytics should download `yolov8n.pt` and `yolo11n.pt` if they are not already cached. Ensure the machine has internet access for this first run and sufficient disk space. The two models run sequentially on the same uploaded image. Output confidence values are raw, uncalibrated model scores; no calibration method is applied. The confidence threshold is only a detection filter.

## Upload constraints

JPG, PNG, and WEBP up to 12 MB. Uploaded images are processed locally by this backend and are not written to the project directory by this code.

## Note on latency

The UI measures each model's `predict` call and excludes checkpoint download/loading and browser rendering. First-run initialization may affect measured time. These are per-image timings on the current local device, not a reproduction of the research benchmark's reported latency.

## After copying into the repository

Check what changed:

```powershell
git status
```

Run the server and verify both pages before committing. Then stage only the added files:

```powershell
git add inference_api.py requirements-inference.txt web/live-detection.html web/live-detection.css web/live-detection.js README_ADDON.md
git commit -m "Add live raw-confidence YOLO comparison interface"
```

Pushing to `origin` requires write access to the repository. Do not push to someone else's repository unless you have permission; otherwise fork it and push to your own fork.
