# CRC Risk Control + Monte Carlo Dropout add-on

This add-on extends the existing live detector with two separate **analysis modes**:

1. **CRC-style risk control**: fit a raw-confidence threshold using COCO ground-truth labels and a bounded, monotone image-level loss. Loss is 1 when an image has at least one retained incorrect detection (same-class match at IoU >= 0.50 fails or the prediction is unmatched), otherwise 0. The script uses a one-sided Hoeffding upper bound and a correction across the predeclared risk-target choices. It chooses the lowest (most permissive) threshold whose bound is at or below the target.
2. **MC Dropout (experimental)**: repeats predictions with dropout (`p=0.10`) injected into the Detect-head feature inputs. It reports per-detection presence frequency, mean matched confidence, confidence standard deviation, and mean matched IoU across passes.

## Important scientific limitations

- The CRC mode is explicitly labelled **CRC-style risk control** because this deployment add-on implements a bounded monotone image-level risk with a simultaneous Hoeffding bound; do not claim it duplicates every detail of canonical Conformal Risk Control without a separate methodological review.
- The risk bound relies on exchangeability between labeled calibration images and future inputs. It does not guarantee performance under arbitrary distribution shift.
- The current 500-image setting is a **pilot for UI development**, not the final research benchmark. The script uses 40% fit/calibration, 20% validation, and 40% test images. Only the fit/calibration split chooses the threshold; validation/test metrics are diagnostics.
- MC Dropout is post-hoc and the pretrained models were not trained with dropout. It is an exploratory stochastic probe, not a calibrated probability or validated Bayesian posterior. This deployment mode runs both models and may not exactly match the research notebook's original MC model configuration.
- Do not commit COCO images, model weights, or the generated `calibration_artifacts/` directory unless you intentionally decide to version them.

## Install/update

Extract the add-on over the existing project. Keep `requirements.txt` and the existing research outputs intact.

```bash
source ~/venvs/yolo-calibration/bin/activate
cd "/mnt/c/Users/PREM KUMAR JHA/Downloads/YOLO_Calibration"
python -m pip install -r requirements-inference.txt
```

## Fit the CRC-style threshold policy

Run a 500-image pilot first:

```bash
python fit_crc.py --data-dir data/coco --max-images 500 --device cpu
```

This writes `calibration_artifacts/crc_risk_thresholds.json`. Inspect target feasibility at `/api/health`. For a full-size evaluation, use `--max-images 5000` only after the pilot works and review the protocol/results before making claims.

## Run the app

Stop the previous server with Ctrl+C, then:

```bash
python -m uvicorn inference_api:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/live-detection` and press Ctrl+F5. The Analysis mode selector contains Standard Detection, CRC-style Risk Control, and MC Dropout (experimental). CRC risk targets are enabled only when both model policies meet the selected target.

## How to interpret the outputs

- **Standard**: uses the existing raw/Platt/Isotonic/Localization-Aware Isotonic flow.
- **CRC**: uses a fitted raw score threshold and reports its calibration empirical risk and upper bound. The threshold is model-specific; do not interpret the target as confidence of each individual object.
- **MC Dropout**: output table shows raw confidence, mean matched confidence over stochastic passes, presence frequency, and confidence standard deviation. Larger variability or lower presence frequency indicate less stable predictions, but are not calibrated risk probabilities.
