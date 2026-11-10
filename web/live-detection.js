(() => {
  "use strict";
  const fileInput = document.getElementById("image-file");
  const dropzone = document.getElementById("dropzone");
  const preview = document.getElementById("original-preview");
  const previewPlaceholder = document.getElementById("preview-placeholder");
  const previewCaption = document.getElementById("preview-caption");
  const filenameLabel = document.getElementById("filename-label");
  const dimensionsLabel = document.getElementById("dimensions-label");
  const runButton = document.getElementById("run-button");
  const statusMessage = document.getElementById("status-message");
  const threshold = document.getElementById("conf-threshold");
  const thresholdValue = document.getElementById("threshold-value");
  const thresholdLabel = document.getElementById("threshold-label");
  const thresholdHelp = document.getElementById("threshold-help");
  const resultsSection = document.getElementById("results-section");
  const modelResults = document.getElementById("model-results");
  const calibrationSelect = document.getElementById("calibration-method");
  const methodPillLabel = document.getElementById("method-pill-label");
  const methodHelp = document.getElementById("method-help");
  const resultsMethodBadge = document.getElementById("results-method-badge");
  const resultsIntro = document.getElementById("results-intro");
  const footerMethod = document.getElementById("footer-method");
  const analysisModeSelect = document.getElementById("analysis-mode");
  const analysisHelp = document.getElementById("analysis-help");
  const riskTargetWrap = document.getElementById("risk-target-wrap");
  const riskTargetSelect = document.getElementById("risk-target");
  const mcPassesWrap = document.getElementById("mc-passes-wrap");
  const mcPassesSelect = document.getElementById("mc-passes");
  const analysisSummary = document.getElementById("analysis-summary");
  const MAX_FILE_BYTES = 12 * 1024 * 1024;
  let selectedFile = null;
  let imageObjectUrl = null;

  const methodNames = {
    none: "None (Raw Confidence)", platt: "Platt Scaling", isotonic: "Isotonic Regression",
    loc_isotonic: "Localization-Aware Isotonic"
  };
  const methodDescriptions = {
    none: "Original detector scores; no confidence calibration is applied.",
    platt: "Uses fitted sigmoid mappings for each model. Fit artifacts must exist for both models.",
    isotonic: "Uses fitted monotonic mappings for each model. Fit artifacts must exist for both models.",
    loc_isotonic: "Maps raw confidence to correctness × IoU; a localization-aware quality score, not a plain probability of correctness."
  };
  const analysisNames = {
    standard: "Standard detection / score calibration",
    crc: "Conformal Risk Control (CRC-style)",
    mc_dropout: "Monte Carlo Dropout (experimental)"
  };
  const analysisDescriptions = {
    standard: "Runs deterministic inference and optionally applies a fitted score calibrator.",
    crc: "Uses a threshold fitted on labeled COCO calibration images to control the per-image chance of at least one false-positive detection. A pilot is not a final research guarantee.",
    mc_dropout: "Runs repeated passes with dropout injected at Detect-head features. These pretrained models were not trained with dropout, so the statistics are exploratory, not calibrated probabilities."
  };
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const bytesToMb = (bytes) => `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  function setStatus(message, kind = "") { statusMessage.textContent = message; statusMessage.className = `status-message ${kind}`.trim(); }

  function updateControls() {
    const mode = analysisModeSelect.value;
    const standard = mode === "standard";
    const crc = mode === "crc";
    const mc = mode === "mc_dropout";
    calibrationSelect.disabled = !standard;
    riskTargetWrap.hidden = !crc;
    mcPassesWrap.hidden = !mc;
    threshold.disabled = crc;
    if (crc) {
      methodPillLabel.textContent = analysisNames.crc;
      methodHelp.textContent = "Score calibration is not applied in CRC mode; the fitted threshold controls retained detections.";
      thresholdLabel.textContent = "Raw confidence threshold";
      thresholdHelp.textContent = "Disabled in CRC mode. The threshold comes from the labeled calibration split.";
      footerMethod.textContent = "Risk-control threshold · Per-image false-positive occurrence risk";
      resultsIntro.textContent = "CRC mode filters predictions using a threshold selected from labeled calibration images. The risk target refers to the chance of at least one false-positive detection per image, not a calibrated confidence probability.";
    } else if (mc) {
      methodPillLabel.textContent = analysisNames.mc_dropout;
      methodHelp.textContent = "Score calibration is not applied in MC Dropout mode; raw confidence is compared with stochastic pass statistics.";
      thresholdLabel.textContent = "Baseline raw confidence threshold";
      thresholdHelp.textContent = "Filters the deterministic baseline detections before showing MC pass statistics.";
      footerMethod.textContent = "Experimental MC Dropout · Stochastic presence and score spread";
      resultsIntro.textContent = "Presence frequency, mean matched confidence, IoU consistency, and score spread come from repeated stochastic passes. This post-hoc dropout probe is exploratory and not a calibrated Bayesian posterior.";
    } else {
      const method = calibrationSelect.value;
      methodPillLabel.textContent = methodNames[method];
      methodHelp.textContent = methodDescriptions[method];
      thresholdLabel.textContent = method === "none" ? "Detection confidence threshold" : method === "loc_isotonic" ? "Localization-aware score threshold" : "Displayed score threshold";
      thresholdHelp.textContent = method === "none" ? "Lower thresholds show more predictions. This is a detection filter, not calibration." : method === "loc_isotonic" ? "This threshold filters the localization-aware output score." : "The filter is applied to the selected method’s output score.";
      footerMethod.textContent = method === "none" ? "Raw confidence baseline · No calibration applied" : method === "loc_isotonic" ? "Localization-aware score · Correctness × IoU target" : `${methodNames[method]} · Fitted post-hoc calibration`;
      resultsIntro.textContent = method === "none" ? "Scores below are original detector confidence outputs. A score of 0.90 is not a guarantee that a prediction is correct with 90% probability." : method === "loc_isotonic" ? "Localization-Aware Isotonic predicts a correctness × IoU quality target, not only probability of correctness." : "Calibrated scores estimate detection correctness under same-class IoU ≥ 0.50 matching on COCO val2017. Transfer to other image distributions must be validated.";
    }
    analysisHelp.textContent = analysisDescriptions[mode];
    resultsSection.hidden = true;
    analysisSummary.hidden = true;
  }

  function chooseFile(file) {
    if (!file) return;
    const validMime = ["image/jpeg", "image/png", "image/webp"].includes(file.type);
    if (!validMime) { setStatus("Please choose a JPG, PNG, or WEBP image.", "error"); return; }
    if (file.size > MAX_FILE_BYTES) { setStatus(`That image is ${bytesToMb(file.size)}. Please choose a file under 12 MB.`, "error"); return; }
    selectedFile = file;
    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = URL.createObjectURL(file);
    preview.src = imageObjectUrl; preview.hidden = false; previewPlaceholder.hidden = true; previewCaption.hidden = false;
    filenameLabel.textContent = file.name; dimensionsLabel.textContent = "Loading dimensions…";
    preview.onload = () => { dimensionsLabel.textContent = `${preview.naturalWidth} × ${preview.naturalHeight}px`; };
    runButton.disabled = false; resultsSection.hidden = true; modelResults.replaceChildren();
    setStatus(`Selected ${file.name} (${bytesToMb(file.size)}). Ready to run both models.`);
  }
  fileInput.addEventListener("change", () => chooseFile(fileInput.files?.[0]));
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("drag-over"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", (event) => { event.preventDefault(); dropzone.classList.remove("drag-over"); chooseFile(event.dataTransfer?.files?.[0]); });
  threshold.addEventListener("input", () => { thresholdValue.textContent = Number(threshold.value).toFixed(2); });
  calibrationSelect.addEventListener("change", updateControls);
  analysisModeSelect.addEventListener("change", updateControls);

  function renderModelCard(model) {
    const mode = model.analysis_mode || analysisModeSelect.value;
    const isCrc = mode === "crc";
    const isMc = mode === "mc_dropout";
    const isRaw = !isCrc && !isMc && model.calibration_method === "none";
    let header = "<tr><th>Class</th><th>Raw score</th><th>Score used</th></tr>";
    let rows;
    if (isMc) {
      header = "<tr><th>Class</th><th>Raw score</th><th>MC mean score</th><th>Presence</th><th>Score SD</th></tr>";
      rows = model.detections.map((d) => `<tr><td><span class="class-chip">${escapeHtml(d.class_name)}</span></td><td>${Number(d.raw_confidence).toFixed(3)}</td><td>${Number(d.mc_mean_confidence).toFixed(3)}</td><td>${Number(d.mc_presence_frequency_percent).toFixed(1)}%</td><td>${Number(d.mc_confidence_std).toFixed(3)}</td></tr>`).join("");
    } else {
      rows = model.detections.map((d) => {
        const raw = Number(d.raw_confidence ?? d.confidence);
        const display = Number(d.display_score ?? d.display_confidence ?? d.confidence);
        const col = isCrc ? "Retained by CRC" : display.toFixed(4) + ` (${(display * 100).toFixed(2)}%)`;
        return `<tr><td><span class="class-chip">${escapeHtml(d.class_name)}</span></td><td>${raw.toFixed(4)} (${(raw * 100).toFixed(2)}%)</td><td>${isCrc ? "Yes" : col}</td></tr>`;
      }).join("");
    }
    const tableOrEmpty = model.detection_count ? `<table class="detection-table">${header}<tbody>${rows}</tbody></table>` : `<div class="empty-detections">No detections passed the selected threshold or risk policy.</div>`;
    let stats = `<div class="metric"><span>Detections shown</span><strong>${model.detection_count}</strong></div><div class="metric"><span>${isMc ? "Total MC inference" : "Prediction call"}</span><strong>${Number(model.inference_ms).toFixed(2)}<small>ms</small></strong></div>`;
    if (isMc) stats += `<div class="metric"><span>Stochastic passes</span><strong>${model.mc_passes}</strong></div><div class="metric"><span>Dropout p</span><strong>${Number(model.mc_dropout_probability).toFixed(2)}</strong></div>`;
    if (isCrc && model.risk_control) stats += `<div class="metric"><span>CRC threshold</span><strong>${Number(model.risk_control.threshold).toFixed(2)}</strong></div><div class="metric"><span>Calib. risk upper bound</span><strong>${Number(model.risk_control.risk_upper_bound).toFixed(3)}</strong></div>`;
    const tableTitle = isMc ? "Raw score and stochastic-pass uncertainty" : isCrc ? "Detections retained by the fitted CRC threshold" : model.calibration_method === "loc_isotonic" ? "Raw confidence vs. localization-aware score" : "Raw score vs. score used by selected method";
    return `<article class="model-card"><div class="model-card-head"><div><h3>${model.model === "yolov8n" ? "YOLOv8n" : "YOLO11n"}</h3><div class="weight-label">Checkpoint: ${escapeHtml(model.weights)}</div></div><span class="stat-pill">${isCrc ? "RISK CONTROL" : isMc ? "MC UNCERTAINTY" : isRaw ? "RAW SCORES" : escapeHtml((model.calibration_method || "").toUpperCase())}</span></div><div class="result-image-wrap"><img src="${model.annotated_image}" alt="${model.model} annotated detection output"></div><div class="model-stats">${stats}</div><div class="detections"><h4>${tableTitle}</h4>${tableOrEmpty}<p class="confidence-note">${escapeHtml(model.confidence_note || "")}</p></div></article>`;
  }

  runButton.addEventListener("click", async () => {
    if (!selectedFile) return;
    runButton.disabled = true; runButton.innerHTML = '<span>Running analysis…</span><span aria-hidden="true">⟳</span>';
    resultsSection.hidden = true; analysisSummary.hidden = true;
    const mode = analysisModeSelect.value;
    const method = mode === "standard" ? calibrationSelect.value : "none";
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);
    formData.append("confidence", threshold.value);
    formData.append("calibration", method);
    formData.append("analysis_mode", mode);
    formData.append("risk_target", riskTargetSelect.value);
    formData.append("mc_passes", mcPassesSelect.value);
    setStatus(`Running ${analysisNames[mode]} for YOLOv8n and YOLO11n. First inference can take longer while models initialize.`, "loading");
    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Server error (${response.status}).`);
      if (!Array.isArray(payload.results) || payload.results.length !== 2) throw new Error("The server returned an unexpected response.");
      modelResults.innerHTML = payload.results.map(renderModelCard).join("");
      resultsSection.hidden = false;
      resultsMethodBadge.textContent = payload.analysis_label || (payload.method_label || methodNames[method]);
      if (mode === "crc") {
        const controls = payload.results.map((r) => `${r.model}: threshold ${Number(r.risk_control.threshold).toFixed(2)}, calibrated upper risk bound ${Number(r.risk_control.risk_upper_bound).toFixed(3)}`).join(" · ");
        analysisSummary.textContent = `Target per-image false-positive risk: ${(Number(payload.risk_target) * 100).toFixed(0)}%. ${controls}. Pilot policy only; bounds assume exchangeable images.`;
        analysisSummary.hidden = false;
      } else if (mode === "mc_dropout") {
        analysisSummary.textContent = `Experimental post-hoc dropout: ${payload.mc_passes} stochastic passes. Presence frequency and confidence spread are exploratory signals; the pretrained models were not trained with dropout.`;
        analysisSummary.hidden = false;
      }
      if (payload.analysis_note) resultsIntro.textContent = payload.analysis_note;
      else if (mode === "standard") updateControls();
      setStatus(`Completed on ${payload.image_width} × ${payload.image_height}px image: ${payload.results[0].detection_count} YOLOv8n detections and ${payload.results[1].detection_count} YOLO11n detections.`, "");
      resultsSection.hidden = false;
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      setStatus(`${error.message} Make sure the FastAPI server is running and check the Ubuntu terminal for details.`, "error");
    } finally {
      runButton.disabled = !selectedFile;
      runButton.innerHTML = '<span>Run both models</span><span aria-hidden="true">→</span>';
    }
  });

  fetch("/api/health").then((response) => response.ok ? response.json() : Promise.reject(new Error("API is not available"))).then((health) => {
    if (health.status !== "ok") throw new Error("API health check failed");
    for (const [method, info] of Object.entries(health.calibrators || {})) {
      const option = calibrationSelect.querySelector(`option[value="${method}"]`);
      if (option) { option.disabled = !info.available; option.textContent = info.available ? methodNames[method] : `${methodNames[method] || method} (fit required)`; }
    }
    const crcOption = analysisModeSelect.querySelector('option[value="crc"]');
    const crcMode = health.analysis_modes?.crc || {};
    if (crcOption) crcOption.disabled = !crcMode.available;
    for (const option of riskTargetSelect.options) {
      const key = Number(option.value).toFixed(2);
      option.disabled = !(crcMode.targets && crcMode.targets[key]);
    }
    if (riskTargetSelect.selectedOptions[0]?.disabled) {
      const first = [...riskTargetSelect.options].find((o) => !o.disabled);
      if (first) riskTargetSelect.value = first.value;
    }
    updateControls();
    setStatus("Backend connected. Select an image to begin.");
  }).catch(() => setStatus("Backend not detected yet. Start the app with the provided uvicorn command, then refresh this page.", "error"));
})();