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
  const resultsSection = document.getElementById("results-section");
  const modelResults = document.getElementById("model-results");
  const MAX_FILE_BYTES = 12 * 1024 * 1024;
  let selectedFile = null;
  let imageObjectUrl = null;

  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const bytesToMb = (bytes) => `${(bytes / (1024 * 1024)).toFixed(2)} MB`;

  function setStatus(message, kind = "") {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${kind}`.trim();
  }

  function formatDimensions(image) {
    return `${image.naturalWidth} × ${image.naturalHeight}px`;
  }

  function chooseFile(file) {
    if (!file) return;
    const validMime = ["image/jpeg", "image/png", "image/webp"].includes(file.type);
    if (!validMime) {
      setStatus("Please choose a JPG, PNG, or WEBP image.", "error");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setStatus(`That image is ${bytesToMb(file.size)}. Please choose a file under 12 MB.`, "error");
      return;
    }
    selectedFile = file;
    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = URL.createObjectURL(file);
    preview.src = imageObjectUrl;
    preview.hidden = false;
    previewPlaceholder.hidden = true;
    previewCaption.hidden = false;
    filenameLabel.textContent = file.name;
    dimensionsLabel.textContent = "Loading dimensions…";
    preview.onload = () => { dimensionsLabel.textContent = formatDimensions(preview); };
    runButton.disabled = false;
    resultsSection.hidden = true;
    modelResults.replaceChildren();
    setStatus(`Selected ${file.name} (${bytesToMb(file.size)}). Ready to run both models.`);
  }

  fileInput.addEventListener("change", () => chooseFile(fileInput.files?.[0]));
  dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("drag-over"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
    chooseFile(event.dataTransfer?.files?.[0]);
  });
  threshold.addEventListener("input", () => { thresholdValue.textContent = Number(threshold.value).toFixed(2); });

  function renderModelCard(model) {
    const rows = model.detections.map((detection) => `
      <tr>
        <td><span class="class-chip">${escapeHtml(detection.class_name)}</span></td>
        <td>${escapeHtml(detection.confidence.toFixed(4))}</td>
        <td>${escapeHtml(detection.confidence_percent.toFixed(2))}%</td>
      </tr>`).join("");
    const tableOrEmpty = model.detection_count
      ? `<table class="detection-table"><thead><tr><th>Class</th><th>Raw score</th><th>Confidence</th></tr></thead><tbody>${rows}</tbody></table>`
      : `<div class="empty-detections">No objects passed this confidence threshold. Try lowering the threshold.</div>`;

    return `
      <article class="model-card">
        <div class="model-card-head">
          <div><h3>${model.model === "yolov8n" ? "YOLOv8n" : "YOLO11n"}</h3><div class="weight-label">Checkpoint: ${escapeHtml(model.weights)}</div></div>
          <span class="stat-pill">RAW SCORES</span>
        </div>
        <div class="result-image-wrap"><img src="${model.annotated_image}" alt="${model.model} annotated detection output"></div>
        <div class="model-stats">
          <div class="metric"><span>Detections</span><strong>${model.detection_count}</strong></div>
          <div class="metric"><span>Prediction call</span><strong>${model.inference_ms.toFixed(2)}<small>ms</small></strong></div>
        </div>
        <div class="detections"><h4>Detected objects and raw confidence</h4>${tableOrEmpty}</div>
      </article>`;
  }

  runButton.addEventListener("click", async () => {
    if (!selectedFile) return;
    runButton.disabled = true;
    runButton.innerHTML = '<span>Running both models…</span><span aria-hidden="true">⟳</span>';
    resultsSection.hidden = true;
    setStatus("Preparing image and running YOLOv8n, then YOLO11n. The first run may take longer while weights download.", "loading");
    const formData = new FormData();
    formData.append("file", selectedFile, selectedFile.name);
    formData.append("confidence", threshold.value);
    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `Server error (${response.status}).`);
      if (!Array.isArray(payload.results) || payload.results.length !== 2) throw new Error("The server returned an unexpected response.");
      modelResults.innerHTML = payload.results.map(renderModelCard).join("");
      resultsSection.hidden = false;
      setStatus(`Completed on ${payload.image_width} × ${payload.image_height}px image: ${payload.results[0].detection_count} YOLOv8n detections and ${payload.results[1].detection_count} YOLO11n detections.`, "");
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      setStatus(`${error.message} Make sure the FastAPI server is running and check the VS Code terminal for details.`, "error");
    } finally {
      runButton.disabled = !selectedFile;
      runButton.innerHTML = '<span>Run both models</span><span aria-hidden="true">→</span>';
    }
  });

  fetch("/api/health")
    .then((response) => response.ok ? response.json() : Promise.reject(new Error("API is not available")))
    .then((health) => {
      if (health.status === "ok") setStatus("Backend connected. Select an image to begin.");
    })
    .catch(() => setStatus("Backend not detected yet. Start the app with the provided uvicorn command, then refresh this page.", "error"));
})();
