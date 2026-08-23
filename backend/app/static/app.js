/**
 * CameraTrace — Explainable Digital Image Forensics & Attribution Platform
 * Complete Frontend Controller & Visual Analysis Engine
 */

// Helper Selectors
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

// Pipeline Stage Labels
const STAGES = {
  QUEUED: "Evidence queued for intake",
  VALIDATING: "Validating image integrity & format",
  METADATA_ANALYSIS: "Extracting embedded metadata & EXIF tags",
  CAMERA_FORENSICS: "Extracting noise, CFA, FFT & camera signals",
  MANIPULATION_ANALYSIS: "Executing ELA, copy-move & splicing detectors",
  CONSISTENCY_ANALYSIS: "Fusing multi-module forensic consistency",
  ROBUSTNESS_ANALYSIS: "Evaluating perturbation stability",
  REPORT_GENERATION: "Generating explainable forensic dossier",
  COMPLETED: "Forensic analysis complete",
  FAILED: "Analysis failed",
};

// Artifact Metadata & Human Descriptions
const ARTIFACT_INFO = {
  suspiciousness_heatmap: {
    title: "Fused Suspiciousness Heatmap",
    subtitle: "Multi-Engine Anomaly Localization",
    desc: "Probabilistic fusion of ELA, noise inconsistency, splicing, and compression boundaries highlighting localized tampering regions.",
  },
  noise_residual: {
    title: "Noise Residual Map",
    subtitle: "High-Frequency Gaussian Residual",
    desc: "Sensor noise pattern extracted by filtering out image content, revealing inconsistencies across spliced or edited segments.",
  },
  frequency_spectrum: {
    title: "2D FFT Spectral Magnitude",
    subtitle: "Log-Scale Fourier Frequency Domain",
    desc: "Radial frequency distribution showing spectral entropy, periodic demosaicing peaks, and unnatural high-frequency cutoffs.",
  },
  ela: {
    title: "Error Level Analysis (ELA)",
    subtitle: "Compression Error Variance",
    desc: "Visualizes error rate differences when resaved at standard JPEG quality. Bright patches indicate different compression histories.",
  },
  noise_heatmap: {
    title: "Local Noise Anomaly Map",
    subtitle: "Spatial Variance Inconsistency",
    desc: "Patch-wise local variance analysis detecting foreign noise signatures spliced from other cameras or synthetic generators.",
  },
  difference_heatmap: {
    title: "Pixel Difference Heatmap",
    subtitle: "Euclidean ECC Comparison",
    desc: "Color-mapped intensity of pixel-level modifications after geometric alignment of original and derivative evidence.",
  },
  change_overlay: {
    title: "Changed-Area Mask Overlay",
    subtitle: "Localized Modification Regions",
    desc: "Original evidence overlaid with red highlight masks marking every detected modified pixel cluster.",
  },
  original: {
    title: "Original Evidence Record",
    subtitle: "Ingested Raw Evidence",
    desc: "Unmodified original image file preserved for cryptographic chain-of-custody verification.",
  },
};

// Application State
let apiBase = localStorage.getItem("cameratrace-api") || "/api/v1";
let activeReport = "";
let currentAnalysisId = null;
let lastPreviewUrl = "";
let pollTimer = null;
let currentArtifactsList = [];
let currentAnalysisBundle = null;
let originalImageDimensions = { width: 0, height: 0 };

// Utility: Resolve Origin
function originFromApi() {
  const base = apiBase.replace(/\/$/, "");
  if (base.startsWith("/")) return window.location.origin;
  try {
    return new URL(base).origin;
  } catch {
    return window.location.origin;
  }
}

// Utility: Toast Notification
function toast(message) {
  const el = $("#toast");
  if (!el) return;
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(el.__timer);
  el.__timer = setTimeout(() => el.classList.remove("show"), 3800);
}

// Utility: Byte Formatting
function formatBytes(bytes) {
  if (bytes == null || isNaN(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// Utility: Percentage Formatting
function formatPct(value) {
  if (value == null || isNaN(value)) return "0%";
  return `${Math.round(Number(value) * 100)}%`;
}

// Utility: HTML Escaping
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// Utility: Parse Error Detail
function parseErrorDetail(payload, status) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || item.detail || JSON.stringify(item)).join("; ");
  }
  if (detail && typeof detail === "object") return detail.message || JSON.stringify(detail);
  return payload?.message || `Request failed (${status})`;
}

// Core API Fetcher
async function call(path, options = {}) {
  const url = `${apiBase.replace(/\/$/, "")}${path}`;
  const res = await fetch(url, options);
  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    throw new Error(parseErrorDetail(payload, res.status));
  }
  if (res.status === 204) return null;
  return res.json();
}

// Artifact URL Formatter (Robust Windows & Linux Path Mapping)
function resolveArtifactUrl(rawPath, analysisId) {
  if (!rawPath) return "";
  if (/^https?:\/\//i.test(rawPath)) return rawPath;
  const origin = originFromApi();
  const normalized = String(rawPath).replaceAll("\\", "/");

  // Check for direct static /artifacts/ prefix
  const marker = "/artifacts/";
  const idx = normalized.toLowerCase().lastIndexOf(marker);
  if (idx >= 0) {
    return `${origin}${normalized.slice(idx)}`;
  }

  // Fallback relative filename
  const filename = normalized.split("/").pop();
  if (analysisId && filename) {
    return `${origin}/artifacts/${analysisId}/${filename}`;
  }
  return "";
}

// Navigation & View Switching
function hideAllViews() {
  ["#intake-view", "#lab-view", "#workspace", "#artifacts-view", "#final-analysis", "#cases-view"].forEach((id) => {
    const el = $(id);
    if (el) el.classList.add("hidden");
  });
  $$("#main-nav .nav-btn").forEach((btn) => btn.classList.remove("active"));
}

function showView(name) {
  hideAllViews();
  const map = {
    intake: "#intake-view",
    lab: "#lab-view",
    workspace: "#workspace",
    artifacts: "#artifacts-view",
    final: "#final-analysis",
    cases: "#cases-view",
  };
  const targetId = map[name] || "#intake-view";
  const el = $(targetId);
  if (el) el.classList.remove("hidden");

  // Activate matching nav button
  const navBtn = $(`#main-nav .nav-btn[data-view="${name}"]`);
  if (navBtn) navBtn.classList.add("active");

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Engine Health Check
async function checkHealth() {
  const statusEl = $("#api-status");
  const indicator = $(".status-indicator", "#system-status-pill");
  try {
    const data = await call("/health");
    if (statusEl) {
      statusEl.textContent = `${data.service || "CameraTrace"} online · v${data.version || "1.0.0"}`;
    }
    if (indicator) {
      indicator.className = "status-indicator live";
    }
  } catch (err) {
    if (statusEl) statusEl.textContent = "Engine offline · check connection";
    if (indicator) indicator.className = "status-indicator offline";
  }
}

// Progress Bar Controller
function setProgress(status, progress = 0) {
  const stageEl = $("#stage-label");
  const valEl = $("#progress-value");
  const barEl = $("#progress-bar");
  const spinner = $("#progress-spinner");

  const label = STAGES[status] || status;
  if (stageEl) stageEl.textContent = label;
  if (valEl) valEl.textContent = `${Math.round(progress)}%`;
  if (barEl) barEl.style.width = `${Math.max(3, progress)}%`;

  if (spinner) {
    spinner.style.display = status === "COMPLETED" || status === "FAILED" ? "none" : "inline-block";
  }
}

// Render Initial Workspace on Upload
function initWorkspace(file, uploadInfo) {
  currentAnalysisId = uploadInfo.analysis_id;
  showView("workspace");

  $("#case-title").textContent = file?.name || uploadInfo.filename || "Evidence Under Review";
  $("#case-id").textContent = uploadInfo.analysis_id.toUpperCase();

  originalImageDimensions = { width: uploadInfo.width || 0, height: uploadInfo.height || 0 };

  const preview = $("#preview-image");
  if (file) {
    if (lastPreviewUrl) URL.revokeObjectURL(lastPreviewUrl);
    lastPreviewUrl = URL.createObjectURL(file);
    preview.src = lastPreviewUrl;
  } else {
    // Attempt to load from static uploads endpoint
    preview.src = `${originFromApi()}/artifacts/uploads/${uploadInfo.analysis_id}/original.jpg`;
  }

  // Reset Layer overlay
  const layerOverlay = $("#layer-overlay-image");
  if (layerOverlay) layerOverlay.classList.add("hidden");
  $$("#layer-selector .layer-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.layer === "original");
  });

  $("#image-format-badge").textContent = uploadInfo.format || "EVIDENCE";
  $("#image-meta").textContent = `${uploadInfo.width} × ${uploadInfo.height} px · ${formatBytes(uploadInfo.file_size)} · ${uploadInfo.format || ""}`;

  // Reset side cards to loading state
  $("#evidence-content").innerHTML = `
    <div class="ledger-list">
      <div class="ledger-row"><span>SHA-256 HASH</span><b>${escapeHtml(uploadInfo.sha256)}</b></div>
      <div class="ledger-row"><span>DIMENSIONS</span><b>${uploadInfo.width} × ${uploadInfo.height} px (${escapeHtml(uploadInfo.format)})</b></div>
      <div class="ledger-row"><span>FILE SIZE</span><b>${formatBytes(uploadInfo.file_size)}</b></div>
    </div>
  `;

  $("#camera-content").innerHTML = `<div class="empty-state-spinner">Extracting noise residual &amp; CFA demosaicing…</div>`;
  $("#integrity-content").innerHTML = `<div class="empty-state-spinner">Running ELA, copy-move &amp; splicing detectors…</div>`;
  $("#integrity-score").textContent = "—";
  $("#report-content").textContent = "A human-readable forensic summary will appear here once analysis is complete.";

  $("#verdict-gauge-value").textContent = "—";
  $("#verdict-title").textContent = "Evaluating Evidence";
  $("#verdict-desc").textContent = "The consistency engine is fusing independent signals from metadata, camera models, and manipulation detectors.";
}

// Render Camera Card
function renderCameraCard(camera) {
  const container = $("#camera-content");
  if (!container || !camera) return;

  const first = camera.top_candidates?.[0];
  const modelName = camera.known_camera
    ? `${camera.manufacturer || ""} ${camera.model || camera.full_label || ""}`.trim()
    : first
      ? first.camera_model || first.label || "Unknown Camera"
      : camera.model || "Unknown / Unseen Camera";

  const confPct = Math.round((camera.confidence || 0) * 100);
  const sourceLabel = camera.prediction_source === "EMBEDDED_METADATA" ? "EMBEDDED EXIF" : "FORENSIC ML";

  $("#camera-source-badge").textContent = sourceLabel;

  const candidatesHtml = (camera.top_candidates || [])
    .slice(0, 3)
    .map(
      (c) => `
      <div class="candidate-mini-row">
        <span>#${c.rank ?? "1"} ${escapeHtml(c.camera_model || c.label)}</span>
        <span>${formatPct(c.confidence)}</span>
      </div>`
    )
    .join("");

  container.innerHTML = `
    <div class="camera-match-box">
      <div class="camera-label-row">
        <span class="camera-name">${escapeHtml(modelName)}</span>
        <span class="camera-conf">${confPct}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-bar-fill" style="width: ${confPct}%"></div>
      </div>
      ${candidatesHtml}
    </div>
  `;
}

// Render Integrity & Tampering Card
function renderManipulationCard(manipulation) {
  const container = $("#integrity-content");
  if (!container || !manipulation) return;

  const score = Math.round((manipulation.overall_suspiciousness || 0) * 100);
  const statusLabel =
    score > 65
      ? "Potentially Manipulated"
      : score > 30
        ? "Suspicious Inconsistencies"
        : "Consistent / Authentic";

  $("#integrity-score").textContent = `${score}%`;

  const chipsHtml = (manipulation.indicators?.length ? manipulation.indicators : ["no_major_anomaly"])
    .map((ind) => `<span class="chip-tag">${escapeHtml(String(ind).replaceAll("_", " "))}</span>`)
    .join("");

  const regionCount = manipulation.regions?.length || 0;
  const typesLabel = (manipulation.types || []).join(", ") || "None";

  container.innerHTML = `
    <div class="integrity-stat-row">
      <span class="stat-label">ASSESSMENT</span>
      <span class="stat-val ${score > 50 ? 'text-accent' : 'text-success'}">${escapeHtml(statusLabel)}</span>
    </div>
    <div class="integrity-stat-row">
      <span class="stat-label">FLAGGED REGIONS</span>
      <span class="stat-val">${regionCount} area(s)</span>
    </div>
    <div class="integrity-stat-row">
      <span class="stat-label">DETECTED TYPE</span>
      <span class="stat-val">${escapeHtml(typesLabel)}</span>
    </div>
    <div class="chips-cloud">${chipsHtml}</div>
  `;

  // Update Verdict Orb
  const gaugeVal = $("#verdict-gauge-value");
  const gaugeRing = $("#verdict-gauge-ring");
  const verdictTitle = $("#verdict-title");
  const verdictDesc = $("#verdict-desc");

  if (gaugeVal) gaugeVal.textContent = `${score}%`;
  if (gaugeRing) {
    const color = score > 60 ? "#E11D48" : score > 30 ? "#F59E0B" : "#10B981";
    gaugeRing.style.background = `conic-gradient(${color} ${score * 3.6}deg, rgba(255,255,255,0.08) 0deg)`;
  }
  if (verdictTitle) verdictTitle.textContent = statusLabel;
  if (verdictDesc) {
    verdictDesc.textContent =
      manipulation.limitations?.[0] ||
      "Forensic signals are synthesized across spatial noise, error level analysis, and frequency domain statistics.";
  }

  // Render Interactive Bounding Boxes on Canvas
  renderCanvasRegions(manipulation.regions || []);
}

// Render Evidence Ledger Card
function renderEvidenceCard(evidence) {
  const container = $("#evidence-content");
  if (!container || !evidence) return;

  container.innerHTML = `
    <div class="ledger-list">
      <div class="ledger-row">
        <span>SHA-256 INTEGRITY HASH</span>
        <b>${escapeHtml(evidence.sha256)}</b>
      </div>
      <div class="ledger-row">
        <span>PERCEPTUAL HASH (pHash)</span>
        <b>${escapeHtml(evidence.perceptual_hash || "Unavailable")}</b>
      </div>
      <div class="ledger-row">
        <span>DIMENSIONS &amp; CHANNELS</span>
        <b>${evidence.width} × ${evidence.height} px · ${evidence.channels || 3} channels (${escapeHtml(evidence.format)})</b>
      </div>
      <div class="ledger-row">
        <span>FILE SIZE</span>
        <b>${formatBytes(evidence.file_size)}</b>
      </div>
    </div>
  `;

  if (evidence.width && evidence.height) {
    originalImageDimensions = { width: evidence.width, height: evidence.height };
    $("#image-meta").textContent = `${evidence.width} × ${evidence.height} px · ${formatBytes(evidence.file_size)} · ${evidence.format || ""}`;
  }
}

// Render Explainable Findings Summary
function renderReportSummary(report) {
  const summaryBox = $("#report-content");
  if (!summaryBox || !report) return;

  activeReport = report.human_readable || JSON.stringify(report.report || report, null, 2);
  summaryBox.textContent = activeReport;
}

// Render Interactive Bounding Boxes on Main Canvas
function renderCanvasRegions(regions) {
  const overlay = $("#regions-overlay-layer");
  if (!overlay) return;
  overlay.innerHTML = "";

  if (!regions || !regions.length) return;

  const w = originalImageDimensions.width || 1000;
  const h = originalImageDimensions.height || 1000;

  regions.forEach((r, idx) => {
    const leftPct = (r.x / w) * 100;
    const topPct = (r.y / h) * 100;
    const widthPct = (r.width / w) * 100;
    const heightPct = (r.height / h) * 100;

    const box = document.createElement("div");
    box.className = "bounding-box-tag";
    box.style.left = `${leftPct}%`;
    box.style.top = `${topPct}%`;
    box.style.width = `${widthPct}%`;
    box.style.height = `${heightPct}%`;

    box.innerHTML = `<span class="box-score-tag">REGION #${idx + 1} (${formatPct(r.score)})</span>`;
    box.title = `Region ${idx + 1}: ${r.width}×${r.height} at (${r.x},${r.y}) · Score: ${formatPct(r.score)}`;

    box.addEventListener("click", () => {
      toast(`Region #${idx + 1}: ${r.width}×${r.height} px · Anomaly Score: ${formatPct(r.score)}`);
    });

    overlay.appendChild(box);
  });
}

// Switch Active Canvas Layer (Original, Heatmap, Noise, FFT, ELA, etc.)
function switchCanvasLayer(layerKey) {
  const overlayImg = $("#layer-overlay-image");
  const previewImg = $("#preview-image");
  if (!overlayImg) return;

  $$("#layer-selector .layer-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.layer === layerKey);
  });

  if (layerKey === "original") {
    overlayImg.classList.add("hidden");
    return;
  }

  // Find artifact path in bundle
  const bundle = window.__lastBundle;
  if (!bundle) return;

  let targetPath = null;
  const artifacts = bundle.artifacts || [];

  if (layerKey === "suspiciousness") {
    targetPath = artifacts.find((a) => a.type === "suspiciousness_heatmap")?.path || bundle.manipulation?.heatmap;
  } else if (layerKey === "noise") {
    targetPath = artifacts.find((a) => a.type === "noise_residual")?.path || bundle.camera?.evidence?.noise?.artifact_path;
  } else if (layerKey === "frequency") {
    targetPath = artifacts.find((a) => a.type === "frequency_spectrum")?.path || bundle.camera?.evidence?.frequency?.artifact_path;
  } else if (layerKey === "ela") {
    targetPath = artifacts.find((a) => a.type === "ela")?.path || bundle.manipulation?.evidence?.ela?.artifact_path;
  } else if (layerKey === "noise_anomaly") {
    targetPath = artifacts.find((a) => a.type === "noise_heatmap")?.path;
  }

  if (targetPath) {
    const url = resolveArtifactUrl(targetPath, bundle.analysisId);
    overlayImg.src = url;
    overlayImg.classList.remove("hidden");
    const opacity = ($("#layer-opacity-slider")?.value || 85) / 100;
    overlayImg.style.opacity = opacity;
  } else {
    toast(`Artifact map '${layerKey}' was not generated for this image.`);
    overlayImg.classList.add("hidden");
  }
}

// Render Full Artifacts Showcase Gallery (VIEW 4)
function renderArtifactsGallery(artifacts, bundle) {
  const grid = $("#artifacts-showcase-grid");
  const countBadge = $("#artifacts-badge-count");
  if (!grid) return;

  const analysisId = bundle.analysisId;
  const camera = bundle.camera || {};
  const manip = bundle.manipulation || {};
  const evidence = bundle.evidence || {};

  const items = [];

  // 1. Original Evidence Card
  const origUrl = `${originFromApi()}/artifacts/uploads/${analysisId}/original.jpg`;
  items.push(`
    <article class="artifact-card-full">
      <div class="artifact-img-wrap" onclick="openLightbox('${origUrl}', 'Original Evidence Record', 'Raw Ingested Image')">
        <img src="${origUrl}" alt="Original Evidence" onerror="this.src='${lastPreviewUrl}'" />
        <span class="artifact-overlay-badge">RAW EVIDENCE</span>
        <div class="artifact-zoom-cta">🔍 CLICK TO INSPECT</div>
      </div>
      <div class="artifact-meta-box">
        <h4 class="artifact-title">Original Evidence</h4>
        <span class="artifact-subtitle">Cryptographic Raw Image</span>
        <p class="artifact-desc">Unaltered master evidence preserved for chain of custody and pixel comparison.</p>
        <div class="artifact-stats-table">
          <div class="artifact-stat-row"><span>Dimensions</span><b>${evidence.width || "—"} × ${evidence.height || "—"} px</b></div>
          <div class="artifact-stat-row"><span>Format</span><b>${evidence.format || "—"}</b></div>
          <div class="artifact-stat-row"><span>File Size</span><b>${formatBytes(evidence.file_size)}</b></div>
        </div>
        <div class="artifact-action-row">
          <a class="btn btn-secondary btn-sm" href="${origUrl}" download="evidence_original.jpg" target="_blank">💾 DOWNLOAD</a>
          <button class="btn btn-secondary btn-sm" onclick="openLightbox('${origUrl}', 'Original Evidence', 'Raw Image')">🔍 FULLSCREEN</button>
        </div>
      </div>
    </article>
  `);

  // 2. Map all backend artifacts
  (artifacts || []).forEach((art) => {
    const type = art.type || art.name || "";
    const url = resolveArtifactUrl(art.path, analysisId);
    if (!url) return;

    const info = ARTIFACT_INFO[type] || {
      title: type.replaceAll("_", " ").toUpperCase(),
      subtitle: "Forensic Pipeline Map",
      desc: "Visual artifact map computed during algorithmic analysis.",
    };

    let statsHtml = "";
    if (type === "noise_residual") {
      const noiseStats = camera.evidence?.noise?.combined_statistics || {};
      statsHtml = `
        <div class="artifact-stats-table">
          <div class="artifact-stat-row"><span>Variance</span><b>${(noiseStats.variance || 0).toFixed(5)}</b></div>
          <div class="artifact-stat-row"><span>Kurtosis</span><b>${(noiseStats.kurtosis || 0).toFixed(2)}</b></div>
          <div class="artifact-stat-row"><span>Neighbor Corr</span><b>${(noiseStats.neighbor_correlation || 0).toFixed(3)}</b></div>
        </div>
      `;
    } else if (type === "frequency_spectrum") {
      const freqStats = camera.evidence?.frequency || {};
      statsHtml = `
        <div class="artifact-stats-table">
          <div class="artifact-stat-row"><span>Spectral Entropy</span><b>${(freqStats.spectral_entropy || 0).toFixed(3)}</b></div>
          <div class="artifact-stat-row"><span>Low Freq Energy</span><b>${(freqStats.low_frequency_energy || 0).toFixed(2)}</b></div>
          <div class="artifact-stat-row"><span>Radial Mean</span><b>${(freqStats.radial_profile_mean || 0).toFixed(2)}</b></div>
        </div>
      `;
    } else if (type === "ela") {
      const elaStats = manip.evidence?.ela?.statistics || {};
      statsHtml = `
        <div class="artifact-stats-table">
          <div class="artifact-stat-row"><span>ELA Mean</span><b>${(elaStats.ela_mean || 0).toFixed(2)}</b></div>
          <div class="artifact-stat-row"><span>ELA Max</span><b>${(elaStats.ela_max || 0).toFixed(1)}</b></div>
          <div class="artifact-stat-row"><span>High Error Ratio</span><b>${formatPct(elaStats.ela_high_ratio)}</b></div>
        </div>
      `;
    } else if (type === "suspiciousness_heatmap") {
      statsHtml = `
        <div class="artifact-stats-table">
          <div class="artifact-stat-row"><span>Overall Score</span><b>${formatPct(manip.overall_suspiciousness)}</b></div>
          <div class="artifact-stat-row"><span>Flagged Regions</span><b>${manip.regions?.length || 0} region(s)</b></div>
          <div class="artifact-stat-row"><span>Tampering Class</span><b>${(manip.types || ["Authentic"])[0]}</b></div>
        </div>
      `;
    } else if (type === "noise_heatmap") {
      statsHtml = `
        <div class="artifact-stats-table">
          <div class="artifact-stat-row"><span>Local Anomaly</span><b>${(manip.local_noise_anomaly_score || 0).toFixed(3)}</b></div>
          <div class="artifact-stat-row"><span>Grid Analysis</span><b>8x8 Patches</b></div>
        </div>
      `;
    }

    items.push(`
      <article class="artifact-card-full">
        <div class="artifact-img-wrap" onclick="openLightbox('${url}', '${escapeHtml(info.title)}', '${escapeHtml(info.subtitle)}')">
          <img src="${url}" alt="${escapeHtml(info.title)}" />
          <span class="artifact-overlay-badge">${escapeHtml(type.toUpperCase())}</span>
          <div class="artifact-zoom-cta">🔍 CLICK TO INSPECT</div>
        </div>
        <div class="artifact-meta-box">
          <h4 class="artifact-title">${escapeHtml(info.title)}</h4>
          <span class="artifact-subtitle">${escapeHtml(info.subtitle)}</span>
          <p class="artifact-desc">${escapeHtml(info.desc)}</p>
          ${statsHtml}
          <div class="artifact-action-row">
            <a class="btn btn-secondary btn-sm" href="${url}" download="${type}.png" target="_blank">💾 DOWNLOAD</a>
            <button class="btn btn-secondary btn-sm" onclick="openLightbox('${url}', '${escapeHtml(info.title)}', '${escapeHtml(info.subtitle)}')">🔍 FULLSCREEN</button>
          </div>
        </div>
      </article>
    `);
  });

  if (countBadge) countBadge.textContent = `${items.length} ARTIFACTS`;
  grid.innerHTML = items.join("") || `<div class="empty-state-text">No artifact images available for this analysis.</div>`;

  // Also update mini artifacts grid in Dossier Overview tab
  const miniGrid = $("#artifact-grid");
  const miniCount = $("#artifact-count");
  if (miniGrid) {
    const miniItems = (artifacts || []).map((art) => {
      const type = art.type || art.name || "";
      const url = resolveArtifactUrl(art.path, analysisId);
      if (!url) return "";
      const info = ARTIFACT_INFO[type] || { title: type.replaceAll("_", " ") };
      return `
        <div class="artifact-mini-card" onclick="openLightbox('${url}', '${escapeHtml(info.title)}', '${escapeHtml(type)}')">
          <div class="artifact-mini-img"><img src="${url}" alt="${escapeHtml(info.title)}" /></div>
          <div class="artifact-mini-body">
            <div class="artifact-mini-title">${escapeHtml(info.title)}</div>
            <div class="artifact-mini-sub">${escapeHtml(type.toUpperCase())}</div>
          </div>
        </div>
      `;
    });
    miniGrid.innerHTML = miniItems.join("");
    if (miniCount) miniCount.textContent = `${miniItems.length} SIGNALS`;
  }
}

// Render Full Tabbed Dossier (VIEW 5)
function renderFullDossier(bundle) {
  const { analysisId, evidence, camera, manipulation, robustness, report, artifacts, result } = bundle;
  const consistency = result?.consistency || {};
  const fusion = result?.evidence_fusion || {};
  const metadata = evidence?.metadata || result?.metadata || {};

  $("#final-id").textContent = analysisId.toUpperCase();

  const manipScore = Math.round((manipulation?.overall_suspiciousness || 0) * 100);
  const camConf = formatPct(camera?.confidence);
  const assessment = (consistency.overall_assessment || "CONSISTENT").toUpperCase();

  // OVERVIEW TAB
  $("#overview-summary").innerHTML = `
    <div class="overview-kpi-card">
      <small>CAMERA ATTRIBUTION</small>
      <strong>${camConf}</strong>
      <span>${escapeHtml(camera?.model || camera?.status || "Unknown Camera")}</span>
    </div>
    <div class="overview-kpi-card">
      <small>MANIPULATION PROBABILITY</small>
      <strong class="${manipScore > 50 ? 'text-accent' : 'text-success'}">${manipScore}%</strong>
      <span>${escapeHtml(manipulation?.status || "Consistent")}</span>
    </div>
    <div class="overview-kpi-card">
      <small>OVERALL ASSESSMENT</small>
      <strong class="text-info">${escapeHtml(assessment)}</strong>
      <span>${fusion.modules_available || 4}/${fusion.modules_total || 4} Forensic modules contributed</span>
    </div>
  `;

  const flags = consistency.consistency_flags || [];
  $("#overview-flags").innerHTML = flags.length
    ? flags.map((f) => `<span class="chip-tag">${escapeHtml(String(f).replaceAll("_", " "))}</span>`).join("")
    : `<span class="badge-success">NO CONFLICTING FORENSIC FLAGS</span>`;

  // CAMERA ATTRIBUTION TAB
  const topCams = (camera?.top_candidates || [])
    .map(
      (c) => `
      <div class="data-row">
        <small>Rank #${c.rank ?? 1}</small>
        <strong>${escapeHtml(c.camera_model || c.label)}</strong>
        <b>${formatPct(c.confidence)}</b>
      </div>`
    )
    .join("") || `<div class="empty-state-text">No ranked candidates found</div>`;

  const cfa = camera?.evidence?.cfa || {};
  const cfaFeatures = cfa.demosaicing_related_features || {};
  const freq = camera?.evidence?.frequency || {};
  const noise = camera?.evidence?.noise?.combined_statistics || {};
  const prnu = camera?.evidence?.prnu || {};

  $("#tab-camera").innerHTML = `
    <div class="findings-grid-2">
      <div class="sub-panel">
        <h3>Camera Attribution Output</h3>
        <div class="data-row"><small>MODEL PREDICTION</small><strong>${escapeHtml(camera?.model || "Unknown")}</strong></div>
        <div class="data-row"><small>MANUFACTURER</small><strong>${escapeHtml(camera?.manufacturer || "—")}</strong></div>
        <div class="data-row"><small>CONFIDENCE SCORE</small><b>${formatPct(camera?.confidence)}</b></div>
        <div class="data-row"><small>PREDICTION SOURCE</small><strong>${escapeHtml(camera?.prediction_source || "FORENSIC_ML")}</strong></div>
        <div class="data-row"><small>KNOWN CAMERA REGISTRY</small><strong>${camera?.known_camera ? "YES (Verified)" : "NO (Open-Set)"}</strong></div>
      </div>
      <div class="sub-panel">
        <h3>Top Ranked Device Matches</h3>
        ${topCams}
      </div>
    </div>

    <div class="findings-grid-2">
      <div class="sub-panel">
        <h3>CFA &amp; Demosaicing Signals</h3>
        <div class="data-row"><small>STATUS</small><strong>${escapeHtml(cfa.status || "—")}</strong></div>
        <div class="data-row"><small>2x2 Periodicity Energy</small><b>${(cfaFeatures.periodic_2x2?.periodic_energy || 0).toFixed(4)}</b></div>
        <div class="data-row"><small>Directional Correlation H</small><b>${(cfaFeatures.directional_correlation?.dir_h || 0).toFixed(4)}</b></div>
        <div class="data-row"><small>Global Channel Corr (RG)</small><b>${(cfaFeatures.global_channel_correlation?.rg || 0).toFixed(4)}</b></div>
      </div>
      <div class="sub-panel">
        <h3>Frequency &amp; PRNU Fingerprint</h3>
        <div class="data-row"><small>PRNU Sensor Match</small><strong>${escapeHtml(prnu.status || "Reference unavailable")}</strong></div>
        <div class="data-row"><small>Spectral Entropy</small><b>${(freq.spectral_entropy || 0).toFixed(3)}</b></div>
        <div class="data-row"><small>Low Frequency Energy</small><b>${(freq.low_frequency_energy || 0).toFixed(2)}</b></div>
        <div class="data-row"><small>Noise Std Dev</small><b>${(noise.std || 0).toFixed(5)}</b></div>
      </div>
    </div>
  `;

  // MANIPULATION TAB
  const regionsHtml = (manipulation?.regions || [])
    .map(
      (r, idx) => `
      <div class="region-pill">
        <small>REGION #${idx + 1} (${r.width}×${r.height} at x:${r.x}, y:${r.y})</small>
        <strong>ANOMALY: ${formatPct(r.score)}</strong>
      </div>`
    )
    .join("") || `<div class="empty-state-text">No suspicious localized tampering regions flagged.</div>`;

  const copyMove = manipulation?.evidence?.copy_move || {};
  const splice = manipulation?.evidence?.splice || {};
  const jpeg = manipulation?.evidence?.jpeg || {};
  const aiGen = manipulation?.ai_generated || {};

  $("#tab-manipulation").innerHTML = `
    <div class="findings-grid-2">
      <div class="sub-panel">
        <h3>Tampering Localization Summary</h3>
        <div class="data-row"><small>OVERALL SUSPICIOUSNESS</small><b class="text-accent">${formatPct(manipulation?.overall_suspiciousness)}</b></div>
        <div class="data-row"><small>CLASSIFIED TYPES</small><strong>${escapeHtml((manipulation?.types || []).join(", ") || "None")}</strong></div>
        <div class="data-row"><small>SPLICING PROBABILITY</small><b>${formatPct(manipulation?.tampered_probability || splice.tampered_probability)}</b></div>
        <div class="data-row"><small>COPY-MOVE DETECTION</small><strong>${manipulation?.copy_move_detected ? "DETECTED (ORB Keypoints Match)" : "None Detected"}</strong></div>
        <div class="data-row"><small>RECOMPRESSION</small><strong>${manipulation?.possible_recompression ? "Possible Double Recompression" : "Single Compression"}</strong></div>
      </div>
      <div class="sub-panel">
        <h3>AI Generator Detector</h3>
        <div class="data-row"><small>AI DETECTOR STATUS</small><strong>${escapeHtml(aiGen.status || manipulation?.learned_detector_status || "model_ready")}</strong></div>
        <div class="data-row"><small>IS AI GENERATED</small><strong>${aiGen.is_ai_generated ? "YES (Synthetic Signal)" : "NO / Inconclusive"}</strong></div>
        <div class="data-row"><small>SYNTHETIC PROBABILITY</small><b>${formatPct(aiGen.ai_probability)}</b></div>
        <div class="data-row"><small>GENERATOR MODEL TYPE</small><strong>${escapeHtml(aiGen.model_type || "N/A")}</strong></div>
      </div>
    </div>

    <div class="sub-panel">
      <h3>Flagged Bounding Regions (${(manipulation?.regions || []).length} localized areas)</h3>
      <div class="regions-table-wrap">${regionsHtml}</div>
    </div>
  `;

  // METADATA TAB
  const metaRows = Object.entries(metadata)
    .filter(([, val]) => val !== null && typeof val !== "object")
    .map(([key, val]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(val)}</td></tr>`)
    .join("");

  $("#tab-metadata").innerHTML = `
    <div class="sub-panel">
      <h3>Extracted Container &amp; EXIF Metadata</h3>
      <table class="meta-table-box">
        <tbody>
          ${metaRows || "<tr><td>No scalar metadata tags found in file.</td></tr>"}
        </tbody>
      </table>
    </div>
    <div class="sub-panel">
      <h3>Raw Metadata JSON Structure</h3>
      <pre class="report-pre-block">${escapeHtml(JSON.stringify(metadata, null, 2))}</pre>
    </div>
  `;

  // ROBUSTNESS TAB
  const rob = robustness || result?.robustness || { enabled: false, results: [], robustness_summary: {} };
  const summary = rob.robustness_summary || {};
  const transformsHtml = (rob.results || [])
    .map(
      (r) => `
      <div class="data-row">
        <small>${escapeHtml(r.transformation)}</small>
        <strong>${escapeHtml(r.prediction || "—")} (${formatPct(r.confidence)})</strong>
        <b class="${r.prediction_changed ? 'text-accent' : 'text-success'}">${r.prediction_changed ? "CHANGED" : "STABLE"}</b>
      </div>`
    )
    .join("") || `<div class="empty-state-text">Robustness perturbation results unavailable.</div>`;

  $("#tab-robustness").innerHTML = `
    <div class="findings-grid-2">
      <div class="sub-panel">
        <h3>Perturbation Stability Score</h3>
        <div class="data-row"><small>ROBUSTNESS SCORE</small><b class="text-info">${formatPct(summary.robustness_score)}</b></div>
        <div class="data-row"><small>TESTS EXECUTED</small><strong>${summary.transformations_tested || 0} transformations</strong></div>
        <div class="data-row"><small>PREDICTIONS CHANGED</small><strong>${summary.predictions_changed || 0}</strong></div>
        <div class="data-row"><small>BASELINE PREDICTION</small><strong>${escapeHtml(summary.baseline_prediction || "—")}</strong></div>
      </div>
      <div class="sub-panel">
        <h3>Transformation Stress Tests</h3>
        ${transformsHtml}
      </div>
    </div>
  `;

  // CONSISTENCY & FUSION TAB
  $("#tab-consistency").innerHTML = `
    <div class="findings-grid-2">
      <div class="sub-panel">
        <h3>Evidence Consistency Engine</h3>
        <div class="data-row"><small>OVERALL ASSESSMENT</small><b class="text-info">${escapeHtml(consistency.overall_assessment || "inconclusive")}</b></div>
        <div class="data-row"><small>CONSISTENCY SCORE</small><strong>${formatPct(consistency.consistency_score)}</strong></div>
        <div class="data-row"><small>CONSISTENT SIGNALS</small><strong>${consistency.consistent ? "YES" : "NO / Conflicts Present"}</strong></div>
      </div>
      <div class="sub-panel">
        <h3>Multi-Module Evidence Fusion</h3>
        <div class="data-row"><small>FUSED VERDICT</small><strong>${escapeHtml(fusion.fused_verdict || "READY")}</strong></div>
        <div class="data-row"><small>FUSED CONFIDENCE</small><b>${formatPct(fusion.fused_confidence)}</b></div>
        <div class="data-row"><small>MODULES CONTRIBUTING</small><strong>${fusion.modules_available || 4} of ${fusion.modules_total || 4} available</strong></div>
      </div>
    </div>
    <div class="sub-panel">
      <h3>Reasoning &amp; Conflicts</h3>
      <pre class="report-pre-block">${escapeHtml(JSON.stringify(consistency.module_summaries || consistency, null, 2))}</pre>
    </div>
  `;

  // REPORT TAB
  $("#tab-report").textContent = report?.human_readable || JSON.stringify(report?.report || result?.report || {}, null, 2);

  // Render gallery
  renderArtifactsGallery(artifacts, bundle);
}

// Load Full Analysis Results Once Completed
async function loadAnalysisResults(id) {
  const endpoints = ["evidence", "camera", "manipulation", "robustness", "report", "artifacts"];
  const settled = await Promise.allSettled(endpoints.map((name) => call(`/analysis/${id}/${name}`)));
  const data = Object.fromEntries(endpoints.map((key, i) => [key, settled[i].status === "fulfilled" ? settled[i].value : null]));

  let full = null;
  try {
    full = await call(`/analysis/${id}`);
  } catch {
    full = null;
  }

  renderEvidenceCard(data.evidence);
  renderCameraCard(data.camera || full?.result?.camera);
  renderManipulationCard(data.manipulation || full?.result?.manipulation);
  renderReportSummary(data.report);

  currentAnalysisId = id;
  currentArtifactsList = data.artifacts?.artifacts || [];

  window.__lastBundle = {
    analysisId: id,
    evidence: data.evidence,
    camera: data.camera || full?.result?.camera,
    manipulation: data.manipulation || full?.result?.manipulation,
    robustness: data.robustness || full?.result?.robustness,
    report: data.report,
    artifacts: currentArtifactsList,
    result: full?.result || {},
  };

  renderArtifactsGallery(currentArtifactsList, window.__lastBundle);
}

// Polling Controller for Running Analysis
function pollAnalysisStatus(id) {
  if (pollTimer) clearInterval(pollTimer);
  let attempts = 0;

  pollTimer = setInterval(async () => {
    try {
      const job = await call(`/analysis/${id}/status`);
      setProgress(job.status, job.progress);

      if (job.status === "COMPLETED") {
        clearInterval(pollTimer);
        pollTimer = null;
        setProgress("COMPLETED", 100);
        await loadAnalysisResults(id);
        toast("Forensic examination completed successfully!");
      } else if (job.status === "FAILED") {
        clearInterval(pollTimer);
        pollTimer = null;
        toast(`Analysis failed: ${job.error || "Unknown error"}`);
        $("#verdict-title").textContent = "Analysis Failed";
        $("#verdict-desc").textContent = job.error || "The pipeline encountered an error during execution.";
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      toast(err.message);
    }

    if (++attempts > 180) {
      clearInterval(pollTimer);
      pollTimer = null;
      toast("Analysis is taking longer than expected; check status in Cases.");
    }
  }, 1200);
}

// Upload & Execute Single Image Pipeline
async function handleImageUpload(file) {
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) {
    return toast("Please select an image smaller than 25 MB.");
  }

  try {
    const formData = new FormData();
    formData.append("file", file);

    const uploadInfo = await call("/analysis/upload", {
      method: "POST",
      body: formData,
    });

    initWorkspace(file, uploadInfo);
    setProgress("QUEUED", 5);

    // Trigger async execution
    await call(`/analysis/${uploadInfo.analysis_id}/run`, { method: "POST" });
    pollAnalysisStatus(uploadInfo.analysis_id);
  } catch (err) {
    toast(`Upload failed: ${err.message}`);
  }
}

// Upload Sample Preset
async function uploadSampleEvidence(samplePath, sampleName) {
  try {
    toast(`Loading sample evidence: ${sampleName}…`);
    const res = await fetch(samplePath);
    const blob = await res.blob();
    const file = new File([blob], sampleName, { type: blob.type || "image/jpeg" });
    await handleImageUpload(file);
  } catch (err) {
    toast(`Failed to load sample: ${err.message}`);
  }
}

// Lightbox Modal Controller
function openLightbox(imgUrl, title, subtitle) {
  const modal = $("#artifact-lightbox");
  const img = $("#lightbox-img");
  const titleEl = $("#lightbox-title");
  const subEl = $("#lightbox-subtitle");
  const downloadLink = $("#lightbox-download");

  if (!modal || !img) return;

  img.src = imgUrl;
  if (titleEl) titleEl.textContent = title || "Artifact Deep Inspection";
  if (subEl) subEl.textContent = subtitle || "Forensic Signal Map";
  if (downloadLink) {
    downloadLink.href = imgUrl;
    downloadLink.download = `${(title || "artifact").toLowerCase().replaceAll(" ", "_")}.png`;
  }

  modal.classList.remove("hidden");
}

function closeLightbox() {
  const modal = $("#artifact-lightbox");
  if (modal) modal.classList.add("hidden");
}

// Cases Manager Controller
async function refreshCasesList() {
  const container = $("#case-list");
  if (!container) return;

  try {
    const cases = await call("/cases");
    if (!cases || !cases.length) {
      container.innerHTML = `<div class="empty-state-text">No forensic cases recorded yet. Create one above.</div>`;
      return;
    }

    container.innerHTML = cases
      .map(
        (c) => `
        <article class="case-item-card" data-id="${escapeHtml(c.case_id)}">
          <div class="case-item-top">
            <small>${escapeHtml(c.created_at || "")}</small>
            <code>${escapeHtml(c.case_id)}</code>
          </div>
          <h4>${escapeHtml(c.title)}</h4>
          <p>${escapeHtml(c.description || "No agency notes attached.")}</p>
        </article>`
      )
      .join("");
  } catch (err) {
    container.innerHTML = `<div class="empty-state-text">${escapeHtml(err.message)}</div>`;
  }
}

// ==========================================================================
// EVENT LISTENERS & INITIALIZATION
// ==========================================================================

// Navigation
$("#nav-brand")?.addEventListener("click", (e) => {
  e.preventDefault();
  showView("intake");
});

$$("#main-nav .nav-btn").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const view = btn.dataset.view;
    if (view === "workspace" && !currentAnalysisId) {
      return toast("Please upload evidence first to view the active pipeline.");
    }
    if (view === "artifacts" && !window.__lastBundle) {
      return toast("Run an analysis first to view generated artifacts.");
    }
    if (view === "final" && !window.__lastBundle) {
      return toast("Run an analysis first to view the forensic dossier.");
    }
    if (view === "cases") {
      showView("cases");
      await refreshCasesList();
      return;
    }
    showView(view);
  });
});

// Dropzone & File Input
const fileInput = $("#file-input");
if (fileInput) {
  fileInput.addEventListener("change", (e) => handleImageUpload(e.target.files[0]));
}

const dropzone = $("#dropzone");
if (dropzone) {
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragging");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragging");
    })
  );
  dropzone.addEventListener("drop", (e) => handleImageUpload(e.dataTransfer.files[0]));
}

$("#enter-lab-cta")?.addEventListener("click", (e) => {
  e.preventDefault();
  e.stopPropagation();
  showView("lab");
});

// Sample Evidence Buttons
$("#try-sample-desk")?.addEventListener("click", () =>
  uploadSampleEvidence("/static/assets/forensic-desk.jpg", "sample-desk.jpg")
);
$("#try-sample-photo")?.addEventListener("click", () =>
  uploadSampleEvidence("/static/assets/detective-photo.jpg", "sample-scene-photo.jpg")
);
$("#try-sample-cyber")?.addEventListener("click", () =>
  uploadSampleEvidence("/static/assets/cyber-alert.jpg", "sample-cyber-alert.jpg")
);

// Lab Mode Switching
$("#mode-single")?.addEventListener("click", () => {
  $("#mode-single").classList.add("active");
  $("#mode-compare").classList.remove("active");
  $("#single-analysis-panel").classList.remove("hidden");
  $$(".compare-only").forEach((el) => el.classList.add("hidden"));
});

$("#mode-compare")?.addEventListener("click", () => {
  $("#mode-compare").classList.add("active");
  $("#mode-single").classList.remove("active");
  $("#single-analysis-panel").classList.add("hidden");
  $$(".compare-only").forEach((el) => el.classList.remove("hidden"));
});

$("#back-to-intake")?.addEventListener("click", () => showView("intake"));

// Single Lab Upload
$("#single-file")?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  $("#single-file-name").textContent = file ? file.name : "Click or Drag Image Here";
});

$("#start-single-analysis")?.addEventListener("click", () => {
  const file = $("#single-file")?.files[0];
  if (!file) return toast("Please select an image file first.");
  handleImageUpload(file);
});

// Comparison Lab Upload & Run
$("#original-file")?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  $("#original-name").textContent = file ? file.name : "Choose Original Image";
});

$("#edited-file")?.addEventListener("change", (e) => {
  const file = e.target.files[0];
  $("#edited-name").textContent = file ? file.name : "Choose Edited Derivative";
});

$("#compare-images")?.addEventListener("click", async () => {
  const orig = $("#original-file")?.files[0];
  const edited = $("#edited-file")?.files[0];

  if (!orig || !edited) {
    return toast("Please select both original and edited images.");
  }

  const btn = $("#compare-images");
  btn.disabled = true;
  btn.textContent = "ALIGNING & COMPARING EVIDENCE…";

  try {
    const form = new FormData();
    form.append("original", orig);
    form.append("edited", edited);

    const result = await call("/compare", { method: "POST", body: form });

    $("#compare-id-tag").textContent = `COMPARISON ID: ${result.comparison_id}`;
    $("#compare-result-line").textContent = result.changed_regions?.length
      ? `${result.changed_regions.length} modified region(s) localized.`
      : "No material visual difference detected.";

    $("#metric-area").textContent = `${result.changed_area_percent}%`;
    $("#metric-align").textContent = `${((result.alignment_score || 0) * 100).toFixed(1)}%`;
    $("#metric-mods").textContent = (result.modifications || []).join(" · ").replaceAll("_", " ") || "Authentic";
    $("#compare-limits").textContent = (result.limitations || []).join(" ");

    const diffUrl = resolveArtifactUrl(result.artifacts?.difference_heatmap);
    const overlayUrl = resolveArtifactUrl(result.artifacts?.change_overlay);

    const diffImg = $("#compare-diff-img");
    const overlayImg = $("#edited-preview");

    if (diffImg && diffUrl) diffImg.src = diffUrl;
    if (overlayImg && overlayUrl) overlayImg.src = overlayUrl;

    // Render comparison regions
    const regContainer = $("#compare-regions-list");
    if (regContainer) {
      regContainer.innerHTML = (result.changed_regions || [])
        .map(
          (r, idx) => `
          <div class="region-pill">
            <small>REGION #${idx + 1} (${r.width}×${r.height} at ${r.x},${r.y})</small>
            <strong>AREA: ${r.area_percent}%</strong>
          </div>`
        )
        .join("") || `<div class="empty-state-text">No distinct bounding clusters.</div>`;
    }

    $("#compare-results").classList.remove("hidden");
    toast("Comparison completed successfully.");
  } catch (err) {
    toast(`Comparison error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "COMPARE EVIDENCE VIA EUCLIDEAN ALIGNMENT <b>→</b>";
  }
});

// Workspace Buttons
$("#new-case")?.addEventListener("click", () => {
  showView("intake");
  if (fileInput) fileInput.value = "";
});

$("#copy-case-id")?.addEventListener("click", async () => {
  if (!currentAnalysisId) return;
  await navigator.clipboard.writeText(currentAnalysisId);
  toast("Analysis ID copied to clipboard.");
});

$("#open-full-report")?.addEventListener("click", () => {
  if (!window.__lastBundle) return toast("Please wait for the analysis to complete.");
  renderFullDossier(window.__lastBundle);
  showView("final");
});

$("#copy-report")?.addEventListener("click", async () => {
  if (!activeReport) return;
  await navigator.clipboard.writeText(activeReport);
  toast("Findings copied to clipboard.");
});

// Layer Switcher Buttons
$$("#layer-selector .layer-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchCanvasLayer(btn.dataset.layer));
});

// Layer Opacity Slider
$("#layer-opacity-slider")?.addEventListener("input", (e) => {
  const val = e.target.value;
  $("#opacity-val").textContent = `${val}%`;
  const overlay = $("#layer-overlay-image");
  if (overlay) overlay.style.opacity = val / 100;
});

// Toggle Bounding Box Highlight
$("#toggle-regions-check")?.addEventListener("change", (e) => {
  const overlay = $("#regions-overlay-layer");
  if (overlay) overlay.style.display = e.target.checked ? "block" : "none";
});

// Fullscreen Zoom of Main Stage
$("#zoom-main-image-btn")?.addEventListener("click", () => {
  const preview = $("#preview-image");
  if (preview?.src) openLightbox(preview.src, "Original Ingested Evidence", "Main Stage View");
});

// Dossier Tabs Navigation
$("#analysis-tabs")?.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-tab]");
  if (!btn) return;

  $$("#analysis-tabs .report-tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.panel !== btn.dataset.tab));
});

// Dossier Actions
$("#final-new-case")?.addEventListener("click", () => showView("intake"));
$("#download-report-txt-btn")?.addEventListener("click", () => {
  if (!activeReport) return toast("Report not available.");
  const blob = new Blob([activeReport], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `cameratrace_report_${currentAnalysisId || "case"}.txt`;
  a.click();
});

$("#download-report-json-btn")?.addEventListener("click", () => {
  if (!window.__lastBundle) return toast("Dossier data not available.");
  const blob = new Blob([JSON.stringify(window.__lastBundle, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `cameratrace_dossier_${currentAnalysisId || "case"}.json`;
  a.click();
});

$("#copy-tab-report")?.addEventListener("click", async () => {
  if (!activeReport) return;
  await navigator.clipboard.writeText(activeReport);
  toast("Dossier copied to clipboard.");
});

// Lightbox Close
$("#lightbox-close")?.addEventListener("click", closeLightbox);
$("#lightbox-backdrop")?.addEventListener("click", closeLightbox);

// Settings Dialog
$("#settings-button")?.addEventListener("click", () => {
  const input = $("#api-url");
  if (input) input.value = apiBase;
  $("#settings-dialog")?.showModal();
});

$("#save-settings")?.addEventListener("click", () => {
  const val = $("#api-url")?.value.trim().replace(/\/$/, "");
  apiBase = val || "/api/v1";
  localStorage.setItem("cameratrace-api", apiBase);
  checkHealth();
  toast("API endpoint configuration saved.");
});

// Cases Form
$("#case-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const title = $("#case-title-input")?.value.trim();
  const desc = $("#case-desc-input")?.value.trim() || null;

  try {
    await call("/cases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, description: desc }),
    });
    $("#case-form").reset();
    toast("Forensic case created successfully.");
    await refreshCasesList();
  } catch (err) {
    toast(`Failed to create case: ${err.message}`);
  }
});

$("#cases-back")?.addEventListener("click", () => showView("intake"));
$("#refresh-cases-btn")?.addEventListener("click", refreshCasesList);

// Reopen Analysis by ID
$("#lookup-open")?.addEventListener("click", async () => {
  const id = $("#lookup-id")?.value.trim();
  if (!id) return toast("Please enter an analysis_id.");

  try {
    toast(`Loading analysis ${id}…`);
    const status = await call(`/analysis/${id}/status`);

    initWorkspace(null, {
      analysis_id: id,
      filename: "Loaded Investigation",
      format: "—",
      width: 0,
      height: 0,
      file_size: 0,
      sha256: "—",
    });

    setProgress(status.status, status.progress);

    if (status.status === "COMPLETED") {
      await loadAnalysisResults(id);
      renderFullDossier(window.__lastBundle);
      showView("final");
      toast("Dossier loaded.");
    } else {
      pollAnalysisStatus(id);
    }
  } catch (err) {
    toast(`Lookup error: ${err.message}`);
  }
});

// Camera Registry Search
$("#camera-search-btn")?.addEventListener("click", async () => {
  const q = encodeURIComponent($("#camera-query")?.value.trim() || "");
  const out = $("#camera-search-out");
  try {
    const data = await call(`/cameras/search?q=${q}&top_k=8`);
    if (out) out.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    if (out) out.textContent = `Search error: ${err.message}`;
  }
});

// Click on Case Item Card
$("#case-list")?.addEventListener("click", async (e) => {
  const card = e.target.closest(".case-item-card");
  if (!card) return;

  try {
    const detail = await call(`/cases/${card.dataset.id}`);
    toast(`Case ${detail.title}: ${(detail.analyses || []).length} linked analyses.`);
    const out = $("#camera-search-out");
    if (out) out.textContent = JSON.stringify(detail, null, 2);
  } catch (err) {
    toast(err.message);
  }
});

// Initial Startup
checkHealth();
