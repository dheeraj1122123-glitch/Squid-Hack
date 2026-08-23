# CameraTrace — Technology Stack and System Flow

## 1. Project purpose

CameraTrace is a digital-image forensics web application. A user can upload one image for forensic inspection or submit an original and an edited derivative for comparison. The system presents evidence rather than claiming certainty: metadata, camera-source signals, manipulation indicators, visual artifact maps, comparison regions, and a readable forensic report.

## 2. Technology stack

| Layer | Technology | Responsibility |
|---|---|---|
| Web application | FastAPI | Serves the API, static frontend, validation errors, and API documentation. |
| Python runtime | Python 3 | Executes services, forensic modules, training scripts, and tests. |
| API validation | Pydantic | Defines request and response schemas. |
| Persistence | SQLAlchemy + SQLite | Stores analyses, evidence records, predictions, manipulation results, and reports. |
| Image decoding | Pillow + OpenCV | Reads uploaded images, validates integrity, normalizes imagery, and generates visual artifacts. |
| Metadata | Pillow EXIF parsing + XMP/container parsing | Extracts EXIF, XMP, and supported container metadata. |
| Image hashing | ImageHash | Builds a perceptual hash for evidence identification. |
| Numerical processing | NumPy, SciPy, scikit-image | Supports feature extraction and statistical image analysis. |
| Machine learning | scikit-learn, XGBoost, joblib | Loads or trains camera and manipulation classifiers. |
| Optional deep learning | PyTorch and torchvision | Supports model-based camera and AI-generated-image analysis. |
| Charts / saved artifact rendering | Matplotlib + Seaborn | Produces visual forensic maps where required. |
| Frontend | HTML, CSS, vanilla JavaScript | Provides the responsive forensic interface with no frontend framework dependency. |
| Styling | CSS custom properties, CSS 3D transforms, keyframes | Supplies the peach/red forensic palette, 3D evidence scene, scanner effects, and accessibility-aware motion. |
| Server | Uvicorn | Runs the FastAPI ASGI application locally or in deployment. |
| Packaging / deployment | Docker + docker-compose files | Provides a container-ready backend setup. |

## 3. Main folders

```text
backend/
├── app/
│   ├── api/routes/         # FastAPI endpoints
│   ├── core/               # configuration, security, logging, utilities
│   ├── db/                 # SQLAlchemy models, database setup, repositories
│   ├── forensic/           # evidence-analysis algorithms
│   ├── schemas/            # Pydantic models
│   ├── services/           # orchestration for each workflow
│   └── static/             # frontend HTML, CSS, JavaScript and visual assets
├── datasets/               # local training/downloaded datasets; not for GitHub commits
├── models/                 # local trained models; not for GitHub commits
├── artifacts/              # generated analysis maps, uploads and reports; runtime-only
├── training/               # preparation, training, splitting and evaluation scripts
├── tests/                  # automated tests
├── run.py                  # local application entry point
└── requirements.txt        # Python dependencies
```

## 4. Frontend flow

```text
User opens /
    ↓
Static CameraTrace interface loads
    ↓
User chooses one of two paths
    ├── Single image analysis
    │     ↓
    │  POST /api/v1/analysis/upload
    │     ↓
    │  POST /api/v1/analysis/{analysis_id}/run
    │     ↓
    │  Frontend polls GET /api/v1/analysis/{analysis_id}/status
    │     ↓
    │  UI requests evidence, camera, manipulation, robustness and report results
    │
    └── Original versus edited comparison
          ↓
       POST /api/v1/compare
          ↓
       UI displays changed-area percentage, detected regions, heatmap and overlay
```

The frontend is intentionally framework-free. `index.html` contains the UI structure, `styles.css` contains the palette, responsive rules, typography and 3D presentation effects, and `app.js` communicates with the API using `fetch`.

## 5. Single-image forensic-analysis flow

```text
Image upload
    ↓
Evidence intake service
    ├── validates extension, MIME type and maximum size
    ├── saves an immutable original upload in runtime artifacts
    ├── verifies that the image can be decoded
    ├── calculates SHA-256 and perceptual hash
    ├── reads dimensions, format and channels
    └── extracts embedded metadata
    ↓
Analysis record created in SQLite
    ↓
Analysis service runs stages
    ├── Metadata analysis
    ├── Camera forensics
    ├── Manipulation analysis
    ├── Consistency analysis and evidence fusion
    ├── Robustness analysis
    └── Report generation
    ↓
JSON result, report and visual artifacts returned to the frontend
```

## 6. Metadata and source-camera flow

1. The metadata extractor checks EXIF fields such as Make, Model, software, ISO, exposure, focal length, date/time, orientation, and GPS availability.
2. It also checks supported XMP packets and image-container text metadata. Each field is labelled by source (`EXIF`, `XMP`, or `CONTAINER`).
3. If embedded camera make/model is available, the result is explicitly marked as embedded metadata evidence.
4. If metadata does not identify the camera, the camera service uses image signals:
   - noise statistics
   - CFA/demosaicing features
   - frequency features
   - JPEG compression features
   - PRNU-related features
5. The camera classifier produces a candidate source-camera prediction and confidence when an appropriate model is available.
6. The consistency engine checks whether metadata and inferred camera signals disagree.

## 7. Manipulation-analysis flow

The manipulation service combines multiple signals rather than relying on a single test.

| Signal | Purpose |
|---|---|
| Error Level Analysis (ELA) | Detects unusual JPEG recompression patterns. |
| Local noise inconsistency | Locates regions with unusual residual-noise characteristics. |
| Copy-move detection | Searches for repeated areas that may indicate duplicated content. |
| Splicing detection | Estimates whether regions may originate from different imagery. |
| JPEG analysis | Checks compression and recompression indicators. |
| AI-generated-image detector | Produces an additional AI-generation signal where configured. |
| Localization fusion | Combines spatial maps into suspicious regions and a heatmap. |

The result records indicators, probable categories, regions, confidence-related values, limitations, and paths to generated artifacts. These are forensic indicators, not definitive proof of editing.

## 8. Original-versus-edited comparison flow

1. The comparison endpoint validates both files.
2. Both images are decoded in memory through OpenCV.
3. The edited image is normalized to the original image coordinate space.
4. ECC alignment attempts to compensate for small rotation and translation differences.
5. Absolute pixel difference is calculated and smoothed to reduce low-level compression noise.
6. Thresholding and morphology isolate meaningful changed areas.
7. Contours are converted into changed-region boxes and area percentages.
8. The service creates two output visuals:
   - difference heatmap
   - changed-area overlay on the original
9. The frontend shows dimensions, alignment score, changed-area percentage, likely change categories, and region count.

## 9. Evidence fusion, robustness, and reporting

### Consistency and fusion

The consistency engine reviews evidence from metadata, camera analysis, manipulation analysis, and PRNU-related output. It raises explainable flags such as a metadata/signal mismatch. Evidence fusion keeps module output separate so that a user can see the reason behind a conclusion.

### Robustness

The robustness service evaluates whether predictions remain stable after controlled transformations such as recompression, brightness changes, or resizing. It is used to communicate how fragile a prediction may be.

### Reporting

The report service generates JSON and text outputs from the pipeline result. The frontend can display the report and expose saved artifact maps.

## 10. API overview

| Method | Endpoint | Use |
|---|---|---|
| `GET` | `/api/v1/health` | Checks service availability. |
| `POST` | `/api/v1/analysis/upload` | Validates and stores single-image evidence. |
| `POST` | `/api/v1/analysis/{id}/run` | Starts the full asynchronous analysis. |
| `GET` | `/api/v1/analysis/{id}/status` | Returns pipeline status and progress. |
| `GET` | `/api/v1/analysis/{id}` | Returns full stored analysis. |
| `GET` | `/api/v1/analysis/{id}/evidence` | Returns evidence metadata. |
| `GET` | `/api/v1/analysis/{id}/camera` | Returns camera-attribution result. |
| `GET` | `/api/v1/analysis/{id}/manipulation` | Returns manipulation result. |
| `GET` | `/api/v1/analysis/{id}/robustness` | Returns robustness result. |
| `GET` | `/api/v1/analysis/{id}/report` | Returns generated report data. |
| `POST` | `/api/v1/compare` | Compares original and edited images. |

## 11. Data and Git policy

The following are runtime or machine-local files and should remain excluded through `.gitignore`:

- `backend/artifacts/` — uploads, maps, overlays and reports
- `backend/cameratrace.db` — local SQLite database
- downloaded camera datasets and archives
- generated feature arrays and train/test splits
- locally trained model weights and joblib files
- virtual environments, cache files and `.env` secrets

The repository should contain source code, configuration templates, tests, documentation, and small deliberate static assets. This keeps GitHub pushes fast and avoids file-size-limit failures.

## 12. Running locally

```powershell
cd backend
venv\Scripts\activate
python run.py
```

Then open `http://127.0.0.1:8000/`. API documentation is available at `http://127.0.0.1:8000/docs`.

## 13. Current visual design principles

- Palette: peach beige, warm white, ink, and forensic red.
- Type: Manrope for readable content, Space Grotesk for high-impact headings, and DM Mono for technical evidence labels.
- Motion: CSS-only scanner, orbit, depth-card and glow animations; no additional JavaScript animation library.
- Accessibility: `prefers-reduced-motion` disables the decorative animations.
- Performance: images remain local static assets and 3D effects use CSS transforms rather than a WebGL dependency.
