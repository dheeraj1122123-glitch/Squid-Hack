# CameraTrace

**Explainable Digital Image Forensics & Source Camera Attribution Platform**

A production-quality FastAPI backend for digital image forensics, camera model identification, and manipulation detection — built for hackathon demonstrations with scientific honesty.

## Architecture

```
IMAGE UPLOAD → EVIDENCE INTAKE
    ├── Metadata Analysis
    ├── Camera Forensics (PRNU, Noise, CFA, Frequency, JPEG)
    └── Manipulation Engine (ELA, Copy-Move, Splicing, Localization)
        → Forensic Consistency Engine → Explainable Report → API Response
```

## Quick Start

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
copy .env.example .env

# Generate demo data & train (optional)
python scripts/generate_demo_data.py
python -m training.camera.prepare_dataset --root datasets/camera/demo --type generic
python -m training.camera.split
python -m training.camera.extract_features
python -m training.camera.train_classifier

# Run server
python run.py
# OR: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/analysis/upload` | Upload image evidence |
| POST | `/api/v1/analysis/{id}/run` | Run forensic analysis |
| GET | `/api/v1/analysis/{id}` | Full analysis result |
| GET | `/api/v1/analysis/{id}/status` | Job status |
| GET | `/api/v1/analysis/{id}/evidence` | Evidence metadata |
| GET | `/api/v1/analysis/{id}/camera` | Camera attribution |
| GET | `/api/v1/analysis/{id}/manipulation` | Manipulation analysis |
| GET | `/api/v1/analysis/{id}/robustness` | Robustness testing |
| GET | `/api/v1/analysis/{id}/report` | Forensic report |
| POST | `/api/v1/cases` | Create forensic case |

### Example

```bash
# Upload
curl -X POST http://localhost:8000/api/v1/analysis/upload \
  -F "file=@image.jpg"

# Run analysis
curl -X POST http://localhost:8000/api/v1/analysis/{analysis_id}/run

# Get report
curl http://localhost:8000/api/v1/analysis/{analysis_id}/report
```

## Datasets

Configured dataset sources:

- **Dresden Image Database** — Camera model ID
- **NIST MFC** — Media forensics challenge
- **CASIA 2.0** — Image tampering detection
- **CASIA Ground Truth** — Tampering masks
- **FAU Manipulation Dataset**
- **UTIA Vision Lab**

```bash
python scripts/download_datasets.py --list
python scripts/verify_datasets.py
```

See [datasets/README.md](datasets/README.md) for structure and preparation.

## Training Pipeline

### Camera Model Classification

```bash
python -m training.camera.prepare_dataset --root datasets/camera/dresden --type dresden
python -m training.camera.split --split-by device
python -m training.camera.extract_features
python -m training.camera.train_classifier --model xgboost
python -m training.camera.evaluate
```

### Manipulation Detection

```bash
python -m training.manipulation.prepare_dataset --root datasets/manipulation/casia2 --type casia
python -m training.manipulation.train_detector
python -m training.manipulation.evaluate
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| APP_ENV | development | Environment |
| DATABASE_URL | sqlite:///./cameratrace.db | Database |
| MAX_UPLOAD_MB | 25 | Upload size limit |
| ENABLE_ROBUSTNESS | true | Robustness testing |
| CAMERA_CLASSIFICATION_MODE | hierarchical | flat or hierarchical |
| UNKNOWN_CAMERA_THRESHOLD | 0.35 | Open-set rejection threshold |

## Docker

```bash
docker-compose up --build
```

## Testing

```bash
pytest tests/ -v
python scripts/verify_datasets.py
```

## Forensic Modules

| Module | Status | Notes |
|--------|--------|-------|
| Metadata Extraction | ✅ Real | EXIF parsing, suspicious flags |
| Noise Residual | ✅ Real | Gaussian/median/high-pass |
| PRNU | ✅ Real | Requires reference fingerprints |
| CFA/Demosaicing | ✅ Real | Image-level features |
| Frequency Analysis | ✅ Real | FFT, spectral entropy |
| JPEG Analysis | ✅ Real | Quality estimate, recompression heuristic |
| ELA | ✅ Real | JPEG only, exploratory |
| Copy-Move | ✅ Real | ORB/SIFT baseline |
| Splicing | ✅ Real | Patch-based baseline |
| Camera Classifier | ⚙️ Trainable | XGBoost/RF/SVM |
| Manipulation Detector | ⚙️ Trainable | Feature-based RF |

## Scientific Rules

- Never claims 100% accuracy or definitive fraud
- Distinguishes observations, predictions, evidence, uncertainty
- Reports `MODEL_NOT_TRAINED` when models unavailable
- PRNU returns `reference_fingerprint_unavailable` without references
- Uses forensic language: "potentially manipulated", "suspicious", "inconclusive"

## Limitations

- Camera attribution requires trained models on forensic datasets
- PRNU matching requires reference images from the same physical device
- ELA and copy-move are baselines, not state-of-the-art
- Metadata is not treated as ground truth

## Project Structure

```
backend/
├── app/           # FastAPI application
│   ├── api/       # REST endpoints
│   ├── core/      # Config, logging, security
│   ├── db/        # SQLAlchemy models
│   ├── forensic/  # Forensic engines
│   ├── schemas/   # Pydantic models
│   └── services/  # Business logic
├── training/      # ML training pipelines
├── datasets/      # Dataset manifests
├── models/        # Trained model registry
├── artifacts/     # Uploads, heatmaps, reports
├── scripts/       # Utility scripts
└── tests/         # Unit & integration tests
```
