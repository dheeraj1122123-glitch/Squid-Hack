# CameraTrace Datasets

This directory stores forensic datasets and generated manifests for training.

## Supported Datasets

| Dataset | URL | Task | Local Path |
|---------|-----|------|------------|
| Dresden Image Database | https://www.mmlab.ie.cuhk.edu.hk/archive/fh/camera_model.htm | Camera ID | `datasets/camera/dresden/` |
| NIST MFC | https://mfc.nist.gov/ | Camera ID | `datasets/camera/nist_mfc/` |
| UTIA Vision Lab | http://staff.utia.cas.cz/novozada/db/ | Camera ID | `datasets/camera/vision/` |
| CASIA 2.0 | http://forensics.idealtest.org/digitalimage/CASIA2.0/CASIA2.0.html | Manipulation | `datasets/manipulation/casia2/` |
| CASIA Ground Truth | https://github.com/namtpham/casia2groundtruth | Manipulation masks | `datasets/manipulation/casia2_groundtruth/` |
| FAU Manipulation | https://www5.cs.fau.de/research/data/image-manipulation/ | Manipulation | `datasets/manipulation/fau/` |

## Download

```bash
python scripts/download_datasets.py --list
python scripts/download_datasets.py --clone-groundtruth
```

Most datasets require manual registration and download. Place extracted files in the paths above.

## Demo Data (Development)

```bash
python scripts/generate_demo_data.py
```

## Expected Structure After Preparation

### Camera (Dresden example)

```
datasets/camera/dresden/
├── Canon/
│   └── EOS_90D/
│       ├── IMG_001.jpg
│       └── ...
├── Sony/
│   └── DSC-H50/
│       └── ...
└── ...
```

### Manipulation (CASIA example)

```
datasets/manipulation/casia2/
├── Au/          # Authentic
├── Tp/          # Tampered
└── ...
```

## Generated Files

After running preparation:

```
datasets/camera/
├── manifest.csv          # Full dataset manifest
├── manifest.meta.json    # Metadata
├── train.csv             # Training split
├── test.csv              # Test split
├── split_meta.json       # Split info
├── features.npz          # Extracted features
└── evaluation.json       # Model evaluation

datasets/manipulation/
├── manifest.csv
└── evaluation.json
```

## Preparation Commands

```bash
# Camera
python -m training.camera.prepare_dataset --root datasets/camera/dresden --type dresden
python -m training.camera.split --manifest datasets/camera/manifest.csv
python -m training.camera.extract_features --manifest datasets/camera/train.csv

# Manipulation
python -m training.manipulation.prepare_dataset --root datasets/manipulation/casia2 --type casia
```

## Manifest Fields

| Field | Description |
|-------|-------------|
| image_path | Absolute path to image |
| sha256 | Cryptographic hash |
| perceptual_hash | pHash for near-duplicate detection |
| manufacturer | Camera manufacturer |
| camera_model | Camera model label |
| physical_device | Device instance ID |
| manipulation_type | For manipulation datasets |
| dataset | Source dataset name |

## Leakage Prevention

- SHA-256 exact duplicate removal
- Perceptual hash near-duplicate removal
- Group-aware train/test splits (by scene or device, not individual near-duplicates)
