param(
    [ValidateSet("image", "device")]
    [string]$SplitBy = "image"
)

$ErrorActionPreference = "Stop"
$Base = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Base "venv\Scripts\python.exe"
$Dataset = Join-Path $Base "datasets\camera\dresden"
$Manifest = Join-Path $Base "datasets\camera\manifest.csv"
$Features = Join-Path $Base "datasets\camera\features_dresden.npz"

function Invoke-PythonStage {
    param([string]$Name, [string[]]$Arguments)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE. Training stopped; no stale model was used."
    }
}

if (!(Test-Path -LiteralPath $Python)) { throw "Python environment missing: $Python" }
if (!(Test-Path -LiteralPath $Dataset)) { throw "Dresden dataset missing: $Dataset" }
if ($SplitBy -eq "image") {
    Write-Warning "This Kaggle subset has one physical device per model. Image split is demo-only, not a real-world benchmark."
}

Invoke-PythonStage "Build Dresden manifest" @("-m", "training.camera.prepare_dataset", "--root", $Dataset, "--type", "dresden", "--output", $Manifest)
Invoke-PythonStage "Create leakage-aware split" @("-m", "training.camera.split", "--manifest", $Manifest, "--split-by", $SplitBy)
Invoke-PythonStage "Extract signal-level camera features" @("-m", "training.camera.extract_features", "--manifest", (Join-Path $Base "datasets\camera\train.csv"), "--output", $Features)
Invoke-PythonStage "Train calibrated XGBoost camera model" @("-m", "training.camera.train_classifier", "--features", $Features, "--model", "xgboost")
Invoke-PythonStage "Evaluate held-out images" @("-m", "training.camera.evaluate", "--test-manifest", (Join-Path $Base "datasets\camera\test.csv"))

Write-Host "`nTraining complete. Start the app with: venv\Scripts\python.exe run.py" -ForegroundColor Green