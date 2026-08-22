"""Application configuration."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


DATASET_LINKS = [
    "https://www.mmlab.ie.cuhk.edu.hk/archive/fh/camera_model.htm",
    "https://mfc.nist.gov/",
    "http://forensics.idealtest.org/digitalimage/CASIA2.0/CASIA2.0.html",
    "https://github.com/namtpham/casia2groundtruth",
    "https://www5.cs.fau.de/research/data/image-manipulation/",
    "http://staff.utia.cas.cz/novozada/db/",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    database_url: str = f"sqlite:///{BASE_DIR / 'cameratrace.db'}"
    upload_dir: Path = BASE_DIR / "artifacts" / "uploads"
    artifact_dir: Path = BASE_DIR / "artifacts"
    model_dir: Path = BASE_DIR / "models"
    max_upload_mb: int = 25
    camera_model_path: str = ""
    manipulation_model_path: str = ""
    ai_generation_model_path: str = ""
    enable_robustness: bool = True
    enable_ai_detector: bool = True
    camera_classification_mode: Literal["hierarchical", "flat"] = "hierarchical"
    demo_mode: bool = False
    log_level: str = "INFO"
    allowed_extensions: str = "jpg,jpeg,png,tiff,bmp,webp"
    unknown_camera_threshold: float = 0.70
    min_camera_training_samples: int = 100
    prnu_correlation_threshold: float = 0.01

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_ext_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",")}

    def ensure_dirs(self) -> None:
        for sub in ("uploads", "residuals", "heatmaps", "spectra", "reports"):
            (self.artifact_dir / sub).mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "camera").mkdir(parents=True, exist_ok=True)
        (self.model_dir / "manipulation").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
