"""SQLAlchemy ORM models."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(256))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    analyses: Mapped[list["Analysis"]] = relationship(back_populates="case")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    case_id: Mapped[int | None] = mapped_column(ForeignKey("cases.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(64), default="QUEUED")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    case: Mapped["Case | None"] = relationship(back_populates="analyses")
    evidence: Mapped["Evidence | None"] = relationship(back_populates="analysis", uselist=False)
    camera_prediction: Mapped["CameraPrediction | None"] = relationship(back_populates="analysis", uselist=False)
    manipulation_result: Mapped["ManipulationResult | None"] = relationship(back_populates="analysis", uselist=False)
    artifacts: Mapped[list["ForensicArtifact"]] = relationship(back_populates="analysis")
    robustness_results: Mapped[list["RobustnessResult"]] = relationship(back_populates="analysis")


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"))
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    channels: Mapped[int] = mapped_column(Integer)
    format: Mapped[str] = mapped_column(String(32))
    file_size: Mapped[int] = mapped_column(Integer)
    original_path: Mapped[str] = mapped_column(String(1024))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    analysis: Mapped["Analysis"] = relationship(back_populates="evidence")


class CameraPrediction(Base):
    __tablename__ = "camera_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"))
    manufacturer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    known_camera: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(64))
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    analysis: Mapped["Analysis"] = relationship(back_populates="camera_prediction")


class ManipulationResult(Base):
    __tablename__ = "manipulation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"))
    status: Mapped[str] = mapped_column(String(64))
    overall_suspiciousness: Mapped[float] = mapped_column(Float, default=0.0)
    indicators: Mapped[list | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    analysis: Mapped["Analysis"] = relationship(back_populates="manipulation_result")


class ForensicArtifact(Base):
    __tablename__ = "forensic_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"))
    artifact_type: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(String(1024))

    analysis: Mapped["Analysis"] = relationship(back_populates="artifacts")


class RobustnessResult(Base):
    __tablename__ = "robustness_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analyses.id"))
    transformation: Mapped[str] = mapped_column(String(128))
    prediction: Mapped[str | None] = mapped_column(String(256), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    prediction_changed: Mapped[bool] = mapped_column(Boolean, default=False)

    analysis: Mapped["Analysis"] = relationship(back_populates="robustness_results")
