"""CameraTrace FastAPI application."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    analysis,
    camera,
    camera_search,
    cases,
    evidence,
    health,
    manipulation,
    reports,
    robustness,
)
from app.core.config import settings
from app.core.exceptions import CameraTraceError, ImageValidationError
from app.core.logging import setup_logging
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.ensure_dirs()
    init_db()
    yield


app = FastAPI(
    title="CameraTrace",
    description="Explainable Digital Image Forensics & Source Camera Attribution Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(analysis.router, prefix=API_PREFIX)
app.include_router(evidence.router, prefix=API_PREFIX)
app.include_router(camera.router, prefix=API_PREFIX)
app.include_router(manipulation.router, prefix=API_PREFIX)
app.include_router(robustness.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(cases.router, prefix=API_PREFIX)
app.include_router(camera_search.router, prefix=API_PREFIX)


@app.exception_handler(ImageValidationError)
async def image_validation_handler(request: Request, exc: ImageValidationError):
    return JSONResponse(status_code=400, content={"detail": exc.message, **exc.details})


@app.exception_handler(CameraTraceError)
async def camera_trace_handler(request: Request, exc: CameraTraceError):
    return JSONResponse(status_code=422, content={"detail": exc.message, **exc.details})


@app.get("/")
def root():
    return {
        "service": "CameraTrace",
        "docs": "/docs",
        "health": f"{API_PREFIX}/health",
    }
