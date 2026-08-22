"""FastAPI dependencies."""
from collections.abc import Generator

from sqlalchemy.orm import Session

from app.db.database import get_db

__all__ = ["get_db"]
