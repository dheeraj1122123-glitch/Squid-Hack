"""Pytest configuration."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import init_db, engine, Base

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()
    yield
