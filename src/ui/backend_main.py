"""Main FastAPI application entry point.

This file provides a clean import path for uvicorn:
    uvicorn src.ui.backend_main:app
"""

from src.ui.backend import app

# Re-export for uvicorn
__all__ = ["app"]
