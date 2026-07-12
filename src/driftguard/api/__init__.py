"""Thin FastAPI adapter for the local canonical sandbox."""

from .app import app, create_app

__all__ = ["app", "create_app"]
