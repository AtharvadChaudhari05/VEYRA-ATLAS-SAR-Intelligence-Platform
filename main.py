from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import ensure_storage, get_settings
from app.core.database import Database


def create_app() -> FastAPI:
    settings = get_settings()
    ensure_storage(settings)
    Database(settings.database_path).initialize()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Backend API for SAR-to-optical prediction, flood analysis, geospatial area calculation, and dataset scanning.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()

