"""
FastAPI application for Blockchain Insider Detection System.

Provides REST API endpoints for investigations, queries, and alerts.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import structlog

from config.settings import get_settings
from src.api.routes import health, investigations, queries
from src.api.middleware.logging import LoggingMiddleware

logger = structlog.get_logger()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info(
        "application_starting",
        environment=settings.environment,
        debug=settings.api.debug,
    )

    # Initialize database connections
    from src.db.clickhouse import get_clickhouse_client
    from src.db.qdrant import get_qdrant_client
    from src.db.neo4j import get_neo4j_client

    app.state.clickhouse = get_clickhouse_client()
    app.state.qdrant = get_qdrant_client()
    app.state.neo4j = await get_neo4j_client()

    logger.info("database_connections_established")

    yield

    # Shutdown
    logger.info("application_shutting_down")

    app.state.clickhouse.close()
    app.state.qdrant.close()
    await app.state.neo4j.close()

    logger.info("database_connections_closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Blockchain Insider Detection API",
        description="API for detecting insider activity on Ethereum blockchain",
        version="0.1.0",
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom logging middleware
    app.add_middleware(LoggingMiddleware)

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(investigations.router, prefix="/api/v1", tags=["Investigations"])
    app.include_router(queries.router, prefix="/api/v1", tags=["Queries"])

    # Mount Prometheus metrics
    if settings.observability.metrics_enabled:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
