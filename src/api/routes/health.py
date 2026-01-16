"""Health check endpoints."""

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    databases: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Check system health and database connectivity.
    
    Returns:
        Health status and database connection states
    """
    db_status = {}

    # Check ClickHouse
    try:
        request.app.state.clickhouse.query("SELECT 1")
        db_status["clickhouse"] = "healthy"
    except Exception:
        db_status["clickhouse"] = "unhealthy"

    # Check Qdrant
    try:
        request.app.state.qdrant._client.get_collections()
        db_status["qdrant"] = "healthy"
    except Exception:
        db_status["qdrant"] = "unhealthy"

    # Check Neo4j
    try:
        await request.app.state.neo4j.query("RETURN 1")
        db_status["neo4j"] = "healthy"
    except Exception:
        db_status["neo4j"] = "unhealthy"

    overall_status = "healthy" if all(s == "healthy" for s in db_status.values()) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="0.1.0",
        databases=db_status,
    )


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}
