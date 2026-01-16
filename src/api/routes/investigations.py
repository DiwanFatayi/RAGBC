"""Investigation API endpoints."""

from datetime import datetime
from typing import Any
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class InvestigationRequest(BaseModel):
    """Request to start a new investigation."""

    query: str = Field(..., description="Natural language investigation query")
    config: dict[str, Any] = Field(default_factory=dict, description="Optional configuration")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find wallets that accumulated TOKEN before the listing on 2024-01-15",
                    "config": {"lookback_days": 30},
                }
            ]
        }
    }


class InvestigationResponse(BaseModel):
    """Response from starting an investigation."""

    investigation_id: str
    status: str
    message: str


class InvestigationStatus(BaseModel):
    """Status of an investigation."""

    investigation_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    started_at: datetime
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class InvestigationResult(BaseModel):
    """Complete investigation result."""

    investigation_id: str
    query: str
    report: dict[str, Any]
    confidence_score: float
    execution_time_seconds: float
    checkpoints: list[str]


# In-memory store for demo (use Redis in production)
_investigations: dict[str, dict] = {}


@router.post("/investigations", response_model=InvestigationResponse)
async def create_investigation(
    request: Request,
    body: InvestigationRequest,
    background_tasks: BackgroundTasks,
) -> InvestigationResponse:
    """
    Start a new insider detection investigation.
    
    The investigation runs asynchronously. Use the returned ID
    to poll for status and results.
    """
    investigation_id = str(uuid.uuid4())

    # Store initial state
    _investigations[investigation_id] = {
        "id": investigation_id,
        "query": body.query,
        "config": body.config,
        "status": "pending",
        "progress": 0.0,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    # Queue background task
    background_tasks.add_task(
        run_investigation,
        investigation_id,
        body.query,
        body.config,
        request.app,
    )

    return InvestigationResponse(
        investigation_id=investigation_id,
        status="pending",
        message="Investigation queued for processing",
    )


@router.get("/investigations/{investigation_id}", response_model=InvestigationStatus)
async def get_investigation_status(investigation_id: str) -> InvestigationStatus:
    """Get the current status of an investigation."""
    if investigation_id not in _investigations:
        raise HTTPException(status_code=404, detail="Investigation not found")

    inv = _investigations[investigation_id]
    return InvestigationStatus(
        investigation_id=inv["id"],
        status=inv["status"],
        progress=inv["progress"],
        started_at=inv["started_at"],
        completed_at=inv["completed_at"],
        result=inv["result"],
        error=inv["error"],
    )


@router.get("/investigations/{investigation_id}/result", response_model=InvestigationResult)
async def get_investigation_result(investigation_id: str) -> InvestigationResult:
    """Get the complete result of a finished investigation."""
    if investigation_id not in _investigations:
        raise HTTPException(status_code=404, detail="Investigation not found")

    inv = _investigations[investigation_id]

    if inv["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Investigation not completed. Current status: {inv['status']}",
        )

    if inv["error"]:
        raise HTTPException(status_code=500, detail=inv["error"])

    return InvestigationResult(
        investigation_id=inv["id"],
        query=inv["query"],
        report=inv["result"]["report"],
        confidence_score=inv["result"].get("confidence_score", 0.0),
        execution_time_seconds=(inv["completed_at"] - inv["started_at"]).total_seconds(),
        checkpoints=inv["result"].get("checkpoints", []),
    )


@router.delete("/investigations/{investigation_id}")
async def cancel_investigation(investigation_id: str) -> dict[str, str]:
    """Cancel a running investigation."""
    if investigation_id not in _investigations:
        raise HTTPException(status_code=404, detail="Investigation not found")

    inv = _investigations[investigation_id]

    if inv["status"] in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel investigation with status: {inv['status']}",
        )

    inv["status"] = "cancelled"
    inv["completed_at"] = datetime.utcnow()

    return {"message": "Investigation cancelled"}


async def run_investigation(
    investigation_id: str,
    query: str,
    config: dict[str, Any],
    app: Any,
) -> None:
    """Background task to run the investigation workflow."""
    inv = _investigations[investigation_id]
    inv["status"] = "running"
    inv["progress"] = 0.1

    try:
        # Import here to avoid circular imports
        from src.workflows.investigation import InvestigationRunner, create_investigation_workflow

        # Create workflow with placeholder nodes for now
        # In production, these would be actual agent implementations
        async def placeholder_node(state):
            return {}

        workflow = create_investigation_workflow(
            query_parser_node=placeholder_node,
            retrieval_node=placeholder_node,
            graph_analysis_node=placeholder_node,
            anomaly_detection_node=placeholder_node,
            validator_node=placeholder_node,
            report_generator_node=placeholder_node,
            error_handler_node=placeholder_node,
        )

        runner = InvestigationRunner(workflow)

        # Run the investigation
        result = await runner.investigate(
            query=query,
            user_id="api-user",
            config=config,
            thread_id=investigation_id,
        )

        inv["result"] = result
        inv["status"] = "completed" if result.get("success") else "failed"
        inv["error"] = result.get("error")
        inv["progress"] = 1.0

    except Exception as e:
        inv["status"] = "failed"
        inv["error"] = str(e)
        inv["progress"] = 1.0

    finally:
        inv["completed_at"] = datetime.utcnow()
