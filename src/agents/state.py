"""Shared state definitions for multi-agent workflows."""

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A verifiable citation to on-chain data."""

    type: Literal["transaction", "address", "block", "timestamp"]
    value: str
    context: str = Field(default="", description="How this citation supports the claim")

    model_config = {"frozen": False}


class Evidence(BaseModel):
    """A piece of evidence from database retrieval."""

    source: Literal["qdrant", "neo4j", "clickhouse"]
    data: dict[str, Any]
    relevance_score: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)

    model_config = {"frozen": False}


class Finding(BaseModel):
    """A single finding with mandatory citations."""

    finding_id: int
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(min_length=1)
    is_inference: bool = False
    evidence: list[Evidence] = Field(default_factory=list)

    model_config = {"frozen": False}


class AnomalyResult(BaseModel):
    """Result from anomaly detection."""

    address: str
    anomaly_type: str
    z_score: float
    is_anomaly: bool
    baseline_mean: float
    baseline_std: float
    observed_value: float
    time_window: tuple[datetime, datetime]

    model_config = {"frozen": False}


class ClusterInfo(BaseModel):
    """Wallet cluster information."""

    cluster_id: int
    addresses: list[str]
    size: int
    central_addresses: list[str]
    total_volume: float

    model_config = {"frozen": False}


class FlowPath(BaseModel):
    """Fund flow path between addresses."""

    source: str
    target: str
    hops: int
    path: list[str]
    total_value: float
    transactions: list[str]

    model_config = {"frozen": False}


class ValidationResult(BaseModel):
    """Result of citation validation."""

    all_valid: bool
    total_checked: int
    verified_count: int
    failed_citations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"frozen": False}


class AnalysisReport(BaseModel):
    """Final analysis report."""

    report_id: str
    generated_at: datetime
    query: str
    summary: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    findings: list[Finding]
    methodology: str
    limitations: list[str]
    raw_evidence_count: int

    model_config = {"frozen": False}


class InvestigationState(TypedDict, total=False):
    """
    Shared state for investigation workflow.
    
    This state is passed between agents in the LangGraph workflow,
    accumulating evidence and analysis results.
    
    Uses TypedDict for LangGraph 0.2+ compatibility.
    """

    # Input (required)
    query: str
    user_id: str

    # Input (optional)
    config: dict[str, Any]
    started_at: datetime

    # Parsed intent
    intent: str | None
    entities: dict[str, Any]

    # Retrieved evidence
    semantic_results: list[dict[str, Any]]
    graph_results: list[dict[str, Any]]
    olap_results: list[dict[str, Any]]
    fused_evidence: list[Evidence]

    # Analysis results
    anomalies: list[AnomalyResult]
    clusters: list[ClusterInfo]
    flow_paths: list[FlowPath]

    # Validation
    validation_result: ValidationResult | None
    validation_attempts: int

    # Output
    report: AnalysisReport | None
    error: str | None

    # Workflow control - messages uses LangGraph's add_messages reducer
    messages: Annotated[list[Any], add_messages]
    next_agent: str | None
    checkpoints: list[str]


def create_initial_state(query: str, user_id: str, config: dict | None = None) -> InvestigationState:
    """Create a new investigation state with default values."""
    return InvestigationState(
        query=query,
        user_id=user_id,
        config=config or {},
        started_at=datetime.utcnow(),
        intent=None,
        entities={},
        semantic_results=[],
        graph_results=[],
        olap_results=[],
        fused_evidence=[],
        anomalies=[],
        clusters=[],
        flow_paths=[],
        validation_result=None,
        validation_attempts=0,
        report=None,
        error=None,
        messages=[],
        next_agent=None,
        checkpoints=[],
    )
