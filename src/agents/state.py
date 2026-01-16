"""Shared state definitions for multi-agent workflows."""

from datetime import datetime
from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A verifiable citation to on-chain data."""

    type: Literal["transaction", "address", "block", "timestamp"]
    value: str
    context: str = Field(default="", description="How this citation supports the claim")


class Evidence(BaseModel):
    """A piece of evidence from database retrieval."""

    source: Literal["qdrant", "neo4j", "clickhouse"]
    data: dict[str, Any]
    relevance_score: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)


class Finding(BaseModel):
    """A single finding with mandatory citations."""

    finding_id: int
    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(min_length=1)
    is_inference: bool = False
    evidence: list[Evidence] = Field(default_factory=list)


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


class ClusterInfo(BaseModel):
    """Wallet cluster information."""

    cluster_id: int
    addresses: list[str]
    size: int
    central_addresses: list[str]
    total_volume: float


class FlowPath(BaseModel):
    """Fund flow path between addresses."""

    source: str
    target: str
    hops: int
    path: list[str]
    total_value: float
    transactions: list[str]


class ValidationResult(BaseModel):
    """Result of citation validation."""

    all_valid: bool
    total_checked: int
    verified_count: int
    failed_citations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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


class InvestigationState(BaseModel):
    """
    Shared state for investigation workflow.
    
    This state is passed between agents in the LangGraph workflow,
    accumulating evidence and analysis results.
    """

    # Input
    query: str
    user_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)

    # Parsed intent
    intent: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)

    # Retrieved evidence
    semantic_results: list[dict[str, Any]] = Field(default_factory=list)
    graph_results: list[dict[str, Any]] = Field(default_factory=list)
    olap_results: list[dict[str, Any]] = Field(default_factory=list)
    fused_evidence: list[Evidence] = Field(default_factory=list)

    # Analysis results
    anomalies: list[AnomalyResult] = Field(default_factory=list)
    clusters: list[ClusterInfo] = Field(default_factory=list)
    flow_paths: list[FlowPath] = Field(default_factory=list)

    # Validation
    validation_result: ValidationResult | None = None
    validation_attempts: int = 0

    # Output
    report: AnalysisReport | None = None
    error: str | None = None

    # Workflow control
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    next_agent: str | None = None
    checkpoints: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
