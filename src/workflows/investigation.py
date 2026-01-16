"""
Main investigation workflow using LangGraph.

This module defines the StateGraph that coordinates all agents
for blockchain insider detection investigations.
"""

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.state import InvestigationState, create_initial_state


def route_after_retrieval(state: InvestigationState) -> Literal["graph_analysis", "anomaly_detection", "error_handler"]:
    """Determine next step based on retrieval results."""
    # Access TypedDict state using dict methods
    if state.get("error"):
        return "error_handler"

    intent = state.get("intent") or ""

    # Graph analysis needed for flow/cluster investigations
    if intent in ["flow_analysis", "cluster_investigation", "wallet_tracing"]:
        return "graph_analysis"

    # Sufficient evidence for anomaly detection
    fused_evidence = state.get("fused_evidence", [])
    if len(fused_evidence) > 0:
        return "anomaly_detection"

    return "error_handler"


def route_after_validation(state: InvestigationState) -> Literal["report_generator", "retrieval", "error_handler"]:
    """Determine next step based on validation results."""
    # Access TypedDict state using dict methods
    validation = state.get("validation_result")

    if validation is None:
        return "error_handler"

    # Handle both Pydantic model and dict
    all_valid = validation.all_valid if hasattr(validation, 'all_valid') else validation.get("all_valid", False)
    if all_valid:
        return "report_generator"

    validation_attempts = state.get("validation_attempts", 0)
    if validation_attempts >= 3:
        return "error_handler"

    # Retry with stricter constraints
    return "retrieval"


def should_continue_to_graph(state: InvestigationState) -> bool:
    """Check if graph analysis is needed based on entities."""
    entities = state.get("entities", {})
    return bool(
        entities.get("addresses")
        or entities.get("trace_flow")
        or entities.get("find_clusters")
    )


def create_investigation_workflow(
    query_parser_node,
    retrieval_node,
    graph_analysis_node,
    anomaly_detection_node,
    validator_node,
    report_generator_node,
    error_handler_node,
    checkpointer=None,
):
    """
    Create the investigation workflow graph.
    
    Args:
        *_node: Agent node functions that take InvestigationState and return updates
        checkpointer: Optional checkpointer for persistence (default: in-memory)
    
    Returns:
        Compiled LangGraph workflow
    """
    # Initialize the graph with state schema
    workflow = StateGraph(InvestigationState)

    # Add all agent nodes
    workflow.add_node("query_parser", query_parser_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("graph_analysis", graph_analysis_node)
    workflow.add_node("anomaly_detection", anomaly_detection_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("error_handler", error_handler_node)

    # Set entry point
    workflow.set_entry_point("query_parser")

    # Define edges
    # Query parser always goes to retrieval
    workflow.add_edge("query_parser", "retrieval")

    # Conditional routing after retrieval
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "graph_analysis": "graph_analysis",
            "anomaly_detection": "anomaly_detection",
            "error_handler": "error_handler",
        },
    )

    # Graph analysis leads to anomaly detection
    workflow.add_edge("graph_analysis", "anomaly_detection")

    # Anomaly detection leads to validator
    workflow.add_edge("anomaly_detection", "validator")

    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "report_generator": "report_generator",
            "retrieval": "retrieval",  # Retry loop
            "error_handler": "error_handler",
        },
    )

    # Terminal nodes
    workflow.add_edge("report_generator", END)
    workflow.add_edge("error_handler", END)

    # Use provided checkpointer or create in-memory one
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Compile the graph
    return workflow.compile(checkpointer=checkpointer)


def create_batch_analysis_workflow(
    batch_loader_node,
    parallel_analysis_node,
    aggregator_node,
    alert_generator_node,
    checkpointer=None,
):
    """
    Create workflow for batch analysis of multiple addresses/tokens.
    
    Used for scheduled anomaly detection across the entire dataset.
    """
    from pydantic import BaseModel, Field

    class BatchState(BaseModel):
        """State for batch analysis workflow."""

        batch_id: str
        targets: list[dict]  # List of {address, token, event} to analyze
        results: list[dict] = Field(default_factory=list)
        alerts: list[dict] = Field(default_factory=list)
        processed_count: int = 0
        error_count: int = 0

    workflow = StateGraph(BatchState)

    workflow.add_node("batch_loader", batch_loader_node)
    workflow.add_node("parallel_analysis", parallel_analysis_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("alert_generator", alert_generator_node)

    workflow.set_entry_point("batch_loader")
    workflow.add_edge("batch_loader", "parallel_analysis")
    workflow.add_edge("parallel_analysis", "aggregator")
    workflow.add_edge("aggregator", "alert_generator")
    workflow.add_edge("alert_generator", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


class InvestigationRunner:
    """
    High-level interface for running investigations.
    
    Handles workflow initialization, execution, and result extraction.
    """

    def __init__(self, workflow, timeout_seconds: int = 600):
        self.workflow = workflow
        self.timeout_seconds = timeout_seconds

    async def investigate(
        self,
        query: str,
        user_id: str,
        config: dict | None = None,
        thread_id: str | None = None,
    ) -> dict:
        """
        Run an investigation for the given query.
        
        Args:
            query: Natural language investigation query
            user_id: ID of the requesting user
            config: Optional configuration overrides
            thread_id: Optional thread ID for resuming investigations
            
        Returns:
            Investigation results including report or error
        """
        import uuid

        # Initialize state using helper function (TypedDict)
        initial_state = create_initial_state(
            query=query,
            user_id=user_id,
            config=config,
        )

        # Create thread config for checkpointing
        thread_config = {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4()),
            }
        }

        # Run the workflow - TypedDict is already a dict
        final_state = await self.workflow.ainvoke(
            initial_state,
            config=thread_config,
        )

        # Extract results
        if final_state.get("error"):
            return {
                "success": False,
                "error": final_state["error"],
                "checkpoints": final_state.get("checkpoints", []),
            }

        report = final_state.get("report")
        if report:
            return {
                "success": True,
                "report": report,
                "checkpoints": final_state.get("checkpoints", []),
            }

        return {
            "success": False,
            "error": "No report generated",
            "checkpoints": final_state.get("checkpoints", []),
        }

    async def resume(self, thread_id: str) -> dict:
        """Resume an interrupted investigation."""
        thread_config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Get current state
        state = await self.workflow.aget_state(thread_config)

        if state.values.get("report") or state.values.get("error"):
            # Already completed
            return {
                "success": not state.values.get("error"),
                "report": state.values.get("report"),
                "error": state.values.get("error"),
            }

        # Continue execution
        final_state = await self.workflow.ainvoke(None, config=thread_config)

        return {
            "success": not final_state.get("error"),
            "report": final_state.get("report"),
            "error": final_state.get("error"),
        }
