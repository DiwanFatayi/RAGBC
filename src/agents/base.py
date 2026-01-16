"""Base agent class for all specialized agents."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from prometheus_client import Counter, Histogram

from src.agents.state import InvestigationState

logger = structlog.get_logger()

# Prometheus metrics
AGENT_INVOCATIONS = Counter(
    "agent_invocations_total",
    "Total agent invocations",
    ["agent_name", "status"],
)
AGENT_DURATION = Histogram(
    "agent_duration_seconds",
    "Agent execution duration",
    ["agent_name"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    
    Provides common functionality:
    - Tool management
    - Error handling
    - Metrics collection
    - Logging
    
    Note: InvestigationState is a TypedDict for LangGraph compatibility.
    Access fields using dict syntax: state["field"] or state.get("field", default)
    """

    name: str = "base_agent"
    description: str = "Base agent class"

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        tools: list[BaseTool] | None = None,
        timeout_seconds: int = 300,
    ):
        self.llm = llm
        self.tools = tools or []
        self.timeout_seconds = timeout_seconds
        self._logger = logger.bind(agent=self.name)

    @abstractmethod
    async def process(self, state: InvestigationState) -> dict[str, Any]:
        """
        Process the current state and return updates.
        
        Args:
            state: Current investigation state (TypedDict)
            
        Returns:
            Dictionary of state updates to apply
        """
        pass

    async def __call__(self, state: InvestigationState) -> dict[str, Any]:
        """
        Execute the agent with error handling and metrics.
        
        This is the entry point called by LangGraph.
        """
        start_time = datetime.utcnow()
        query = state.get("query", "")
        self._logger.info("agent_started", query=query[:100] if query else "")

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self.process(state),
                timeout=self.timeout_seconds,
            )

            # Record success metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            AGENT_INVOCATIONS.labels(agent_name=self.name, status="success").inc()
            AGENT_DURATION.labels(agent_name=self.name).observe(duration)

            self._logger.info(
                "agent_completed",
                duration_seconds=duration,
                updates=list(result.keys()),
            )

            # Add checkpoint - state is TypedDict, use .get() for safe access
            existing_checkpoints = list(state.get("checkpoints", []))
            existing_checkpoints.append(f"{self.name}:{datetime.utcnow().isoformat()}")
            result["checkpoints"] = existing_checkpoints

            return result

        except asyncio.TimeoutError:
            AGENT_INVOCATIONS.labels(agent_name=self.name, status="timeout").inc()
            self._logger.error("agent_timeout", timeout=self.timeout_seconds)
            return {"error": f"Agent {self.name} timed out after {self.timeout_seconds}s"}

        except Exception as e:
            AGENT_INVOCATIONS.labels(agent_name=self.name, status="error").inc()
            self._logger.exception("agent_error", error=str(e))
            return {"error": f"Agent {self.name} failed: {str(e)}"}

    def get_tool(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def bind_tools_to_llm(self) -> BaseChatModel:
        """Bind tools to the LLM for function calling."""
        if not self.llm:
            raise ValueError("LLM not configured for this agent")
        if not self.tools:
            return self.llm
        return self.llm.bind_tools(self.tools)


class ToolExecutor:
    """
    Execute tools with error handling and retries.
    
    Provides a consistent interface for tool execution across agents.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._logger = logger.bind(component="tool_executor")

    async def execute(
        self,
        tool: BaseTool | Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a tool with retries."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if isinstance(tool, BaseTool):
                    result = await tool.ainvoke(kwargs)
                else:
                    result = await tool(*args, **kwargs)
                return result

            except Exception as e:
                last_error = e
                self._logger.warning(
                    "tool_execution_failed",
                    tool=getattr(tool, "name", str(tool)),
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"Tool execution failed after {self.max_retries} attempts: {last_error}")

    async def execute_parallel(
        self,
        tool_calls: list[tuple[BaseTool | Callable, dict[str, Any]]],
    ) -> list[Any]:
        """Execute multiple tools in parallel."""
        tasks = [self.execute(tool, **kwargs) for tool, kwargs in tool_calls]
        return await asyncio.gather(*tasks, return_exceptions=True)
