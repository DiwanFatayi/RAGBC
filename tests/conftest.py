"""Pytest configuration and fixtures."""

import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_clickhouse_client() -> MagicMock:
    """Mock ClickHouse client for testing."""
    client = MagicMock()
    client.query.return_value.result_rows = []
    client.query.return_value.column_names = []
    return client


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    """Mock Qdrant client for testing."""
    client = MagicMock()
    client.search.return_value = []
    client._client.get_collections.return_value.collections = []
    return client


@pytest.fixture
def mock_neo4j_client() -> AsyncMock:
    """Mock Neo4j client for testing."""
    client = AsyncMock()
    client.query.return_value = []
    return client


@pytest.fixture
def mock_embedding_model() -> MagicMock:
    """Mock embedding model for testing."""
    import numpy as np

    model = MagicMock()
    model.encode.return_value = np.random.rand(1024).astype(np.float32)
    return model


@pytest.fixture
def api_client(
    mock_clickhouse_client: MagicMock,
    mock_qdrant_client: MagicMock,
    mock_neo4j_client: AsyncMock,
) -> Generator[TestClient, None, None]:
    """Create test client with mocked dependencies."""
    from src.api.main import app

    # Inject mocks
    app.state.clickhouse = mock_clickhouse_client
    app.state.qdrant = mock_qdrant_client
    app.state.neo4j = mock_neo4j_client

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_transaction() -> dict:
    """Sample transaction for testing."""
    return {
        "hash": "0x" + "a" * 64,
        "block_number": 18234567,
        "block_timestamp": "2024-01-15T10:30:00Z",
        "from_address": "0x" + "1" * 40,
        "to_address": "0x" + "2" * 40,
        "value": 1000000000000000000,  # 1 ETH
        "gas_used": 21000,
        "gas_price": 50000000000,
        "status": 1,
    }


@pytest.fixture
def sample_token_transfer() -> dict:
    """Sample token transfer for testing."""
    return {
        "tx_hash": "0x" + "b" * 64,
        "log_index": 0,
        "block_number": 18234567,
        "block_timestamp": "2024-01-15T10:30:00Z",
        "token_address": "0x" + "t" * 40,
        "from_address": "0x" + "1" * 40,
        "to_address": "0x" + "2" * 40,
        "value": 1000000000000000000000,  # 1000 tokens
        "token_type": "ERC20",
    }


@pytest.fixture
def sample_investigation_state() -> dict:
    """Sample investigation state for testing."""
    from datetime import datetime

    return {
        "query": "Find wallets accumulating TOKEN before listing",
        "user_id": "test-user",
        "config": {},
        "started_at": datetime.utcnow(),
        "intent": "accumulation_detection",
        "entities": {
            "tokens": ["0x" + "t" * 40],
            "time_range": {
                "end": datetime(2024, 1, 15),
            },
        },
        "semantic_results": [],
        "graph_results": [],
        "olap_results": [],
        "fused_evidence": [],
        "anomalies": [],
        "clusters": [],
        "flow_paths": [],
        "validation_result": None,
        "validation_attempts": 0,
        "report": None,
        "error": None,
        "messages": [],
        "next_agent": None,
        "checkpoints": [],
    }
