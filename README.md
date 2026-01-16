# Blockchain Insider Detection System

A production-ready system for detecting insider activity on the Ethereum blockchain using RAG (Retrieval-Augmented Generation) architecture and multi-agent workflows.

## Overview

This system identifies anomalous transactional patterns indicating actions by project teams, venture funds, or major consultants before public announcements (token listings, partnerships, chart movements).

### Key Features

- **Hybrid Search**: Combines semantic search (embeddings) with precise metadata filtering
- **Graph Analysis**: Wallet clustering, fund flow tracing, and network structure analysis
- **Multi-Agent Architecture**: Specialized agents for data collection, retrieval, analysis, and reporting
- **Zero Hallucination**: Every LLM conclusion is grounded in verifiable on-chain data with citations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                          │
│                 (Apache Airflow + LangGraph)                    │
├─────────────────────────────────────────────────────────────────┤
│  Ingestion → ETL → Retrieval → Graph → Anomaly → Report        │
│    Agent    Agent    Agent    Agent    Agent    Agent          │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Qdrant  │        │  Neo4j   │        │ClickHouse│
   │ (Vectors)│        │ (Graph)  │        │  (OLAP)  │
   └──────────┘        └──────────┘        └──────────┘
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector DB | Qdrant | Semantic search, pattern matching |
| Graph DB | Neo4j + GDS | Wallet clustering, flow analysis |
| OLAP | ClickHouse | Time-series, aggregations |
| Orchestration | Apache Airflow | Pipeline scheduling |
| LLM Framework | LangGraph | Multi-agent coordination |
| Embedding | BGE-large-en-v1.5 | Semantic encoding |
| LLM | Qwen3-30B-A3B | Analysis, report generation |
| API | FastAPI | REST/WebSocket interface |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Poetry (for dependency management)
- GPU with 48GB+ VRAM (for LLM inference) or API access

### 1. Clone and Setup

```bash
git clone <repository-url>
cd insider_detection_blockchain

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
vim .env
```

### 2. Start Infrastructure

```bash
# Start all databases
cd docker
docker-compose up -d

# Verify services are healthy
docker-compose ps
```

### 3. Install Dependencies

```bash
# Install Python dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 4. Initialize Databases

```bash
# Run database initialization
python scripts/setup_databases.py

# Load sample data (optional)
python scripts/load_sample_data.py
```

### 5. Start the API

```bash
# Development mode
uvicorn src.api.main:app --reload --port 8080

# Or with Docker
docker-compose up api
```

### 6. Access the System

- **API Documentation**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health
- **Metrics**: http://localhost:8080/metrics

## Usage

### Running an Investigation

```python
import httpx

async with httpx.AsyncClient() as client:
    # Start investigation
    response = await client.post(
        "http://localhost:8080/api/v1/investigations",
        json={
            "query": "Find wallets that accumulated PEPE before the Binance listing on April 2023",
            "config": {"lookback_days": 30}
        }
    )
    investigation_id = response.json()["investigation_id"]
    
    # Poll for results
    while True:
        status = await client.get(
            f"http://localhost:8080/api/v1/investigations/{investigation_id}"
        )
        if status.json()["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(5)
    
    # Get results
    result = await client.get(
        f"http://localhost:8080/api/v1/investigations/{investigation_id}/result"
    )
    print(result.json())
```

### Ad-hoc Queries

```bash
# Address activity
curl "http://localhost:8080/api/v1/address/0x.../activity"

# Token holders
curl "http://localhost:8080/api/v1/token/0x.../holders?limit=100"

# Wallet cluster
curl "http://localhost:8080/api/v1/wallet/0x.../cluster"

# Hybrid search
curl -X POST "http://localhost:8080/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "large accumulation before listing", "limit": 50}'
```

## Project Structure

```
insider_detection_blockchain/
├── config/                 # Configuration files
├── src/
│   ├── agents/            # Multi-agent implementations
│   ├── workflows/         # LangGraph workflow definitions
│   ├── db/                # Database clients
│   ├── api/               # FastAPI application
│   └── utils/             # Shared utilities
├── etl/                   # ETL pipeline code
├── dags/                  # Airflow DAG definitions
├── docker/                # Docker configurations
├── k8s/                   # Kubernetes manifests
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Documentation

- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [Technology Stack](docs/architecture/TECHNOLOGY_STACK.md)
- [Multi-Agent Architecture](docs/architecture/MULTI_AGENT_ARCHITECTURE.md)
- [Implementation Roadmap](docs/architecture/IMPLEMENTATION_ROADMAP.md)
- [Validation & Hallucination Mitigation](docs/architecture/VALIDATION_HALLUCINATION_MITIGATION.md)
- [File Structure](docs/architecture/FILE_STRUCTURE.md)

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check src tests

# Type checking
mypy src

# Format code
ruff format src tests
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Deployment

### Kubernetes

```bash
# Apply base configuration
kubectl apply -k k8s/overlays/dev

# Production deployment
kubectl apply -k k8s/overlays/prod
```

### Helm

```bash
helm install insider-detection helm/insider-detection \
  -f helm/insider-detection/values-prod.yaml
```

## Monitoring

- **Grafana Dashboards**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Jaeger Tracing**: http://localhost:16686

## License

[Your License Here]

## Contributing

[Contributing Guidelines]
