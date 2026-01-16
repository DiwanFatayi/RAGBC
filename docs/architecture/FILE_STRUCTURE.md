# Project File & Directory Structure

## Complete Directory Tree

```
insider_detection_blockchain/
│
├── README.md                           # Project overview and quickstart
├── pyproject.toml                      # Python project configuration (Poetry)
├── poetry.lock                         # Locked dependencies
├── Makefile                            # Common development commands
├── .env.example                        # Environment variables template
├── .gitignore                          # Git ignore patterns
├── .pre-commit-config.yaml             # Pre-commit hooks configuration
│
├── config/                             # Configuration files
│   ├── __init__.py
│   ├── settings.py                     # Pydantic settings management
│   ├── logging.yaml                    # Logging configuration
│   ├── agents.yaml                     # Agent-specific configurations
│   ├── prompts/                        # LLM prompt templates
│   │   ├── system_prompts.yaml
│   │   ├── query_parser.yaml
│   │   ├── report_generator.yaml
│   │   └── validator.yaml
│   └── detection_rules/                # Anomaly detection rule configs
│       ├── accumulation_patterns.yaml
│       ├── flow_patterns.yaml
│       └── timing_patterns.yaml
│
├── src/                                # Main source code
│   ├── __init__.py
│   │
│   ├── agents/                         # Multi-agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                     # Base agent class
│   │   ├── state.py                    # Shared state definitions
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   └── ingestion_agent.py
│   │   ├── etl/
│   │   │   ├── __init__.py
│   │   │   └── etl_agent.py
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval_agent.py
│   │   │   ├── query_parser.py
│   │   │   └── fusion.py               # RRF implementation
│   │   ├── graph/
│   │   │   ├── __init__.py
│   │   │   ├── graph_agent.py
│   │   │   ├── clustering.py
│   │   │   └── flow_analysis.py
│   │   ├── anomaly/
│   │   │   ├── __init__.py
│   │   │   ├── anomaly_agent.py
│   │   │   ├── detectors/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── volume_spike.py
│   │   │   │   ├── timing_correlation.py
│   │   │   │   └── coordinated_activity.py
│   │   │   └── scoring.py
│   │   ├── validator/
│   │   │   ├── __init__.py
│   │   │   ├── validator_agent.py
│   │   │   ├── citation_extractor.py
│   │   │   └── verifier.py
│   │   ├── report/
│   │   │   ├── __init__.py
│   │   │   ├── report_agent.py
│   │   │   └── templates.py
│   │   └── supervisor/
│   │       ├── __init__.py
│   │       └── supervisor_agent.py
│   │
│   ├── workflows/                      # LangGraph workflow definitions
│   │   ├── __init__.py
│   │   ├── investigation.py            # Main investigation workflow
│   │   ├── batch_analysis.py           # Batch processing workflow
│   │   └── alert_workflow.py           # Alert generation workflow
│   │
│   ├── db/                             # Database clients and utilities
│   │   ├── __init__.py
│   │   ├── clickhouse/
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   ├── queries.py              # Pre-built query templates
│   │   │   └── schema.sql              # DDL statements
│   │   ├── qdrant/
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   └── collections.py          # Collection schemas
│   │   ├── neo4j/
│   │   │   ├── __init__.py
│   │   │   ├── client.py
│   │   │   └── queries.py              # Cypher templates
│   │   └── redis/
│   │       ├── __init__.py
│   │       └── client.py
│   │
│   ├── ingestion/                      # Data ingestion modules
│   │   ├── __init__.py
│   │   ├── ethereum_rpc.py             # Ethereum RPC client
│   │   ├── cryptohouse.py              # CryptoHouse integration
│   │   ├── event_feeds/                # External event sources
│   │   │   ├── __init__.py
│   │   │   ├── listings.py             # CEX listing feeds
│   │   │   └── announcements.py        # Project announcements
│   │   └── loaders/
│   │       ├── __init__.py
│   │       ├── clickhouse_loader.py
│   │       ├── qdrant_loader.py
│   │       └── neo4j_loader.py
│   │
│   ├── embeddings/                     # Embedding generation
│   │   ├── __init__.py
│   │   ├── model.py                    # BGE model wrapper
│   │   ├── patterns.py                 # Pattern text formatters
│   │   └── batch_processor.py          # Batch embedding generation
│   │
│   ├── llm/                            # LLM integration
│   │   ├── __init__.py
│   │   ├── client.py                   # vLLM/Ollama client
│   │   ├── prompts.py                  # Prompt management
│   │   └── structured_output.py        # Output parsing
│   │
│   ├── api/                            # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app initialization
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── investigations.py       # Investigation endpoints
│   │   │   ├── queries.py              # Ad-hoc query endpoints
│   │   │   ├── alerts.py               # Alert management
│   │   │   └── health.py               # Health checks
│   │   ├── schemas/                    # Pydantic request/response models
│   │   │   ├── __init__.py
│   │   │   ├── investigation.py
│   │   │   ├── query.py
│   │   │   └── report.py
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                 # JWT authentication
│   │   │   ├── rate_limit.py           # Rate limiting
│   │   │   └── logging.py              # Request logging
│   │   └── websocket/
│   │       ├── __init__.py
│   │       └── streaming.py            # WebSocket handlers
│   │
│   └── utils/                          # Shared utilities
│       ├── __init__.py
│       ├── address.py                  # Address utilities (checksum, etc.)
│       ├── time.py                     # Time/timestamp utilities
│       ├── crypto.py                   # Hashing, checksums
│       ├── metrics.py                  # Prometheus metrics
│       └── logging.py                  # Structured logging
│
├── etl/                                # ETL pipeline code
│   ├── __init__.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── transactions.py
│   │   ├── token_transfers.py
│   │   └── events.py
│   ├── transformers/
│   │   ├── __init__.py
│   │   ├── normalize.py
│   │   ├── enrich.py
│   │   └── deduplicate.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── clickhouse.py
│   │   ├── qdrant.py
│   │   └── neo4j.py
│   └── pipelines/
│       ├── __init__.py
│       ├── incremental.py
│       ├── backfill.py
│       └── graph_sync.py
│
├── dags/                               # Airflow DAG definitions
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── ethereum_incremental.py
│   │   ├── ethereum_backfill.py
│   │   └── event_ingestion.py
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── embedding_generation.py
│   │   ├── graph_sync.py
│   │   └── aggregate_refresh.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py
│   │   └── alert_generation.py
│   └── maintenance/
│       ├── __init__.py
│       ├── data_retention.py
│       └── consistency_check.py
│
├── tests/                              # Test suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── agents/
│   │   │   ├── test_retrieval_agent.py
│   │   │   ├── test_graph_agent.py
│   │   │   ├── test_anomaly_agent.py
│   │   │   └── test_validator_agent.py
│   │   ├── db/
│   │   │   ├── test_clickhouse.py
│   │   │   ├── test_qdrant.py
│   │   │   └── test_neo4j.py
│   │   ├── embeddings/
│   │   │   └── test_patterns.py
│   │   └── utils/
│   │       ├── test_address.py
│   │       └── test_citation_extractor.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_workflow.py
│   │   ├── test_hybrid_search.py
│   │   └── test_validation_loop.py
│   ├── e2e/
│   │   ├── __init__.py
│   │   ├── test_investigation_flow.py
│   │   └── test_known_cases.py
│   └── fixtures/
│       ├── transactions.json
│       ├── token_transfers.json
│       └── known_insider_cases.json
│
├── scripts/                            # Utility scripts
│   ├── setup_databases.sh
│   ├── load_sample_data.py
│   ├── run_backfill.py
│   ├── benchmark_search.py
│   └── validate_known_cases.py
│
├── docker/                             # Docker configurations
│   ├── Dockerfile                      # Main application Dockerfile
│   ├── Dockerfile.airflow              # Airflow worker Dockerfile
│   ├── docker-compose.yml              # Local development stack
│   ├── docker-compose.prod.yml         # Production overrides
│   └── init/
│       ├── clickhouse/
│       │   └── init.sql
│       ├── neo4j/
│       │   └── init.cypher
│       └── qdrant/
│           └── init.json
│
├── k8s/                                # Kubernetes manifests
│   ├── base/
│   │   ├── kustomization.yaml
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   └── secrets.yaml
│   ├── apps/
│   │   ├── api/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   └── hpa.yaml
│   │   ├── workers/
│   │   │   ├── deployment.yaml
│   │   │   └── hpa.yaml
│   │   └── airflow/
│   │       ├── scheduler.yaml
│   │       ├── webserver.yaml
│   │       └── workers.yaml
│   ├── databases/
│   │   ├── clickhouse/
│   │   │   ├── statefulset.yaml
│   │   │   ├── service.yaml
│   │   │   └── pvc.yaml
│   │   ├── qdrant/
│   │   │   ├── statefulset.yaml
│   │   │   ├── service.yaml
│   │   │   └── pvc.yaml
│   │   └── neo4j/
│   │       ├── statefulset.yaml
│   │       ├── service.yaml
│   │       └── pvc.yaml
│   ├── llm-inference/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── gpu-nodepool.yaml
│   ├── monitoring/
│   │   ├── prometheus/
│   │   │   ├── deployment.yaml
│   │   │   ├── config.yaml
│   │   │   └── alerts.yaml
│   │   ├── grafana/
│   │   │   ├── deployment.yaml
│   │   │   └── dashboards/
│   │   └── loki/
│   │       └── deployment.yaml
│   └── overlays/
│       ├── dev/
│       │   └── kustomization.yaml
│       ├── staging/
│       │   └── kustomization.yaml
│       └── prod/
│           └── kustomization.yaml
│
├── helm/                               # Helm charts (alternative to raw k8s)
│   └── insider-detection/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-dev.yaml
│       ├── values-prod.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           └── ingress.yaml
│
├── docs/                               # Documentation
│   ├── architecture/
│   │   ├── ARCHITECTURE.md
│   │   ├── TECHNOLOGY_STACK.md
│   │   ├── MULTI_AGENT_ARCHITECTURE.md
│   │   ├── IMPLEMENTATION_ROADMAP.md
│   │   ├── VALIDATION_HALLUCINATION_MITIGATION.md
│   │   └── FILE_STRUCTURE.md
│   ├── api/
│   │   └── openapi.yaml
│   ├── runbooks/
│   │   ├── deployment.md
│   │   ├── incident_response.md
│   │   └── data_recovery.md
│   └── adrs/                           # Architecture Decision Records
│       ├── 001-vector-db-selection.md
│       ├── 002-llm-choice.md
│       └── 003-agent-coordination.md
│
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_pattern_analysis.ipynb
│   ├── 03_embedding_evaluation.ipynb
│   └── 04_known_case_validation.ipynb
│
└── .github/                            # GitHub configurations
    ├── workflows/
    │   ├── ci.yaml                     # CI pipeline
    │   ├── cd.yaml                     # CD pipeline
    │   └── security.yaml               # Security scanning
    ├── CODEOWNERS
    └── pull_request_template.md
```

---

## Directory Purposes

### `/config`
Central configuration management using Pydantic Settings. All environment-specific values, feature flags, and tunable parameters live here.

### `/src/agents`
Multi-agent implementations following the LangGraph pattern. Each agent directory contains:
- Main agent class
- Agent-specific tools
- Helper utilities

### `/src/workflows`
LangGraph StateGraph definitions that coordinate agents. Workflows define the execution flow, conditional branching, and checkpointing.

### `/src/db`
Database client wrappers providing:
- Connection pooling
- Query builders
- Schema definitions
- Common query templates

### `/src/ingestion`
Data ingestion from external sources:
- Ethereum RPC integration
- CryptoHouse connector
- External event feeds (listings, announcements)

### `/etl`
ETL pipeline components following Extract-Transform-Load pattern. Designed for both batch and streaming processing.

### `/dags`
Apache Airflow DAG definitions. Organized by function:
- Ingestion DAGs
- ETL DAGs
- Analysis DAGs
- Maintenance DAGs

### `/tests`
Comprehensive test suite:
- **Unit tests**: Individual component testing
- **Integration tests**: Cross-component interaction
- **E2E tests**: Full workflow validation
- **Fixtures**: Test data and known case examples

### `/docker`
Containerization configurations for local development and production deployment.

### `/k8s`
Kubernetes manifests using Kustomize for environment overlays. Includes:
- Application deployments
- Database StatefulSets
- Monitoring stack
- GPU inference nodes

### `/docs`
Technical documentation including architecture documents, API specs, runbooks, and ADRs.
