# Technology Stack - Detailed Analysis & Recommendations

## Overview

This document provides in-depth justification for each technology choice, analyzes alternatives, and makes final recommendations optimized for the insider detection use case.

---

## 1. Vector Database (Hybrid Search)

### Baseline: Weaviate or Qdrant

| Criteria | Qdrant | Weaviate | Milvus |
|----------|--------|----------|--------|
| **Hybrid Search** | Native (sparse+dense) | Native (BM25+vector) | Requires config |
| **Filtering Performance** | Excellent (HNSW+payload) | Good | Good |
| **Scalability** | Horizontal sharding | Horizontal | Horizontal |
| **Memory Efficiency** | Quantization support | Moderate | Quantization |
| **API Simplicity** | Excellent (gRPC+REST) | GraphQL-heavy | gRPC |
| **Self-hosted Ease** | Docker-native | Docker-native | Complex |
| **Production Maturity** | High | High | High |

### **Recommendation: Qdrant**

**Justification:**
1. **Superior payload filtering**: Critical for metadata filtering (address, timestamp, token) combined with semantic search
2. **Efficient sparse-dense fusion**: Native hybrid search without external BM25 index
3. **Memory-mapped storage**: Handles large embedding collections (millions of transaction patterns)
4. **Simpler operational model**: No GraphQL complexity, direct REST/gRPC
5. **Quantization**: Scalar/binary quantization reduces memory 4-8x for large-scale deployment

**Configuration for Use Case:**
```yaml
# Optimal Qdrant settings for blockchain data
collections:
  transaction_patterns:
    vectors:
      size: 1024  # BGE-large output dimension
      distance: Cosine
    optimizers:
      indexing_threshold: 20000
    quantization:
      scalar:
        type: int8
        quantile: 0.99
```

---

## 2. Graph Database

### Baseline: Neo4j

| Criteria | Neo4j | Amazon Neptune | TigerGraph | ArangoDB |
|----------|-------|----------------|------------|----------|
| **Cypher Support** | Native | OpenCypher | GSQL | AQL |
| **Path Analysis** | Excellent | Good | Excellent | Good |
| **APOC Procedures** | 450+ functions | Limited | Native | Limited |
| **Clustering Algos** | GDS Library | Basic | Advanced | Basic |
| **Visualization** | Neo4j Bloom | Limited | GraphStudio | Built-in |
| **Self-hosted** | Community/Enterprise | Managed only | Enterprise | Open source |

### **Recommendation: Neo4j with Graph Data Science (GDS)**

**Justification:**
1. **Wallet clustering**: GDS provides Louvain, Label Propagation, Weakly Connected Components - essential for identifying wallet clusters controlled by same entity
2. **Path finding**: Shortest path, all paths between addresses - critical for tracing fund flows
3. **Centrality analysis**: PageRank, Betweenness - identifies key intermediary wallets
4. **Community detection**: Discovers hidden relationships between seemingly unrelated addresses
5. **APOC library**: Time-series windowing, data import/export, triggers

**Schema Design for Blockchain:**
```cypher
// Core node types
(:Wallet {address: string, first_seen: datetime, label: string})
(:Token {address: string, symbol: string, name: string})
(:Transaction {hash: string, block: int, timestamp: datetime})
(:Event {type: string, timestamp: datetime, description: string})

// Core relationships
(:Wallet)-[:SENT {value: float, token: string, tx_hash: string}]->(:Wallet)
(:Wallet)-[:HOLDS {balance: float, as_of: datetime}]->(:Token)
(:Wallet)-[:PARTICIPATED_IN]->(:Transaction)
(:Wallet)-[:ASSOCIATED_WITH]->(:Event)  // e.g., listing, partnership
```

---

## 3. OLAP Storage

### Baseline: ClickHouse

| Criteria | ClickHouse | Apache Druid | TimescaleDB | DuckDB |
|----------|------------|--------------|-------------|--------|
| **Columnar Storage** | Native | Native | Extension | Native |
| **Compression** | 10-40x | 5-10x | 3-5x | 5-10x |
| **Aggregation Speed** | Fastest | Very fast | Fast | Fast (single-node) |
| **Time-series** | Excellent | Excellent | Excellent | Good |
| **SQL Compatibility** | High | Limited | PostgreSQL | Full |
| **Ethereum Datasets** | CryptoHouse ready | Manual | Manual | Manual |
| **Horizontal Scale** | Native sharding | Native | Limited | Single-node |

### **Recommendation: ClickHouse**

**Justification:**
1. **CryptoHouse integration**: Pre-built Ethereum dataset with optimized schema - dramatically accelerates development
2. **Compression**: 20-40x compression on blockchain data reduces storage costs significantly
3. **Materialized views**: Pre-aggregate common patterns (daily volumes, address activity)
4. **Array functions**: Native support for analyzing token transfer arrays, event logs
5. **Window functions**: Time-based analysis critical for detecting pre-announcement activity

**CryptoHouse Schema (Pre-built):**
```sql
-- Available tables in CryptoHouse
-- ethereum.transactions - All transactions with decoded input
-- ethereum.traces - Internal transactions
-- ethereum.logs - Event logs
-- ethereum.blocks - Block metadata
-- ethereum.token_transfers - ERC20/721/1155 transfers (derived)

-- Example: Pre-announcement accumulation detection
SELECT 
    to_address,
    toDate(block_timestamp) as date,
    sum(value) as daily_volume,
    count() as tx_count
FROM ethereum.token_transfers
WHERE token_address = '0x...'
  AND block_timestamp BETWEEN event_date - INTERVAL 30 DAY AND event_date
GROUP BY to_address, date
HAVING daily_volume > threshold
ORDER BY daily_volume DESC
```

---

## 4. Pipeline Orchestration

### Baseline: Apache Airflow or Prefect

| Criteria | Apache Airflow | Prefect | Dagster | Temporal |
|----------|----------------|---------|---------|----------|
| **DAG Definition** | Python | Python | Python | Code/YAML |
| **UI/Monitoring** | Excellent | Good | Excellent | Good |
| **Dynamic Tasks** | TaskFlow API | Native | Native | Native |
| **Backfill Support** | Excellent | Good | Excellent | Manual |
| **Community/Plugins** | Largest | Growing | Growing | Moderate |
| **Kubernetes Native** | KubernetesExecutor | K8s Agent | K8s | Native |
| **Data Lineage** | Datasets (2.4+) | Limited | Native | N/A |

### **Recommendation: Apache Airflow 2.8+**

**Justification:**
1. **Maturity**: Battle-tested for ETL at scale, extensive documentation
2. **ClickHouse provider**: Official `apache-airflow-providers-clickhouse`
3. **Dataset-aware scheduling**: Trigger downstream DAGs when data lands
4. **KubernetesExecutor**: Scale workers dynamically based on load
5. **Backfill**: Critical for reprocessing historical blockchain data with new detection logic

**DAG Structure:**
```
dags/
├── ingestion/
│   ├── ethereum_blocks_incremental.py    # Every 12 seconds
│   ├── ethereum_backfill.py              # Historical data
│   └── event_feed_ingestion.py           # Listings, announcements
├── etl/
│   ├── compute_embeddings.py             # Generate pattern embeddings
│   ├── build_graph.py                    # Sync Neo4j from ClickHouse
│   └── aggregate_metrics.py              # Materialized views refresh
├── analysis/
│   ├── anomaly_detection_daily.py        # Scheduled detection runs
│   └── alert_generation.py               # Generate and send alerts
└── maintenance/
    ├── data_retention.py                 # Cleanup old data
    └── consistency_check.py              # Cross-DB validation
```

---

## 5. LLM Orchestration Framework

### Baseline: LangChain/LangGraph

| Criteria | LangGraph | LangChain | LlamaIndex | Haystack | CrewAI |
|----------|-----------|-----------|------------|----------|--------|
| **Multi-Agent** | Native (StateGraph) | Agents | Agents | Pipelines | Native |
| **State Management** | Checkpointing | Memory | Memory | Limited | Limited |
| **Cycles/Loops** | Native | Workarounds | Limited | Linear | Native |
| **Tool Calling** | Structured | Structured | Limited | Structured | High-level |
| **Streaming** | Native | Native | Native | Native | Limited |
| **Observability** | LangSmith | LangSmith | Callbacks | Limited | Limited |

### **Recommendation: LangGraph**

**Justification:**
1. **Stateful agents**: Maintains context across multi-step investigation
2. **Conditional branching**: Route to different analysis paths based on findings
3. **Human-in-the-loop**: Interrupt for approval before generating final reports
4. **Checkpointing**: Resume long investigations, audit agent decisions
5. **Parallel branches**: Run semantic search, graph query, OLAP query simultaneously

**Agent Graph Structure:**
```python
from langgraph.graph import StateGraph, END

# Define the investigation workflow
workflow = StateGraph(InvestigationState)

# Add nodes (agents)
workflow.add_node("query_parser", query_parser_agent)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("graph_analysis", graph_agent)
workflow.add_node("anomaly_detection", anomaly_agent)
workflow.add_node("validator", validator_agent)
workflow.add_node("report_generator", report_agent)

# Define edges (flow)
workflow.add_edge("query_parser", "retrieval")
workflow.add_conditional_edges(
    "retrieval",
    route_by_evidence_type,
    {
        "graph_needed": "graph_analysis",
        "sufficient": "anomaly_detection"
    }
)
workflow.add_edge("graph_analysis", "anomaly_detection")
workflow.add_edge("anomaly_detection", "validator")
workflow.add_conditional_edges(
    "validator",
    check_validation_result,
    {
        "valid": "report_generator",
        "invalid": "retrieval"  # Retry with stricter constraints
    }
)
workflow.add_edge("report_generator", END)
```

---

## 6. Embedding Model

### Baseline: BAAI/bge-large-en-v1.5

| Model | Dimensions | MTEB Score | Speed | Memory |
|-------|------------|------------|-------|--------|
| **bge-large-en-v1.5** | 1024 | 64.23 | Moderate | 1.3GB |
| bge-m3 | 1024 | 66.1 | Slow | 2.2GB |
| e5-large-v2 | 1024 | 62.4 | Moderate | 1.3GB |
| gte-large | 1024 | 63.1 | Moderate | 1.3GB |
| nomic-embed-text-v1.5 | 768 | 62.3 | Fast | 0.5GB |

### **Recommendation: BAAI/bge-large-en-v1.5** (confirmed)

**Justification:**
1. **Instruction-tuned**: Supports query prefixes for retrieval optimization
2. **Strong on technical text**: Performs well on structured/semi-structured data
3. **Balanced tradeoff**: Good accuracy without excessive compute requirements
4. **Matrikine normalization**: L2-normalized outputs work well with cosine similarity

**Alternative for Scale: nomic-embed-text-v1.5**
- If embedding millions of transactions, 768-dim reduces storage 25%
- Marginal accuracy loss acceptable for first-pass retrieval

**Usage Pattern:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# For queries (retrieval)
query_embedding = model.encode(
    "Represent this query for retrieving relevant documents: " + query,
    normalize_embeddings=True
)

# For documents (indexing)
doc_embedding = model.encode(
    document_text,
    normalize_embeddings=True
)
```

---

## 7. Large Language Model

### Baseline: Qwen3-30B-A3B-Thinking-2507

| Model | Parameters | Context | Reasoning | Self-hosted | Cost |
|-------|------------|---------|-----------|-------------|------|
| **Qwen3-30B-A3B** | 30B (3B active) | 128K | Excellent | Yes (vLLM) | GPU only |
| Qwen2.5-72B | 72B | 128K | Excellent | Heavy | High |
| Llama-3.1-70B | 70B | 128K | Very good | Heavy | High |
| Mixtral-8x22B | 141B (39B active) | 64K | Good | Moderate | Moderate |
| DeepSeek-V3 | 671B MoE | 128K | Excellent | Very heavy | API |

### **Recommendation: Qwen3-30B-A3B-Thinking-2507**

**Justification:**
1. **Mixture-of-Experts efficiency**: Only 3B parameters active per token - fast inference
2. **Extended thinking**: "Thinking" variant excels at multi-step reasoning required for pattern analysis
3. **128K context**: Can ingest substantial evidence without truncation
4. **Self-hosted**: Full control over data, no external API calls for sensitive financial analysis
5. **Structured output**: Strong JSON mode for citation extraction

**Deployment Options:**

```yaml
# Option 1: vLLM (Recommended for production)
# Requires: 2x A100 80GB or 4x A10G 24GB
docker run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --enable-prefix-caching

# Option 2: Ollama (Development/smaller scale)
ollama pull qwen3:30b-a3b
```

**Prompt Engineering for Citation Compliance:**
```python
SYSTEM_PROMPT = """You are a blockchain forensics analyst. Your analysis must be:

1. GROUNDED: Only state facts present in the provided EVIDENCE section
2. CITED: Every factual claim must include a citation in format:
   - Transaction: [TX:0x...]
   - Address: [ADDR:0x...]
   - Block: [BLOCK:12345678]
   - Timestamp: [TS:2024-01-15T10:30:00Z]
3. UNCERTAIN: If evidence is insufficient, explicitly state uncertainty
4. STRUCTURED: Respond in the requested JSON format

You will be penalized for any claim without a citation.
You will be penalized for any citation not present in EVIDENCE."""
```

---

## 8. Additional Infrastructure Components

### 8.1 API Framework

**Recommendation: FastAPI**
- Async-native for non-blocking DB queries
- Automatic OpenAPI documentation
- Pydantic validation for request/response schemas
- WebSocket support for streaming analysis results

### 8.2 Task Queue (Long-running Analysis)

**Recommendation: Celery with Redis**
- Distributed task execution
- Result backend for async query status
- Priority queues for urgent investigations

### 8.3 Caching Layer

**Recommendation: Redis**
- LLM response caching (identical queries)
- Embedding cache for frequently queried addresses
- Rate limiting for API endpoints

### 8.4 Containerization & Orchestration

**Recommendation: Docker + Kubernetes**
- Helm charts for reproducible deployments
- Horizontal Pod Autoscaler for agents
- GPU node pools for LLM inference

---

## 9. Final Technology Stack Summary

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Vector DB** | Qdrant | 1.9+ | Hybrid search, pattern matching |
| **Graph DB** | Neo4j + GDS | 5.x + 2.x | Wallet clustering, flow analysis |
| **OLAP** | ClickHouse | 24.x | Time-series, aggregations |
| **Orchestration** | Apache Airflow | 2.8+ | Pipeline scheduling |
| **LLM Framework** | LangGraph | 0.2+ | Multi-agent coordination |
| **Embedding** | bge-large-en-v1.5 | - | Semantic encoding |
| **LLM** | Qwen3-30B-A3B | 2507 | Analysis, report generation |
| **API** | FastAPI | 0.110+ | REST/WebSocket interface |
| **Task Queue** | Celery + Redis | 5.x + 7.x | Async job processing |
| **Cache** | Redis | 7.x | Response/embedding cache |
| **Container** | Docker + K8s | 24.x + 1.29 | Deployment |
| **Monitoring** | Prometheus + Grafana | - | Metrics and dashboards |
| **Logging** | Loki | - | Log aggregation |
| **Tracing** | OpenTelemetry + Jaeger | - | Distributed tracing |
