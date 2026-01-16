# Implementation Roadmap

## Overview

This document provides a phased implementation plan for the Blockchain Insider Detection System. Each phase builds upon the previous, with clear deliverables and validation criteria.

**Total Estimated Timeline:** 16-20 weeks for MVP, additional 8-12 weeks for production hardening.

---

## Phase 1: Foundation & Data Infrastructure (Weeks 1-4)

### 1.1 Environment Setup (Week 1)

**Objectives:**
- Establish development environment
- Configure infrastructure-as-code
- Set up CI/CD pipeline

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Create project repository structure | P0 | 2h | DevOps |
| Docker Compose for local development | P0 | 4h | DevOps |
| Kubernetes manifests (dev namespace) | P1 | 8h | DevOps |
| GitHub Actions CI pipeline | P0 | 4h | DevOps |
| Pre-commit hooks (ruff, mypy, pytest) | P1 | 2h | DevOps |
| Secrets management setup (Vault/env) | P0 | 4h | DevOps |

**Deliverables:**
- [ ] Working `docker-compose.yml` with all databases
- [ ] CI pipeline running tests on PR
- [ ] Development environment documentation

**Validation:**
```bash
# All containers healthy
docker-compose up -d
docker-compose ps  # All services "Up"

# CI passes
git push origin feature/setup
# GitHub Actions: ✓ lint ✓ type-check ✓ test
```

---

### 1.2 ClickHouse Setup & Historical Data Load (Weeks 1-2)

**Objectives:**
- Deploy ClickHouse with optimized schema
- Load historical Ethereum data via CryptoHouse
- Create derived materialized views

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| ClickHouse deployment (single-node dev) | P0 | 4h | Data Eng |
| Schema design and DDL scripts | P0 | 8h | Data Eng |
| CryptoHouse connection setup | P0 | 4h | Data Eng |
| Historical data backfill (2 years) | P0 | 16h | Data Eng |
| Materialized views for aggregates | P1 | 8h | Data Eng |
| Data quality validation queries | P1 | 4h | Data Eng |

**Schema (Core Tables):**
```sql
-- Raw transactions (from CryptoHouse or RPC)
CREATE TABLE ethereum.transactions
(
    hash String,
    block_number UInt64,
    block_timestamp DateTime64(3),
    from_address String,
    to_address String,
    value UInt256,
    gas_used UInt64,
    gas_price UInt64,
    input String,
    status UInt8,
    INDEX idx_from from_address TYPE bloom_filter GRANULARITY 4,
    INDEX idx_to to_address TYPE bloom_filter GRANULARITY 4,
    INDEX idx_block block_number TYPE minmax GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(block_timestamp)
ORDER BY (block_timestamp, hash);

-- Token transfers (derived)
CREATE TABLE ethereum.token_transfers
(
    tx_hash String,
    log_index UInt32,
    block_number UInt64,
    block_timestamp DateTime64(3),
    token_address String,
    from_address String,
    to_address String,
    value UInt256,
    token_type Enum8('ERC20' = 1, 'ERC721' = 2, 'ERC1155' = 3),
    INDEX idx_token token_address TYPE bloom_filter GRANULARITY 4,
    INDEX idx_from from_address TYPE bloom_filter GRANULARITY 4,
    INDEX idx_to to_address TYPE bloom_filter GRANULARITY 4
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(block_timestamp)
ORDER BY (token_address, block_timestamp, tx_hash);

-- Daily address activity (materialized view)
CREATE MATERIALIZED VIEW ethereum.daily_address_activity
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (address, date)
AS SELECT
    from_address AS address,
    toDate(block_timestamp) AS date,
    count() AS tx_count,
    sum(value) AS total_value_sent,
    uniqExact(to_address) AS unique_recipients
FROM ethereum.transactions
GROUP BY address, date;
```

**Validation:**
```sql
-- Verify data loaded
SELECT count() FROM ethereum.transactions;  -- Expected: > 1B
SELECT min(block_timestamp), max(block_timestamp) FROM ethereum.transactions;

-- Verify materialized view
SELECT * FROM ethereum.daily_address_activity 
WHERE address = '0x...' 
ORDER BY date DESC 
LIMIT 10;
```

---

### 1.3 Neo4j Setup & Graph Construction (Weeks 2-3)

**Objectives:**
- Deploy Neo4j with GDS plugin
- Design and implement graph schema
- Build initial graph from ClickHouse data

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Neo4j deployment with GDS | P0 | 4h | Data Eng |
| Graph schema design | P0 | 4h | Data Eng |
| ETL: ClickHouse → Neo4j pipeline | P0 | 16h | Data Eng |
| GDS graph projection setup | P1 | 4h | Data Eng |
| Initial clustering run (Louvain) | P1 | 4h | Data Eng |

**Graph Construction Pipeline:**
```python
# etl/neo4j_builder.py
async def build_wallet_graph(start_block: int, end_block: int):
    """Build Neo4j graph from ClickHouse token transfers."""
    
    # 1. Extract unique wallets
    wallets_query = """
    SELECT DISTINCT address FROM (
        SELECT from_address AS address FROM ethereum.token_transfers
        WHERE block_number BETWEEN %(start)s AND %(end)s
        UNION ALL
        SELECT to_address AS address FROM ethereum.token_transfers
        WHERE block_number BETWEEN %(start)s AND %(end)s
    )
    """
    
    # 2. Extract transfer relationships
    transfers_query = """
    SELECT 
        from_address,
        to_address,
        token_address,
        sum(value) AS total_value,
        count() AS tx_count,
        min(block_timestamp) AS first_tx,
        max(block_timestamp) AS last_tx
    FROM ethereum.token_transfers
    WHERE block_number BETWEEN %(start)s AND %(end)s
    GROUP BY from_address, to_address, token_address
    """
    
    # 3. Batch insert into Neo4j
    async with neo4j_driver.session() as session:
        # Create wallet nodes
        await session.run("""
            UNWIND $wallets AS wallet
            MERGE (w:Wallet {address: wallet.address})
            ON CREATE SET w.first_seen = wallet.first_seen
        """, wallets=wallet_records)
        
        # Create transfer relationships
        await session.run("""
            UNWIND $transfers AS t
            MATCH (from:Wallet {address: t.from_address})
            MATCH (to:Wallet {address: t.to_address})
            MERGE (from)-[r:TRANSFERRED {token: t.token_address}]->(to)
            ON CREATE SET r.total_value = t.total_value, r.tx_count = t.tx_count
            ON MATCH SET r.total_value = r.total_value + t.total_value,
                         r.tx_count = r.tx_count + t.tx_count
        """, transfers=transfer_records)
```

**Validation:**
```cypher
// Verify node counts
MATCH (w:Wallet) RETURN count(w) AS wallet_count;  // Expected: > 10M

// Verify relationship counts
MATCH ()-[r:TRANSFERRED]->() RETURN count(r) AS transfer_count;

// Test GDS projection
CALL gds.graph.project(
    'wallet-graph',
    'Wallet',
    'TRANSFERRED',
    {relationshipProperties: 'total_value'}
);

// Run sample Louvain
CALL gds.louvain.stream('wallet-graph')
YIELD nodeId, communityId
RETURN communityId, count(*) AS size
ORDER BY size DESC
LIMIT 10;
```

---

### 1.4 Qdrant Setup & Embedding Pipeline (Weeks 3-4)

**Objectives:**
- Deploy Qdrant with optimized configuration
- Implement embedding generation pipeline
- Build initial vector index

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Qdrant deployment | P0 | 4h | Data Eng |
| Collection schema design | P0 | 4h | ML Eng |
| Embedding model integration (BGE) | P0 | 8h | ML Eng |
| Pattern text generation logic | P0 | 8h | ML Eng |
| Batch embedding pipeline | P0 | 12h | ML Eng |
| Hybrid search testing | P1 | 4h | ML Eng |

**Pattern Text Generation:**
```python
# utils/embedding_utils.py
def format_transaction_pattern(tx_data: dict) -> str:
    """Convert transaction data to embeddable text pattern."""
    
    pattern_parts = []
    
    # Transaction type classification
    if tx_data.get("token_type") == "ERC20":
        pattern_parts.append(f"ERC20 token transfer of {tx_data['token_symbol']}")
    elif tx_data.get("is_contract_creation"):
        pattern_parts.append("Contract deployment")
    else:
        pattern_parts.append("ETH transfer")
    
    # Value classification
    value_usd = tx_data.get("value_usd", 0)
    if value_usd > 1_000_000:
        pattern_parts.append("whale transaction (>$1M)")
    elif value_usd > 100_000:
        pattern_parts.append("large transaction ($100K-$1M)")
    elif value_usd > 10_000:
        pattern_parts.append("medium transaction ($10K-$100K)")
    else:
        pattern_parts.append("small transaction (<$10K)")
    
    # Timing context
    if tx_data.get("hours_before_event"):
        pattern_parts.append(f"{tx_data['hours_before_event']} hours before {tx_data['event_type']}")
    
    # Address labels if known
    if tx_data.get("from_label"):
        pattern_parts.append(f"from {tx_data['from_label']}")
    if tx_data.get("to_label"):
        pattern_parts.append(f"to {tx_data['to_label']}")
    
    return ". ".join(pattern_parts)


# Example output:
# "ERC20 token transfer of PEPE. whale transaction (>$1M). 
#  48 hours before listing announcement. from unlabeled wallet. 
#  to Binance deposit address."
```

**Validation:**
```python
# Test embedding generation
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

# Verify collection
collection_info = client.get_collection("transaction_patterns")
assert collection_info.points_count > 1_000_000

# Test hybrid search
results = client.search(
    collection_name="transaction_patterns",
    query_vector=embedding_model.encode("accumulation before listing"),
    query_filter={
        "must": [
            {"key": "token_address", "match": {"value": "0x..."}}
        ]
    },
    limit=10
)
assert len(results) > 0
```

---

## Phase 2: Core Agent Implementation (Weeks 5-8)

### 2.1 LLM Infrastructure (Week 5)

**Objectives:**
- Deploy Qwen3-30B via vLLM
- Implement LangChain/LangGraph integration
- Create base agent tooling

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| vLLM deployment on GPU nodes | P0 | 8h | ML Eng |
| LangChain model wrapper | P0 | 4h | ML Eng |
| Tool calling validation | P0 | 4h | ML Eng |
| Prompt templates library | P0 | 8h | ML Eng |
| Response parsing utilities | P1 | 4h | ML Eng |

**vLLM Configuration:**
```yaml
# k8s/llm-inference/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-qwen3
  namespace: llm-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-qwen3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:v0.5.0
        args:
          - "--model"
          - "Qwen/Qwen3-30B-A3B-Thinking-2507"
          - "--tensor-parallel-size"
          - "2"
          - "--max-model-len"
          - "32768"
          - "--enable-prefix-caching"
          - "--gpu-memory-utilization"
          - "0.9"
        resources:
          limits:
            nvidia.com/gpu: 2
        ports:
          - containerPort: 8000
```

---

### 2.2 Retrieval Agent (Week 5-6)

**Objectives:**
- Implement hybrid search across all databases
- Build query parsing and intent classification
- Implement RRF fusion

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Query parser implementation | P0 | 8h | Backend |
| Qdrant search tool | P0 | 4h | Backend |
| Neo4j query tool | P0 | 4h | Backend |
| ClickHouse query tool | P0 | 4h | Backend |
| RRF fusion implementation | P0 | 4h | Backend |
| Integration tests | P1 | 8h | Backend |

**Implementation:**
```python
# src/agents/retrieval/retrieval_agent.py
from langchain_core.tools import tool
from typing import Annotated

class RetrievalAgent:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        neo4j_driver: AsyncDriver,
        clickhouse_client: Client,
        embedding_model: SentenceTransformer
    ):
        self.qdrant = qdrant_client
        self.neo4j = neo4j_driver
        self.clickhouse = clickhouse_client
        self.embedder = embedding_model
        
        self.tools = [
            self.semantic_search,
            self.graph_query,
            self.olap_query,
            self.fuse_results
        ]
    
    @tool
    async def semantic_search(
        self,
        query: Annotated[str, "Natural language search query"],
        filters: Annotated[dict | None, "Metadata filters"] = None,
        limit: Annotated[int, "Max results"] = 100
    ) -> list[dict]:
        """Search for similar transaction patterns using semantic similarity."""
        query_vector = self.embedder.encode(f"query: {query}", normalize_embeddings=True)
        
        qdrant_filter = None
        if filters:
            qdrant_filter = self._build_qdrant_filter(filters)
        
        results = self.qdrant.search(
            collection_name="transaction_patterns",
            query_vector=query_vector.tolist(),
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "score": hit.score,
                "payload": hit.payload,
                "source": "qdrant_semantic"
            }
            for hit in results
        ]
    
    @tool
    async def graph_query(
        self,
        cypher: Annotated[str, "Cypher query to execute"],
        params: Annotated[dict, "Query parameters"] = {}
    ) -> list[dict]:
        """Execute a Cypher query on the wallet graph."""
        async with self.neo4j.session() as session:
            result = await session.run(cypher, params)
            records = [record.data() async for record in result]
            return [
                {**r, "source": "neo4j_graph"}
                for r in records
            ]
    
    @tool
    async def olap_query(
        self,
        sql: Annotated[str, "SQL query to execute"],
        params: Annotated[dict, "Query parameters"] = {}
    ) -> list[dict]:
        """Execute an analytical SQL query on ClickHouse."""
        result = self.clickhouse.query(sql, parameters=params)
        return [
            {**dict(zip(result.column_names, row)), "source": "clickhouse_olap"}
            for row in result.result_rows
        ]
    
    def fuse_results(
        self,
        result_sets: list[list[dict]],
        k: int = 60
    ) -> list[dict]:
        """Fuse multiple result sets using Reciprocal Rank Fusion."""
        scores = {}
        
        for result_set in result_sets:
            for rank, item in enumerate(result_set):
                key = self._get_item_key(item)
                if key not in scores:
                    scores[key] = {"item": item, "rrf_score": 0}
                scores[key]["rrf_score"] += 1 / (k + rank + 1)
        
        fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [
            {**item["item"], "rrf_score": item["rrf_score"]}
            for item in fused
        ]
```

---

### 2.3 Graph Analysis Agent (Week 6-7)

**Objectives:**
- Implement wallet clustering algorithms
- Build fund flow tracing
- Create centrality analysis tools

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Wallet clustering (Louvain) | P0 | 8h | Backend |
| Fund flow tracing | P0 | 8h | Backend |
| Centrality metrics | P1 | 4h | Backend |
| Community detection | P1 | 4h | Backend |
| Mixer pattern detection | P2 | 8h | Backend |

---

### 2.4 Anomaly Detection Agent (Week 7-8)

**Objectives:**
- Implement statistical anomaly detection
- Build pattern matching for known insider behaviors
- Create scoring system

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Volume spike detection | P0 | 8h | ML Eng |
| Baseline computation | P0 | 8h | ML Eng |
| Timing correlation | P0 | 8h | ML Eng |
| Coordinated activity detection | P1 | 8h | ML Eng |
| Composite scoring | P0 | 4h | ML Eng |

---

### 2.5 Validator & Report Agents (Week 8)

**Objectives:**
- Implement citation extraction and verification
- Build report generation with mandatory citations

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Citation parser | P0 | 4h | Backend |
| Transaction verifier | P0 | 4h | Backend |
| Address verifier | P0 | 4h | Backend |
| Report generator | P0 | 8h | Backend |
| JSON schema validation | P1 | 4h | Backend |

---

## Phase 3: Pipeline Orchestration (Weeks 9-11)

### 3.1 Airflow Setup & DAG Implementation (Weeks 9-10)

**Objectives:**
- Deploy Airflow with KubernetesExecutor
- Implement ingestion DAGs
- Implement ETL DAGs

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Airflow deployment | P0 | 8h | Data Eng |
| Incremental ingestion DAG | P0 | 12h | Data Eng |
| Historical backfill DAG | P0 | 8h | Data Eng |
| Embedding generation DAG | P0 | 8h | Data Eng |
| Graph sync DAG | P0 | 8h | Data Eng |
| Anomaly detection DAG | P1 | 8h | Data Eng |

**DAG Example:**
```python
# dags/ingestion/ethereum_incremental.py
from airflow import DAG
from airflow.decorators import task
from airflow.providers.clickhouse.operators.clickhouse import ClickHouseOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': True,
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ethereum_incremental_ingestion',
    default_args=default_args,
    description='Incremental Ethereum block ingestion',
    schedule_interval='*/1 * * * *',  # Every minute
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['ingestion', 'ethereum'],
) as dag:

    @task
    def get_latest_block() -> int:
        """Get the latest processed block from ClickHouse."""
        from clickhouse_driver import Client
        client = Client('clickhouse')
        result = client.execute(
            "SELECT max(block_number) FROM ethereum.transactions"
        )
        return result[0][0] or 0

    @task
    def fetch_new_blocks(start_block: int) -> dict:
        """Fetch new blocks from RPC."""
        from src.ingestion.ethereum_rpc import EthereumRPC
        rpc = EthereumRPC()
        latest = rpc.get_latest_block_number()
        
        # Process in batches of 100 blocks
        end_block = min(start_block + 100, latest)
        
        blocks = rpc.get_blocks_range(start_block + 1, end_block)
        return {
            'blocks': blocks,
            'start': start_block + 1,
            'end': end_block
        }

    @task
    def insert_blocks(data: dict):
        """Insert blocks into ClickHouse."""
        from src.ingestion.clickhouse_loader import ClickHouseLoader
        loader = ClickHouseLoader()
        loader.insert_transactions(data['blocks'])
        return {'processed': data['end'] - data['start'] + 1}

    @task
    def trigger_downstream(result: dict):
        """Trigger downstream ETL tasks."""
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator
        # Dataset-based triggering handles this automatically in Airflow 2.4+
        pass

    latest_block = get_latest_block()
    new_blocks = fetch_new_blocks(latest_block)
    insert_result = insert_blocks(new_blocks)
    trigger_downstream(insert_result)
```

---

### 3.2 LangGraph Workflow Integration (Week 10-11)

**Objectives:**
- Integrate all agents into LangGraph workflow
- Implement checkpointing and error handling
- Add human-in-the-loop capabilities

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| StateGraph definition | P0 | 8h | Backend |
| Conditional routing | P0 | 4h | Backend |
| Checkpointing (PostgreSQL) | P0 | 4h | Backend |
| Error handling nodes | P0 | 4h | Backend |
| Human-in-the-loop | P1 | 8h | Backend |
| Workflow testing | P0 | 8h | Backend |

---

## Phase 4: API & Interface (Weeks 12-14)

### 4.1 FastAPI Backend (Week 12-13)

**Objectives:**
- Build REST API for investigations
- Implement WebSocket for streaming results
- Add authentication and rate limiting

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| FastAPI project setup | P0 | 4h | Backend |
| Investigation endpoints | P0 | 12h | Backend |
| WebSocket streaming | P1 | 8h | Backend |
| OAuth2/JWT authentication | P0 | 8h | Backend |
| Rate limiting | P1 | 4h | Backend |
| OpenAPI documentation | P1 | 4h | Backend |

---

### 4.2 Monitoring & Observability (Week 13-14)

**Objectives:**
- Deploy Prometheus + Grafana
- Implement distributed tracing
- Set up alerting

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Prometheus deployment | P0 | 4h | DevOps |
| Application metrics | P0 | 8h | Backend |
| Grafana dashboards | P1 | 8h | DevOps |
| OpenTelemetry integration | P1 | 8h | Backend |
| Jaeger deployment | P2 | 4h | DevOps |
| PagerDuty/Slack alerts | P1 | 4h | DevOps |

---

## Phase 5: Testing & Validation (Weeks 15-16)

### 5.1 Test Suite Development

**Objectives:**
- Comprehensive unit tests
- Integration tests for agent workflows
- End-to-end system tests

**Tasks:**
| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| Unit tests (>80% coverage) | P0 | 16h | All |
| Integration tests | P0 | 16h | All |
| E2E workflow tests | P0 | 12h | QA |
| Performance benchmarks | P1 | 8h | QA |
| Chaos testing | P2 | 8h | QA |

---

### 5.2 Validation with Known Cases

**Objectives:**
- Validate system against historical insider cases
- Tune detection parameters
- Document accuracy metrics

**Test Cases:**
1. **$PEPE Listing (Apr 2023)**: Known pre-listing accumulation
2. **$ARB Airdrop (Mar 2023)**: Sybil wallet clusters
3. **FTX Collapse (Nov 2022)**: Insider withdrawals
4. **Terra/LUNA (May 2022)**: Pre-depeg movements

---

## Phase 6: Production Hardening (Weeks 17-20+)

### 6.1 Security Audit
- Penetration testing
- Code security review
- Secrets rotation procedures

### 6.2 Performance Optimization
- Query optimization
- Caching tuning
- Load testing at 10x expected volume

### 6.3 Documentation
- API documentation
- Runbooks for operations
- Architecture decision records (ADRs)

### 6.4 Launch Preparation
- Staged rollout plan
- Rollback procedures
- On-call rotation setup

---

## Milestone Summary

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| **M1: Data Infrastructure** | 4 | All DBs deployed, historical data loaded |
| **M2: Core Agents** | 8 | All agents implemented and unit tested |
| **M3: Orchestration** | 11 | Airflow + LangGraph workflows operational |
| **M4: API & Monitoring** | 14 | REST API live, dashboards available |
| **M5: Validated MVP** | 16 | System validated against known cases |
| **M6: Production Ready** | 20 | Security audit passed, performance validated |
