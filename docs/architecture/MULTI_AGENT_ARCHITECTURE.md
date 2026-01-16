# Multi-Agent Architecture Specification

## Overview

The system employs a multi-agent architecture where specialized agents handle distinct responsibilities. Agents are coordinated via LangGraph's StateGraph, enabling complex workflows with conditional branching, parallel execution, and human-in-the-loop checkpoints.

---

## 1. Agent Catalog

### 1.1 Ingestion Agent

**Purpose:** Acquire raw blockchain data from external sources and normalize it for storage.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | Scheduled (Airflow) or real-time (WebSocket) |
| **Input** | Block range, RPC endpoint, data source config |
| **Output** | Normalized records in ClickHouse staging tables |
| **Tools** | `ethereum_rpc_client`, `cryptohouse_client`, `schema_validator` |
| **State** | Stateless (idempotent per block range) |

**Responsibilities:**
1. Connect to Ethereum RPC (Alchemy/Infura/QuickNode) for real-time blocks
2. Query CryptoHouse for historical backfill
3. Validate schema conformance before insertion
4. Handle reorgs by checking block finality
5. Emit metrics (blocks processed, latency, errors)

**Tool Definitions:**
```python
@tool
def fetch_blocks(start_block: int, end_block: int, rpc_url: str) -> list[dict]:
    """Fetch blocks and transactions from Ethereum RPC."""
    ...

@tool  
def fetch_token_transfers(start_block: int, end_block: int) -> list[dict]:
    """Fetch ERC20/721/1155 transfers from event logs."""
    ...

@tool
def validate_and_insert(records: list[dict], table: str) -> InsertResult:
    """Validate schema and insert into ClickHouse staging."""
    ...
```

---

### 1.2 ETL Agent

**Purpose:** Transform raw data into analysis-ready formats across all three databases.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | Downstream of Ingestion (Airflow Dataset) |
| **Input** | ClickHouse staging tables with new data |
| **Output** | Enriched ClickHouse tables, Qdrant vectors, Neo4j nodes/edges |
| **Tools** | `embedding_generator`, `graph_builder`, `aggregation_runner` |
| **State** | Checkpointed (last processed block per pipeline) |

**Responsibilities:**
1. Generate embeddings for transaction patterns
2. Build/update wallet graph in Neo4j
3. Compute materialized aggregations (daily volumes, address metrics)
4. Maintain cross-database consistency checksums

**Data Flow:**
```
ClickHouse (raw)
      │
      ├──▶ Generate embeddings ──▶ Qdrant
      │
      ├──▶ Extract relationships ──▶ Neo4j
      │
      └──▶ Compute aggregates ──▶ ClickHouse (derived)
```

**Tool Definitions:**
```python
@tool
def generate_transaction_embedding(tx_data: dict) -> list[float]:
    """Generate BGE embedding for transaction pattern."""
    pattern_text = format_transaction_pattern(tx_data)
    return embedding_model.encode(pattern_text, normalize=True)

@tool
def upsert_wallet_graph(transfers: list[dict]) -> GraphUpdateResult:
    """Update Neo4j graph with new wallet relationships."""
    ...

@tool
def refresh_materialized_views(views: list[str]) -> RefreshResult:
    """Refresh ClickHouse materialized views."""
    ...
```

---

### 1.3 Retrieval Agent

**Purpose:** Execute hybrid search across all databases to gather evidence for analysis.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | User query or scheduled analysis |
| **Input** | Natural language query or structured filter criteria |
| **Output** | Ranked, deduplicated evidence set with provenance |
| **Tools** | `semantic_search`, `metadata_filter`, `graph_query`, `olap_query` |
| **State** | Query context (for multi-turn investigations) |

**Responsibilities:**
1. Parse user intent and extract entities (addresses, tokens, time ranges)
2. Formulate parallel queries across Qdrant, Neo4j, ClickHouse
3. Apply Reciprocal Rank Fusion (RRF) to merge results
4. Attach full provenance (source DB, query, confidence) to each result

**Hybrid Search Strategy:**
```
User Query: "wallets accumulating TOKEN before listing on Jan 15"
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Semantic Search │  │ Metadata Filter │  │   OLAP Query    │
│    (Qdrant)     │  │   (ClickHouse)  │  │  (ClickHouse)   │
│                 │  │                 │  │                 │
│ "accumulation   │  │ token_address = │  │ SELECT address, │
│  pattern before │  │ 0x... AND       │  │ SUM(value)      │
│  major event"   │  │ timestamp <     │  │ WHERE ts <      │
│                 │  │ '2024-01-15'    │  │ listing_ts      │
│ k=100           │  │                 │  │ GROUP BY address│
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  RRF Fusion     │
                    │                 │
                    │  score = Σ 1/   │
                    │  (k + rank_i)   │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Top-N Results   │
                    │ with Provenance │
                    └─────────────────┘
```

**Tool Definitions:**
```python
@tool
def semantic_search(
    query: str,
    collection: str,
    filters: dict | None = None,
    limit: int = 100
) -> list[SearchResult]:
    """Search Qdrant with optional metadata filtering."""
    query_vector = embedding_model.encode(f"query: {query}")
    return qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        query_filter=build_qdrant_filter(filters),
        limit=limit,
        with_payload=True
    )

@tool
def graph_query(cypher: str, params: dict) -> list[dict]:
    """Execute Cypher query on Neo4j."""
    with neo4j_driver.session() as session:
        result = session.run(cypher, params)
        return [record.data() for record in result]

@tool
def olap_query(sql: str, params: dict) -> list[dict]:
    """Execute SQL query on ClickHouse."""
    return clickhouse_client.query(sql, params).named_results()
```

---

### 1.4 Graph Analysis Agent

**Purpose:** Perform specialized graph algorithms to uncover wallet clusters and flow patterns.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | Called by Retrieval Agent when graph analysis needed |
| **Input** | Seed addresses, analysis type (clustering, flow, centrality) |
| **Output** | Graph analysis results (clusters, paths, scores) |
| **Tools** | `wallet_clustering`, `flow_analysis`, `centrality_metrics`, `community_detection` |
| **State** | Cached cluster assignments (refreshed daily) |

**Responsibilities:**
1. Execute Neo4j GDS algorithms (Louvain, PageRank, shortest paths)
2. Identify wallet clusters likely controlled by same entity
3. Trace fund flows between addresses
4. Detect mixer/tumbler patterns

**Key Algorithms:**
```cypher
// Wallet Clustering (Louvain)
CALL gds.louvain.stream('wallet-graph', {
    nodeLabels: ['Wallet'],
    relationshipTypes: ['SENT'],
    relationshipWeightProperty: 'value'
})
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).address AS address, communityId
ORDER BY communityId

// Flow Analysis (All Paths)
MATCH path = (source:Wallet {address: $source_addr})-[:SENT*1..5]->(target:Wallet {address: $target_addr})
WHERE all(r IN relationships(path) WHERE r.timestamp >= $start_ts AND r.timestamp <= $end_ts)
RETURN path, reduce(total = 0, r IN relationships(path) | total + r.value) AS flow_value
ORDER BY flow_value DESC
LIMIT 10

// Centrality (PageRank)
CALL gds.pageRank.stream('wallet-graph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).address AS address, score
ORDER BY score DESC
LIMIT 100
```

**Tool Definitions:**
```python
@tool
def find_wallet_clusters(seed_addresses: list[str]) -> list[WalletCluster]:
    """Find clusters containing seed addresses using Louvain algorithm."""
    ...

@tool
def trace_fund_flow(
    source: str, 
    target: str, 
    max_hops: int = 5,
    time_range: tuple[datetime, datetime] | None = None
) -> list[FlowPath]:
    """Find all fund flow paths between two addresses."""
    ...

@tool
def identify_central_wallets(
    cluster_id: int,
    metric: Literal["pagerank", "betweenness", "degree"] = "pagerank"
) -> list[CentralityResult]:
    """Identify most central wallets in a cluster."""
    ...
```

---

### 1.5 Anomaly Detection Agent

**Purpose:** Identify statistically anomalous patterns indicating potential insider activity.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | After Retrieval Agent gathers evidence |
| **Input** | Retrieved evidence, detection parameters |
| **Output** | Scored anomalies with statistical significance |
| **Tools** | `time_series_anomaly`, `volume_spike_detector`, `pattern_matcher`, `baseline_comparator` |
| **State** | Baseline statistics (rolling 30-day windows) |

**Responsibilities:**
1. Detect volume spikes relative to historical baselines
2. Identify unusual accumulation patterns before events
3. Flag addresses with statistically significant behavior changes
4. Compute confidence scores for each anomaly

**Detection Patterns:**

```python
# Pattern 1: Pre-announcement Accumulation
"""
Detect wallets that significantly increased holdings of a token
in the 7-30 days before a major announcement (listing, partnership).
"""

# Pattern 2: Coordinated Movement
"""
Detect multiple wallets (potentially in same cluster) that
executed similar transactions within a short time window.
"""

# Pattern 3: Unusual Source of Funds
"""
Detect wallets receiving funds from known project/VC wallets
shortly before public announcements.
"""

# Pattern 4: Timing Correlation
"""
Detect transaction timing patterns that correlate with
non-public information (e.g., test transactions before launch).
"""
```

**Tool Definitions:**
```python
@tool
def detect_volume_anomaly(
    address: str,
    token: str,
    event_timestamp: datetime,
    lookback_days: int = 30,
    z_threshold: float = 3.0
) -> AnomalyResult:
    """Detect if address volume is anomalous relative to baseline."""
    baseline = get_baseline_stats(address, token, lookback_days)
    pre_event_volume = get_pre_event_volume(address, token, event_timestamp)
    z_score = (pre_event_volume - baseline.mean) / baseline.std
    return AnomalyResult(
        is_anomaly=abs(z_score) > z_threshold,
        z_score=z_score,
        baseline=baseline,
        observed=pre_event_volume
    )

@tool
def detect_coordinated_activity(
    addresses: list[str],
    time_window: timedelta = timedelta(hours=1),
    min_participants: int = 3
) -> list[CoordinatedActivityResult]:
    """Detect coordinated transactions across multiple addresses."""
    ...

@tool
def compute_anomaly_score(
    evidence: list[Evidence],
    weights: dict[str, float] | None = None
) -> float:
    """Compute composite anomaly score from multiple signals."""
    ...
```

---

### 1.6 Validator Agent

**Purpose:** Verify all citations and claims before final report generation.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | Before Report Agent generates output |
| **Input** | Draft analysis with citations |
| **Output** | Validation result (pass/fail with details) |
| **Tools** | `verify_transaction`, `verify_address`, `verify_block`, `cross_reference_check` |
| **State** | Validation cache (avoid redundant lookups) |

**Responsibilities:**
1. Extract all citations from LLM-generated content
2. Verify each citation exists in source databases
3. Confirm cited data matches claimed facts
4. Flag any hallucinated or mismatched citations

**Validation Pipeline:**
```
LLM Draft Output
       │
       ▼
┌─────────────────┐
│ Citation Parser │
│                 │
│ Extract:        │
│ - [TX:0x...]    │
│ - [ADDR:0x...]  │
│ - [BLOCK:...]   │
│ - [TS:...]      │
└─────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PARALLEL VERIFICATION                    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Verify TX   │  │ Verify ADDR │  │ Verify BLOCK│         │
│  │             │  │             │  │             │         │
│  │ - Exists?   │  │ - Exists?   │  │ - Exists?   │         │
│  │ - Data match│  │ - Data match│  │ - Data match│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────┐
│ Validation      │
│ Result          │
│                 │
│ - all_valid: T/F│
│ - failures: []  │
│ - warnings: []  │
└─────────────────┘
       │
       ├──── If valid ────▶ Proceed to Report Agent
       │
       └──── If invalid ──▶ Return to Retrieval with constraints
```

**Tool Definitions:**
```python
@tool
def verify_transaction(tx_hash: str, claimed_data: dict) -> VerificationResult:
    """Verify transaction exists and claimed data matches."""
    actual = clickhouse_client.query(
        "SELECT * FROM ethereum.transactions WHERE hash = %(hash)s",
        {"hash": tx_hash}
    ).first_row_or_none()
    
    if not actual:
        return VerificationResult(valid=False, error="Transaction not found")
    
    mismatches = []
    for key, claimed_value in claimed_data.items():
        if key in actual and actual[key] != claimed_value:
            mismatches.append(f"{key}: claimed={claimed_value}, actual={actual[key]}")
    
    return VerificationResult(
        valid=len(mismatches) == 0,
        mismatches=mismatches
    )

@tool
def verify_address_activity(
    address: str,
    claimed_activity: dict,
    tolerance: float = 0.01
) -> VerificationResult:
    """Verify address activity claims (volumes, counts, etc.)."""
    ...

@tool
def cross_reference_sources(
    claim: str,
    sources: list[Source]
) -> CrossReferenceResult:
    """Verify claim is supported by multiple independent sources."""
    ...
```

---

### 1.7 Report Agent

**Purpose:** Generate human-readable analysis reports with mandatory citations.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | After Validator Agent confirms all citations |
| **Input** | Validated evidence, analysis results, user query |
| **Output** | Structured report with citations and confidence scores |
| **Tools** | `format_report`, `generate_visualization_spec`, `export_report` |
| **State** | Report templates, formatting preferences |

**Responsibilities:**
1. Synthesize validated evidence into coherent narrative
2. Ensure every factual claim has inline citation
3. Generate confidence scores based on evidence strength
4. Format output according to specified template

**Report Structure:**
```json
{
  "report_id": "uuid",
  "generated_at": "2024-01-20T15:30:00Z",
  "query": "original user query",
  "executive_summary": "Brief summary with key findings",
  "confidence_score": 0.87,
  "findings": [
    {
      "finding_id": 1,
      "title": "Pre-listing Accumulation Detected",
      "description": "Detailed description with [TX:0x...] citations",
      "confidence": 0.92,
      "evidence": [
        {
          "type": "transaction",
          "hash": "0xabc...",
          "block": 18234567,
          "timestamp": "2024-01-10T08:15:00Z",
          "relevance": "Shows initial accumulation"
        }
      ],
      "addresses_involved": [
        {
          "address": "0xdef...",
          "label": "Suspected insider wallet",
          "cluster_id": 42,
          "activity_summary": {...}
        }
      ]
    }
  ],
  "methodology": "Description of analysis approach",
  "limitations": ["List of caveats and limitations"],
  "raw_evidence_refs": ["Links to full evidence payloads"]
}
```

**LLM Prompt Template:**
```python
REPORT_GENERATION_PROMPT = """
You are generating a blockchain forensics report. 

STRICT REQUIREMENTS:
1. Every factual statement MUST include a citation
2. Citation format: [TX:hash], [ADDR:address], [BLOCK:number], [TS:timestamp]
3. If evidence is insufficient, state "INSUFFICIENT EVIDENCE" explicitly
4. Never extrapolate beyond provided evidence
5. Express confidence as a percentage with justification

VALIDATED EVIDENCE:
{evidence_json}

USER QUERY:
{user_query}

ANALYSIS RESULTS:
{analysis_results}

Generate a report following the JSON schema:
{report_schema}
"""
```

---

### 1.8 Supervisor Agent

**Purpose:** Coordinate agent execution, handle errors, and manage workflow state.

| Attribute | Specification |
|-----------|---------------|
| **Trigger** | Every workflow invocation |
| **Input** | Initial request, workflow configuration |
| **Output** | Final result or error with diagnostics |
| **Tools** | `route_to_agent`, `handle_error`, `checkpoint_state`, `escalate_to_human` |
| **State** | Full workflow state, execution history |

**Responsibilities:**
1. Route requests to appropriate agents based on intent
2. Handle agent failures with retry logic
3. Maintain workflow checkpoints for resumption
4. Escalate to human when confidence is low or errors persist

---

## 2. LangGraph Workflow Implementation

### 2.1 State Schema

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages

class InvestigationState(TypedDict):
    # Input
    query: str
    user_id: str
    config: dict
    
    # Parsed intent
    intent: str  # "accumulation_detection", "flow_analysis", "cluster_investigation"
    entities: dict  # {"addresses": [...], "tokens": [...], "time_range": {...}}
    
    # Retrieved evidence
    semantic_results: list[dict]
    graph_results: list[dict]
    olap_results: list[dict]
    fused_evidence: list[dict]
    
    # Analysis results
    anomalies: list[dict]
    clusters: list[dict]
    flow_paths: list[dict]
    
    # Validation
    validation_result: dict
    validation_attempts: int
    
    # Output
    report: dict | None
    error: str | None
    
    # Workflow control
    messages: Annotated[list, add_messages]
    next_agent: str | None
```

### 2.2 Graph Definition

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def create_investigation_workflow():
    workflow = StateGraph(InvestigationState)
    
    # Add agent nodes
    workflow.add_node("query_parser", query_parser_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("graph_analysis", graph_analysis_node)
    workflow.add_node("anomaly_detection", anomaly_detection_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("report_generator", report_generator_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Entry point
    workflow.set_entry_point("query_parser")
    
    # Define edges
    workflow.add_edge("query_parser", "retrieval")
    
    # Conditional routing after retrieval
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "graph_needed": "graph_analysis",
            "proceed_to_anomaly": "anomaly_detection",
            "error": "error_handler"
        }
    )
    
    workflow.add_edge("graph_analysis", "anomaly_detection")
    workflow.add_edge("anomaly_detection", "validator")
    
    # Validation loop
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "valid": "report_generator",
            "retry": "retrieval",
            "max_retries_exceeded": "error_handler"
        }
    )
    
    workflow.add_edge("report_generator", END)
    workflow.add_edge("error_handler", END)
    
    # Compile with checkpointing
    memory = SqliteSaver.from_conn_string(":memory:")  # Use PostgreSQL in production
    return workflow.compile(checkpointer=memory)


def route_after_retrieval(state: InvestigationState) -> str:
    """Determine next step based on retrieval results."""
    if state.get("error"):
        return "error"
    
    intent = state.get("intent", "")
    if intent in ["flow_analysis", "cluster_investigation"]:
        return "graph_needed"
    
    if len(state.get("fused_evidence", [])) > 0:
        return "proceed_to_anomaly"
    
    return "error"


def route_after_validation(state: InvestigationState) -> str:
    """Determine next step based on validation results."""
    validation = state.get("validation_result", {})
    attempts = state.get("validation_attempts", 0)
    
    if validation.get("all_valid", False):
        return "valid"
    
    if attempts >= 3:
        return "max_retries_exceeded"
    
    return "retry"
```

### 2.3 Agent Node Implementations

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

async def query_parser_node(state: InvestigationState) -> dict:
    """Parse user query to extract intent and entities."""
    
    llm_with_tools = llm.bind_tools([
        extract_addresses,
        extract_tokens,
        extract_time_range,
        classify_intent
    ])
    
    messages = [
        SystemMessage(content=QUERY_PARSER_SYSTEM_PROMPT),
        HumanMessage(content=state["query"])
    ]
    
    response = await llm_with_tools.ainvoke(messages)
    
    # Extract tool calls
    intent = None
    entities = {"addresses": [], "tokens": [], "time_range": None}
    
    for tool_call in response.tool_calls:
        if tool_call["name"] == "classify_intent":
            intent = tool_call["args"]["intent"]
        elif tool_call["name"] == "extract_addresses":
            entities["addresses"] = tool_call["args"]["addresses"]
        elif tool_call["name"] == "extract_tokens":
            entities["tokens"] = tool_call["args"]["tokens"]
        elif tool_call["name"] == "extract_time_range":
            entities["time_range"] = tool_call["args"]
    
    return {
        "intent": intent,
        "entities": entities,
        "messages": [response]
    }


async def retrieval_node(state: InvestigationState) -> dict:
    """Execute hybrid retrieval across all databases."""
    
    entities = state["entities"]
    intent = state["intent"]
    
    # Parallel retrieval
    semantic_task = semantic_search(
        query=state["query"],
        filters=build_filters(entities)
    )
    
    olap_task = execute_olap_queries(
        intent=intent,
        entities=entities
    )
    
    semantic_results, olap_results = await asyncio.gather(
        semantic_task, olap_task
    )
    
    # Fuse results using RRF
    fused = reciprocal_rank_fusion([
        semantic_results,
        olap_results
    ])
    
    return {
        "semantic_results": semantic_results,
        "olap_results": olap_results,
        "fused_evidence": fused
    }


async def validator_node(state: InvestigationState) -> dict:
    """Validate all citations in the analysis."""
    
    anomalies = state.get("anomalies", [])
    
    validation_tasks = []
    for anomaly in anomalies:
        for evidence in anomaly.get("evidence", []):
            if evidence["type"] == "transaction":
                validation_tasks.append(
                    verify_transaction(evidence["hash"], evidence)
                )
            elif evidence["type"] == "address":
                validation_tasks.append(
                    verify_address_activity(evidence["address"], evidence)
                )
    
    results = await asyncio.gather(*validation_tasks, return_exceptions=True)
    
    failures = [r for r in results if isinstance(r, Exception) or not r.valid]
    
    return {
        "validation_result": {
            "all_valid": len(failures) == 0,
            "failures": failures,
            "total_checked": len(validation_tasks)
        },
        "validation_attempts": state.get("validation_attempts", 0) + 1
    }
```

---

## 3. Inter-Agent Communication

### 3.1 Message Protocol

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Any

class AgentMessage(BaseModel):
    """Standard message format between agents."""
    
    message_id: str
    timestamp: datetime
    source_agent: str
    target_agent: str
    message_type: Literal["request", "response", "error", "checkpoint"]
    payload: dict[str, Any]
    correlation_id: str  # Links related messages
    
    class Config:
        json_schema_extra = {
            "example": {
                "message_id": "msg_123",
                "timestamp": "2024-01-20T15:30:00Z",
                "source_agent": "retrieval",
                "target_agent": "graph_analysis",
                "message_type": "request",
                "payload": {
                    "addresses": ["0xabc...", "0xdef..."],
                    "analysis_type": "clustering"
                },
                "correlation_id": "investigation_456"
            }
        }
```

### 3.2 Error Handling Protocol

```python
class AgentError(BaseModel):
    """Standardized error format."""
    
    error_code: str
    error_message: str
    agent: str
    timestamp: datetime
    recoverable: bool
    retry_after_seconds: int | None
    context: dict[str, Any]

# Error codes
ERROR_CODES = {
    "E001": "Database connection failed",
    "E002": "Query timeout",
    "E003": "Invalid input format",
    "E004": "Validation failed",
    "E005": "LLM generation failed",
    "E006": "Rate limit exceeded",
    "E007": "Insufficient evidence",
    "E008": "Max retries exceeded"
}
```

---

## 4. Scaling Considerations

### 4.1 Agent Parallelization

```python
# Parallel execution within retrieval
async def parallel_retrieval(state: InvestigationState) -> dict:
    """Execute all retrieval operations in parallel."""
    
    async with asyncio.TaskGroup() as tg:
        semantic_task = tg.create_task(semantic_search(...))
        graph_task = tg.create_task(graph_query(...))
        olap_task = tg.create_task(olap_query(...))
    
    return {
        "semantic_results": semantic_task.result(),
        "graph_results": graph_task.result(),
        "olap_results": olap_task.result()
    }
```

### 4.2 Horizontal Scaling

- **Agent Workers**: Deploy multiple instances behind load balancer
- **Database Sharding**: ClickHouse shards by block range, Qdrant shards by collection
- **LLM Inference**: Multiple vLLM replicas with request routing

### 4.3 Caching Strategy

```python
# Redis caching for common patterns
CACHE_CONFIG = {
    "embedding_cache": {
        "ttl": 86400,  # 24 hours
        "key_pattern": "emb:{text_hash}"
    },
    "query_cache": {
        "ttl": 3600,  # 1 hour
        "key_pattern": "query:{query_hash}:{params_hash}"
    },
    "cluster_cache": {
        "ttl": 86400,  # 24 hours (clusters change slowly)
        "key_pattern": "cluster:{address}"
    }
}
```
