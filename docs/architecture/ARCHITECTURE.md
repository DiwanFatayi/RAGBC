# Blockchain Insider Detection System - Technical Architecture

## Executive Summary

This document describes the architecture for a production-ready system that detects insider activity on the Ethereum blockchain. The system identifies anomalous transactional patterns indicating actions by project teams, venture funds, or major consultants before public announcements (token listings, partnerships, significant chart movements).

**Design Principles:**
- **Accuracy over Speed**: All conclusions must be grounded in verifiable on-chain data
- **Zero Hallucination Tolerance**: Every LLM output must cite transaction hashes, addresses, and block numbers
- **Hybrid Intelligence**: Combines semantic understanding with precise graph and time-series analysis
- **Modular Scalability**: Multi-agent architecture enables independent scaling and upgrades

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATION LAYER                                     │
│                         (Apache Airflow + LangGraph)                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │   Ingestion  │  │     ETL      │  │  Retrieval   │  │    Report    │            │
│  │    Agent     │──▶│    Agent     │──▶│    Agent     │──▶│    Agent     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘            │
│         │                 │                 │                 │                     │
│         │                 │                 │                 │                     │
│         ▼                 ▼                 ▼                 ▼                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │    Graph     │  │   Anomaly    │  │  Validator   │  │  Supervisor  │            │
│  │    Agent     │  │    Agent     │  │    Agent     │  │    Agent     │            │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         │                              │                              │
         ▼                              ▼                              ▼
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│                 │          │                 │          │                 │
│    Qdrant       │          │     Neo4j       │          │   ClickHouse    │
│  (Vector DB)    │          │   (Graph DB)    │          │    (OLAP)       │
│                 │          │                 │          │                 │
│  - Embeddings   │          │  - Wallet       │          │  - Transactions │
│  - Semantic     │          │    Clusters     │          │  - Blocks       │
│    Search       │          │  - Address      │          │  - Token        │
│  - Pattern      │          │    Relations    │          │    Transfers    │
│    Matching     │          │  - Flow         │          │  - Time-series  │
│                 │          │    Analysis     │          │    Aggregates   │
└─────────────────┘          └─────────────────┘          └─────────────────┘
         │                              │                              │
         └──────────────────────────────┼──────────────────────────────┘
                                        │
                              ┌─────────┴─────────┐
                              │                   │
                              ▼                   ▼
                    ┌─────────────────┐  ┌─────────────────┐
                    │  Ethereum RPC   │  │   CryptoHouse   │
                    │  (Alchemy/      │  │   (ClickHouse   │
                    │   Infura/       │  │    Ethereum     │
                    │   QuickNode)    │  │    Dataset)     │
                    └─────────────────┘  └─────────────────┘
```

---

## 2. Data Flow Architecture

### 2.1 Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

  External Sources                    Processing                    Storage
  ────────────────                    ──────────                    ───────

  ┌───────────────┐                                           ┌───────────────┐
  │ Ethereum RPC  │───┐                                       │               │
  │ (Real-time)   │   │     ┌───────────────────────┐        │  ClickHouse   │
  └───────────────┘   │     │                       │        │  (Raw Data)   │
                      ├────▶│   Ingestion Agent     │───────▶│               │
  ┌───────────────┐   │     │                       │        └───────────────┘
  │ CryptoHouse   │───┤     │  - Schema Validation  │                │
  │ (Historical)  │   │     │  - Deduplication      │                │
  └───────────────┘   │     │  - Normalization      │                ▼
                      │     │                       │        ┌───────────────┐
  ┌───────────────┐   │     └───────────────────────┘        │               │
  │ Event Streams │───┘                                      │   ETL Agent   │
  │ (Listings,    │                                          │               │
  │  Announcements)                                          └───────────────┘
  └───────────────┘                                                  │
                                                          ┌──────────┼──────────┐
                                                          │          │          │
                                                          ▼          ▼          ▼
                                                    ┌─────────┐ ┌─────────┐ ┌─────────┐
                                                    │ Qdrant  │ │  Neo4j  │ │ClickHse │
                                                    │(Vectors)│ │ (Graph) │ │(Derived)│
                                                    └─────────┘ └─────────┘ └─────────┘
```

### 2.2 Query/Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS QUERY FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

  User Query                    Multi-Stage Retrieval              Output
  ──────────                    ─────────────────────              ──────

  ┌───────────────┐
  │  "Find wallets│
  │   accumulating│
  │   $TOKEN before│      ┌─────────────────────────────────────────────┐
  │   listing"    │──────▶│           RETRIEVAL AGENT                   │
  └───────────────┘       │                                             │
                          │  1. Parse intent + extract entities         │
                          │  2. Generate hybrid query plan              │
                          └─────────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
           ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
           │ Semantic Search│         │ Graph Query   │         │ OLAP Query    │
           │ (Qdrant)       │         │ (Neo4j)       │         │ (ClickHouse)  │
           │               │         │               │         │               │
           │ "Similar to   │         │ MATCH         │         │ SELECT        │
           │  accumulation │         │ (w:Wallet)-   │         │   address,    │
           │  pattern"     │         │ [:TRANSFERRED]│         │   SUM(value)  │
           └───────────────┘         │ ->(t:Token)   │         │ WHERE ts <    │
                    │                └───────────────┘         │   listing_ts  │
                    │                         │                └───────────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                          ┌─────────────────────────────────────────────┐
                          │           FUSION & RANKING                  │
                          │                                             │
                          │  - Reciprocal Rank Fusion (RRF)             │
                          │  - Evidence aggregation                     │
                          │  - Confidence scoring                       │
                          └─────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌─────────────────────────────────────────────┐
                          │           VALIDATOR AGENT                   │
                          │                                             │
                          │  - Cross-reference all cited data           │
                          │  - Verify tx hashes exist on-chain          │
                          │  - Confirm timestamp consistency            │
                          └─────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌─────────────────────────────────────────────┐
                          │           REPORT AGENT (LLM)                │
                          │                                             │
                          │  - Generate human-readable analysis         │
                          │  - Mandatory source citations               │
                          │  - Confidence intervals                     │
                          └─────────────────────────────────────────────┘
                                              │
                                              ▼
                          ┌─────────────────────────────────────────────┐
                          │           FINAL OUTPUT                      │
                          │                                             │
                          │  {                                          │
                          │    "finding": "...",                        │
                          │    "confidence": 0.87,                      │
                          │    "evidence": [                            │
                          │      {"tx": "0xabc...", "block": 18234567}, │
                          │      {"address": "0xdef...", "label": "..."}│
                          │    ]                                        │
                          │  }                                          │
                          └─────────────────────────────────────────────┘
```

---

## 3. Component Interactions

### 3.1 Database Synchronization

```
                    ┌─────────────────────────────────────────┐
                    │         SYNC COORDINATOR                │
                    │         (Airflow DAG)                   │
                    └─────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
    ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
    │  ClickHouse   │         │    Qdrant     │         │     Neo4j     │
    │  (Source of   │────────▶│  (Derived     │         │   (Derived    │
    │   Truth)      │         │   Embeddings) │         │    Graph)     │
    └───────────────┘         └───────────────┘         └───────────────┘
            │                         │                         │
            │    Checksum: SHA256     │   Checksum: SHA256      │
            │    of block range       │   of embedding batch    │
            │                         │                         │
            └─────────────────────────┴─────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │         CONSISTENCY VERIFIER            │
                    │                                         │
                    │  - Compare record counts                │
                    │  - Validate referential integrity       │
                    │  - Detect drift between DBs             │
                    └─────────────────────────────────────────┘
```

### 3.2 LLM Integration Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM INTERACTION PATTERN                              │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐
  │  Retrieved      │
  │  Evidence       │
  │  (Structured)   │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        PROMPT CONSTRUCTION                              │
  │                                                                         │
  │  SYSTEM: You are a blockchain forensics analyst. You MUST:              │
  │  - Only state facts present in the provided evidence                    │
  │  - Cite every claim with [TX:hash] or [ADDR:address] or [BLOCK:number]  │
  │  - Express uncertainty when evidence is insufficient                    │
  │  - Never infer information not explicitly in the data                   │
  │                                                                         │
  │  EVIDENCE:                                                              │
  │  {structured_json_evidence}                                             │
  │                                                                         │
  │  QUERY: {user_query}                                                    │
  │                                                                         │
  │  Respond with analysis. Every factual claim requires a citation.        │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
  │   Qwen3-30B     │─────▶│  Response with  │─────▶│   Citation      │
  │   (via vLLM/    │      │  Citations      │      │   Extractor     │
  │   Ollama)       │      │                 │      │                 │
  └─────────────────┘      └─────────────────┘      └─────────────────┘
                                                             │
                                                             ▼
                                                   ┌─────────────────┐
                                                   │   Validator     │
                                                   │   Agent         │
                                                   │                 │
                                                   │  For each cite: │
                                                   │  - Query source │
                                                   │  - Verify match │
                                                   └─────────────────┘
                                                             │
                                              ┌──────────────┴──────────────┐
                                              │                             │
                                              ▼                             ▼
                                    ┌─────────────────┐           ┌─────────────────┐
                                    │  All Valid      │           │  Invalid Found  │
                                    │  ───────────    │           │  ─────────────  │
                                    │  Return report  │           │  Flag + retry   │
                                    │  with verified  │           │  with stricter  │
                                    │  citations      │           │  constraints    │
                                    └─────────────────┘           └─────────────────┘
```

---

## 4. Security & Access Control

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SECURITY ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   API Gateway   │
                              │   (FastAPI +    │
                              │   OAuth2/JWT)   │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │  Read-Only    │  │  Analyst      │  │  Admin        │
           │  Access       │  │  Access       │  │  Access       │
           │               │  │               │  │               │
           │  - Query      │  │  - Query      │  │  - Full CRUD  │
           │  - View       │  │  - View       │  │  - Config     │
           │    reports    │  │  - Create     │  │  - User mgmt  │
           │               │  │    alerts     │  │               │
           └───────────────┘  └───────────────┘  └───────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         SECRETS MANAGEMENT                              │
  │                                                                         │
  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
  │  │  HashiCorp  │    │   AWS       │    │   .env      │                 │
  │  │  Vault      │ OR │   Secrets   │ OR │   (dev)     │                 │
  │  │  (prod)     │    │   Manager   │    │             │                 │
  │  └─────────────┘    └─────────────┘    └─────────────┘                 │
  │                                                                         │
  │  Stored Secrets:                                                        │
  │  - RPC API keys (Alchemy, Infura)                                       │
  │  - Database credentials                                                 │
  │  - LLM API keys (if cloud-hosted)                                       │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Monitoring & Observability

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBSERVABILITY STACK                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │   Application   │     │   Prometheus    │     │    Grafana      │
  │   Metrics       │────▶│   (Scraping)    │────▶│   (Dashboards)  │
  │                 │     │                 │     │                 │
  │  - Query latency│     │  - Time-series  │     │  - Real-time    │
  │  - Agent exec   │     │    storage      │     │    monitoring   │
  │  - DB response  │     │                 │     │                 │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │   Structured    │     │    Loki /       │     │    Grafana      │
  │   Logs (JSON)   │────▶│    ELK Stack    │────▶│   (Log Search)  │
  │                 │     │                 │     │                 │
  │  - Agent traces │     │  - Log indexing │     │  - Correlation  │
  │  - LLM I/O      │     │  - Retention    │     │  - Alerting     │
  │  - Errors       │     │                 │     │                 │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  ┌─────────────────┐     ┌─────────────────┐
  │   OpenTelemetry │     │     Jaeger      │
  │   (Tracing)     │────▶│   (Distributed  │
  │                 │     │    Traces)      │
  │  - Request flow │     │                 │
  │  - Agent calls  │     │  - Latency      │
  │  - DB queries   │     │    analysis     │
  └─────────────────┘     └─────────────────┘
```

---

## 6. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES DEPLOYMENT                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         KUBERNETES CLUSTER                              │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  NAMESPACE: insider-detection                                   │   │
  │  │                                                                 │   │
  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │   │
  │  │  │  API Service  │  │  Agent Worker │  │  Airflow      │       │   │
  │  │  │  (Deployment) │  │  (Deployment) │  │  (StatefulSet)│       │   │
  │  │  │               │  │               │  │               │       │   │
  │  │  │  Replicas: 3  │  │  Replicas: 5  │  │  Scheduler +  │       │   │
  │  │  │  HPA: 3-10    │  │  HPA: 5-20    │  │  Workers      │       │   │
  │  │  └───────────────┘  └───────────────┘  └───────────────┘       │   │
  │  │                                                                 │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  NAMESPACE: databases                                           │   │
  │  │                                                                 │   │
  │  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │   │
  │  │  │  ClickHouse   │  │  Qdrant       │  │  Neo4j        │       │   │
  │  │  │  (StatefulSet)│  │  (StatefulSet)│  │  (StatefulSet)│       │   │
  │  │  │               │  │               │  │               │       │   │
  │  │  │  3 Shards     │  │  3 Replicas   │  │  Cluster mode │       │   │
  │  │  │  + Replicas   │  │  + Shards     │  │               │       │   │
  │  │  └───────────────┘  └───────────────┘  └───────────────┘       │   │
  │  │                                                                 │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │  NAMESPACE: llm-inference                                       │   │
  │  │                                                                 │   │
  │  │  ┌───────────────────────────────────────────────────────────┐ │   │
  │  │  │  vLLM / Ollama Deployment (GPU Nodes)                     │ │   │
  │  │  │                                                           │ │   │
  │  │  │  Model: Qwen3-30B-A3B-Thinking-2507                       │ │   │
  │  │  │  Resources: 2x A100 80GB or 4x A10G                       │ │   │
  │  │  │  Replicas: 2 (for HA)                                     │ │   │
  │  │  └───────────────────────────────────────────────────────────┘ │   │
  │  │                                                                 │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘
```
