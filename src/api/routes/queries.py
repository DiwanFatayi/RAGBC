"""Ad-hoc query endpoints for direct database access."""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter()


class AddressActivityResponse(BaseModel):
    """Response for address activity query."""

    address: str
    transaction_count: int
    total_value_sent: float
    total_value_received: float
    unique_counterparties: int
    first_seen: datetime | None
    last_seen: datetime | None
    tokens_interacted: list[str]


class TokenHoldersResponse(BaseModel):
    """Response for token holders query."""

    token_address: str
    holders: list[dict[str, Any]]
    total_holders: int
    query_time_ms: int


class WalletClusterResponse(BaseModel):
    """Response for wallet cluster query."""

    seed_address: str
    cluster_id: int | None
    cluster_size: int
    addresses: list[str]
    central_addresses: list[str]


class HybridSearchRequest(BaseModel):
    """Request for hybrid search."""

    query: str = Field(..., description="Natural language search query")
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = Field(default=50, ge=1, le=500)


class HybridSearchResponse(BaseModel):
    """Response for hybrid search."""

    query: str
    results: list[dict[str, Any]]
    total_results: int
    sources: dict[str, int]


@router.get("/address/{address}/activity", response_model=AddressActivityResponse)
async def get_address_activity(
    request: Request,
    address: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> AddressActivityResponse:
    """
    Get activity summary for a specific address.
    
    Returns transaction counts, volumes, and counterparty information.
    """
    clickhouse = request.app.state.clickhouse

    # Build date filters
    date_filter = ""
    params: dict[str, Any] = {"address": address.lower()}

    if start_date:
        date_filter += " AND block_timestamp >= %(start_date)s"
        params["start_date"] = start_date
    if end_date:
        date_filter += " AND block_timestamp <= %(end_date)s"
        params["end_date"] = end_date

    # Query sent transactions
    sent_query = f"""
    SELECT 
        count() as tx_count,
        sum(value) as total_value,
        uniqExact(to_address) as unique_recipients,
        min(block_timestamp) as first_tx,
        max(block_timestamp) as last_tx
    FROM ethereum.transactions
    WHERE from_address = %(address)s {date_filter}
    """

    sent_result = clickhouse.query(sent_query, parameters=params)
    sent_data = dict(zip(sent_result.column_names, sent_result.result_rows[0]))

    # Query received transactions
    received_query = f"""
    SELECT 
        count() as tx_count,
        sum(value) as total_value,
        uniqExact(from_address) as unique_senders
    FROM ethereum.transactions
    WHERE to_address = %(address)s {date_filter}
    """

    received_result = clickhouse.query(received_query, parameters=params)
    received_data = dict(zip(received_result.column_names, received_result.result_rows[0]))

    # Query token interactions
    tokens_query = f"""
    SELECT DISTINCT token_address
    FROM ethereum.token_transfers
    WHERE from_address = %(address)s OR to_address = %(address)s
    LIMIT 100
    """
    tokens_result = clickhouse.query(tokens_query, parameters=params)
    tokens = [row[0] for row in tokens_result.result_rows]

    return AddressActivityResponse(
        address=address,
        transaction_count=sent_data["tx_count"] + received_data["tx_count"],
        total_value_sent=float(sent_data["total_value"] or 0),
        total_value_received=float(received_data["total_value"] or 0),
        unique_counterparties=sent_data["unique_recipients"] + received_data["unique_senders"],
        first_seen=sent_data["first_tx"],
        last_seen=sent_data["last_tx"],
        tokens_interacted=tokens,
    )


@router.get("/token/{token_address}/holders", response_model=TokenHoldersResponse)
async def get_token_holders(
    request: Request,
    token_address: str,
    limit: int = Query(default=100, ge=1, le=1000),
    min_balance: float = Query(default=0, ge=0),
) -> TokenHoldersResponse:
    """
    Get top holders for a specific token.
    
    Returns addresses with their balances sorted by holdings.
    """
    import time

    start_time = time.time()
    clickhouse = request.app.state.clickhouse

    # Calculate net balances from transfers
    query = """
    SELECT 
        address,
        sum(net_value) as balance
    FROM (
        SELECT to_address as address, sum(value) as net_value
        FROM ethereum.token_transfers
        WHERE token_address = %(token)s
        GROUP BY to_address
        
        UNION ALL
        
        SELECT from_address as address, -sum(value) as net_value
        FROM ethereum.token_transfers
        WHERE token_address = %(token)s
        GROUP BY from_address
    )
    GROUP BY address
    HAVING balance > %(min_balance)s
    ORDER BY balance DESC
    LIMIT %(limit)s
    """

    result = clickhouse.query(
        query,
        parameters={
            "token": token_address.lower(),
            "min_balance": min_balance,
            "limit": limit,
        },
    )

    holders = [
        {"address": row[0], "balance": float(row[1])}
        for row in result.result_rows
    ]

    query_time = int((time.time() - start_time) * 1000)

    return TokenHoldersResponse(
        token_address=token_address,
        holders=holders,
        total_holders=len(holders),
        query_time_ms=query_time,
    )


@router.get("/wallet/{address}/cluster", response_model=WalletClusterResponse)
async def get_wallet_cluster(
    request: Request,
    address: str,
    max_depth: int = Query(default=2, ge=1, le=5),
) -> WalletClusterResponse:
    """
    Find the wallet cluster containing the given address.
    
    Uses graph analysis to identify related wallets.
    """
    neo4j = request.app.state.neo4j

    # Find cluster using connected components
    try:
        cluster_result = await neo4j.find_wallet_cluster(address.lower(), max_depth)
        addresses = [r["address"] for r in cluster_result]

        if not addresses:
            return WalletClusterResponse(
                seed_address=address,
                cluster_id=None,
                cluster_size=1,
                addresses=[address],
                central_addresses=[address],
            )

        # Get central addresses (highest degree)
        central_query = """
        MATCH (w:Wallet)
        WHERE w.address IN $addresses
        WITH w, size((w)-[:TRANSFERRED]-()) as degree
        ORDER BY degree DESC
        LIMIT 5
        RETURN w.address as address
        """
        central_result = await neo4j.query(central_query, {"addresses": addresses})
        central_addresses = [r["address"] for r in central_result]

        return WalletClusterResponse(
            seed_address=address,
            cluster_id=hash(tuple(sorted(addresses))) % 1000000,
            cluster_size=len(addresses),
            addresses=addresses[:100],  # Limit response size
            central_addresses=central_addresses,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")


@router.post("/search", response_model=HybridSearchResponse)
async def hybrid_search(
    request: Request,
    body: HybridSearchRequest,
) -> HybridSearchResponse:
    """
    Execute hybrid search across all databases.
    
    Combines semantic search (Qdrant), graph queries (Neo4j),
    and analytical queries (ClickHouse) with rank fusion.
    """
    from src.agents.retrieval.fusion import reciprocal_rank_fusion

    qdrant = request.app.state.qdrant
    clickhouse = request.app.state.clickhouse

    results_by_source: dict[str, list] = {
        "qdrant": [],
        "clickhouse": [],
    }

    # Semantic search
    try:
        from sentence_transformers import SentenceTransformer

        # Load embedding model (cached)
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        query_vector = model.encode(f"query: {body.query}", normalize_embeddings=True)

        semantic_results = qdrant.search(
            collection_name="transaction_patterns",
            query_vector=query_vector.tolist(),
            limit=body.limit,
        )

        results_by_source["qdrant"] = [
            {"source": "qdrant", "score": hit.score, **hit.payload}
            for hit in semantic_results
        ]
    except Exception:
        pass  # Qdrant search failed, continue with other sources

    # Keyword search in ClickHouse
    try:
        # Simple keyword matching for addresses and hashes
        keyword_query = """
        SELECT 
            hash as tx_hash,
            from_address,
            to_address,
            value,
            block_timestamp
        FROM ethereum.transactions
        WHERE 
            hash LIKE %(pattern)s
            OR from_address LIKE %(pattern)s
            OR to_address LIKE %(pattern)s
        LIMIT %(limit)s
        """
        pattern = f"%{body.query}%"
        ch_result = clickhouse.query(
            keyword_query,
            parameters={"pattern": pattern, "limit": body.limit},
        )

        results_by_source["clickhouse"] = [
            {"source": "clickhouse", **dict(zip(ch_result.column_names, row))}
            for row in ch_result.result_rows
        ]
    except Exception:
        pass

    # Fuse results
    all_results = list(results_by_source.values())
    fused = reciprocal_rank_fusion(all_results, k=60)

    source_counts = {
        source: len(results)
        for source, results in results_by_source.items()
    }

    return HybridSearchResponse(
        query=body.query,
        results=fused[:body.limit],
        total_results=len(fused),
        sources=source_counts,
    )
