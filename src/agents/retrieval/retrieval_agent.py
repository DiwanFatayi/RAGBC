"""
Retrieval Agent - Executes hybrid search across all databases.

Combines semantic search (Qdrant), graph queries (Neo4j), and
analytical queries (ClickHouse) with Reciprocal Rank Fusion.
"""

import asyncio
from typing import Any

import structlog
from langchain_core.tools import tool

from src.agents.base import BaseAgent, ToolExecutor
from src.agents.retrieval.fusion import reciprocal_rank_fusion
from src.agents.state import Citation, Evidence, InvestigationState

logger = structlog.get_logger()


class RetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant evidence from all databases.
    
    Implements hybrid search combining:
    - Semantic search via Qdrant embeddings
    - Graph traversal via Neo4j Cypher
    - Analytical queries via ClickHouse SQL
    """

    name = "retrieval_agent"
    description = "Retrieves evidence using hybrid search across vector, graph, and OLAP databases"

    def __init__(
        self,
        qdrant_client,
        neo4j_driver,
        clickhouse_client,
        embedding_model,
        llm=None,
        top_k: int = 100,
        rrf_k: int = 60,
    ):
        super().__init__(llm=llm)
        self.qdrant = qdrant_client
        self.neo4j = neo4j_driver
        self.clickhouse = clickhouse_client
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.tool_executor = ToolExecutor()

    async def process(self, state: InvestigationState) -> dict[str, Any]:
        """Execute hybrid retrieval based on parsed intent and entities."""
        query = state.query
        intent = state.intent
        entities = state.entities

        self._logger.info(
            "starting_retrieval",
            intent=intent,
            entities_count=len(entities),
        )

        # Build filters from entities
        filters = self._build_filters(entities)

        # Execute retrievals in parallel
        semantic_task = self._semantic_search(query, filters)
        olap_task = self._olap_search(intent, entities)

        # Graph search only if addresses provided
        if entities.get("addresses"):
            graph_task = self._graph_search(entities)
        else:
            graph_task = asyncio.coroutine(lambda: [])()

        results = await asyncio.gather(
            semantic_task,
            olap_task,
            graph_task,
            return_exceptions=True,
        )

        semantic_results = results[0] if not isinstance(results[0], Exception) else []
        olap_results = results[1] if not isinstance(results[1], Exception) else []
        graph_results = results[2] if not isinstance(results[2], Exception) else []

        # Log any errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self._logger.error(
                    "retrieval_error",
                    source=["semantic", "olap", "graph"][i],
                    error=str(result),
                )

        # Fuse results using RRF
        fused = reciprocal_rank_fusion(
            [semantic_results, olap_results, graph_results],
            k=self.rrf_k,
        )

        # Convert to Evidence objects
        evidence_list = [
            self._to_evidence(item)
            for item in fused[:self.top_k]
        ]

        self._logger.info(
            "retrieval_complete",
            semantic_count=len(semantic_results),
            olap_count=len(olap_results),
            graph_count=len(graph_results),
            fused_count=len(evidence_list),
        )

        return {
            "semantic_results": semantic_results,
            "olap_results": olap_results,
            "graph_results": graph_results,
            "fused_evidence": evidence_list,
        }

    async def _semantic_search(
        self,
        query: str,
        filters: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Execute semantic search on Qdrant."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )

        # Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(filters) if filters else None

        # Execute search
        results = self.qdrant.search(
            collection_name="transaction_patterns",
            query_vector=query_embedding.tolist(),
            query_filter=qdrant_filter,
            limit=self.top_k,
            with_payload=True,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "source": "qdrant",
                **hit.payload,
            }
            for hit in results
        ]

    async def _olap_search(
        self,
        intent: str | None,
        entities: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute analytical queries on ClickHouse."""
        queries = self._generate_olap_queries(intent, entities)
        results = []

        for query_name, (sql, params) in queries.items():
            try:
                result = self.clickhouse.query(sql, parameters=params)
                for row in result.result_rows:
                    results.append({
                        "source": "clickhouse",
                        "query_type": query_name,
                        **dict(zip(result.column_names, row)),
                    })
            except Exception as e:
                self._logger.warning(
                    "olap_query_failed",
                    query=query_name,
                    error=str(e),
                )

        return results

    async def _graph_search(self, entities: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute graph queries on Neo4j."""
        addresses = entities.get("addresses", [])
        results = []

        async with self.neo4j.session() as session:
            # Find connected wallets
            if addresses:
                cypher = """
                MATCH (w:Wallet)-[r:TRANSFERRED]->(target:Wallet)
                WHERE w.address IN $addresses
                RETURN w.address as source, target.address as target,
                       r.total_value as value, r.tx_count as tx_count
                ORDER BY r.total_value DESC
                LIMIT $limit
                """
                result = await session.run(
                    cypher,
                    {"addresses": addresses, "limit": self.top_k},
                )
                records = [record.data() async for record in result]
                for record in records:
                    results.append({
                        "source": "neo4j",
                        "query_type": "outgoing_transfers",
                        **record,
                    })

            # Find incoming transfers
            if addresses:
                cypher = """
                MATCH (source:Wallet)-[r:TRANSFERRED]->(w:Wallet)
                WHERE w.address IN $addresses
                RETURN source.address as source, w.address as target,
                       r.total_value as value, r.tx_count as tx_count
                ORDER BY r.total_value DESC
                LIMIT $limit
                """
                result = await session.run(
                    cypher,
                    {"addresses": addresses, "limit": self.top_k},
                )
                records = [record.data() async for record in result]
                for record in records:
                    results.append({
                        "source": "neo4j",
                        "query_type": "incoming_transfers",
                        **record,
                    })

        return results

    def _build_filters(self, entities: dict[str, Any]) -> dict[str, Any]:
        """Build filter dictionary from extracted entities."""
        filters = {}

        if entities.get("addresses"):
            filters["addresses"] = entities["addresses"]

        if entities.get("tokens"):
            filters["token_addresses"] = entities["tokens"]

        if entities.get("time_range"):
            time_range = entities["time_range"]
            if time_range.get("start"):
                filters["start_timestamp"] = time_range["start"]
            if time_range.get("end"):
                filters["end_timestamp"] = time_range["end"]

        return filters

    def _build_qdrant_filter(self, filters: dict[str, Any]) -> dict:
        """Convert filters to Qdrant filter format."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, Range

        conditions = []

        if filters.get("addresses"):
            conditions.append(
                FieldCondition(
                    key="address",
                    match=MatchAny(any=filters["addresses"]),
                )
            )

        if filters.get("token_addresses"):
            conditions.append(
                FieldCondition(
                    key="token_address",
                    match=MatchAny(any=filters["token_addresses"]),
                )
            )

        if filters.get("start_timestamp"):
            conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(gte=filters["start_timestamp"].timestamp()),
                )
            )

        if filters.get("end_timestamp"):
            conditions.append(
                FieldCondition(
                    key="timestamp",
                    range=Range(lte=filters["end_timestamp"].timestamp()),
                )
            )

        return Filter(must=conditions) if conditions else None

    def _generate_olap_queries(
        self,
        intent: str | None,
        entities: dict[str, Any],
    ) -> dict[str, tuple[str, dict]]:
        """Generate OLAP queries based on intent and entities."""
        queries = {}

        addresses = entities.get("addresses", [])
        tokens = entities.get("tokens", [])
        time_range = entities.get("time_range", {})

        # Address activity summary
        if addresses:
            queries["address_activity"] = (
                """
                SELECT
                    from_address as address,
                    count() as tx_count,
                    sum(value) as total_value,
                    min(block_timestamp) as first_tx,
                    max(block_timestamp) as last_tx,
                    uniqExact(to_address) as unique_recipients
                FROM ethereum.transactions
                WHERE from_address IN %(addresses)s
                GROUP BY from_address
                """,
                {"addresses": addresses},
            )

        # Token transfer analysis
        if tokens:
            queries["token_transfers"] = (
                """
                SELECT
                    to_address,
                    sum(value) as total_received,
                    count() as transfer_count,
                    min(block_timestamp) as first_transfer,
                    max(block_timestamp) as last_transfer
                FROM ethereum.token_transfers
                WHERE token_address IN %(tokens)s
                GROUP BY to_address
                ORDER BY total_received DESC
                LIMIT 100
                """,
                {"tokens": tokens},
            )

        # Pre-event accumulation (if time range specified)
        if time_range.get("end") and tokens:
            queries["pre_event_accumulation"] = (
                """
                SELECT
                    to_address,
                    sum(value) as accumulated,
                    count() as tx_count,
                    min(block_timestamp) as start_accumulation
                FROM ethereum.token_transfers
                WHERE token_address IN %(tokens)s
                  AND block_timestamp < %(end_time)s
                  AND block_timestamp > %(end_time)s - INTERVAL 30 DAY
                GROUP BY to_address
                HAVING accumulated > 0
                ORDER BY accumulated DESC
                LIMIT 100
                """,
                {"tokens": tokens, "end_time": time_range["end"]},
            )

        return queries

    def _to_evidence(self, item: dict[str, Any]) -> Evidence:
        """Convert a result item to Evidence object."""
        source_map = {
            "qdrant": "qdrant",
            "clickhouse": "clickhouse",
            "neo4j": "neo4j",
        }
        source = source_map.get(item.get("source", ""), "clickhouse")

        # Extract citations from the item
        citations = []
        if item.get("tx_hash"):
            citations.append(Citation(
                type="transaction",
                value=item["tx_hash"],
                context="Retrieved transaction",
            ))
        if item.get("address") or item.get("from_address"):
            addr = item.get("address") or item.get("from_address")
            citations.append(Citation(
                type="address",
                value=addr,
                context="Retrieved address",
            ))
        if item.get("block_number"):
            citations.append(Citation(
                type="block",
                value=str(item["block_number"]),
                context="Block reference",
            ))

        return Evidence(
            source=source,
            data=item,
            relevance_score=item.get("score", item.get("rrf_score", 0.5)),
            citations=citations,
        )
