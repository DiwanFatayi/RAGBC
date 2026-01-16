"""Neo4j graph database client wrapper."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
import structlog

from config.settings import get_settings

logger = structlog.get_logger()


class Neo4jClient:
    """
    Neo4j client wrapper for wallet graph operations.
    
    Provides:
    - Async connection management
    - Graph query utilities
    - GDS algorithm wrappers
    - Batch operations
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        settings = get_settings()
        db_settings = settings.database

        self.uri = uri or db_settings.neo4j_uri
        self.user = user or db_settings.neo4j_user
        self.password = password or db_settings.neo4j_password.get_secret_value()
        self.database = database or db_settings.neo4j_database

        self._driver: AsyncDriver | None = None
        self._logger = logger.bind(component="neo4j")

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            # Verify connectivity
            await self._driver.verify_connectivity()
            self._logger.info("neo4j_connected", uri=self.uri)

    async def close(self) -> None:
        """Close the connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._logger.info("neo4j_disconnected")

    @property
    def driver(self) -> AsyncDriver:
        """Get the driver, ensuring it's connected."""
        if self._driver is None:
            raise RuntimeError("Neo4j client not connected. Call connect() first.")
        return self._driver

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get a session for executing queries."""
        async with self.driver.session(database=self.database) as session:
            yield session

    async def query(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as dictionaries."""
        self._logger.debug("executing_cypher", query=cypher[:200])
        async with self.session() as session:
            result = await session.run(cypher, parameters or {})
            records = [record.data() async for record in result]
            return records

    async def execute(
        self,
        cypher: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Execute a Cypher command without returning results."""
        async with self.session() as session:
            await session.run(cypher, parameters or {})

    async def create_wallet_node(self, address: str, **properties: Any) -> None:
        """Create or merge a wallet node."""
        cypher = """
        MERGE (w:Wallet {address: $address})
        SET w += $properties
        """
        await self.execute(cypher, {"address": address, "properties": properties})

    async def create_transfer_relationship(
        self,
        from_address: str,
        to_address: str,
        token_address: str,
        value: float,
        tx_hash: str,
        timestamp: int,
    ) -> None:
        """Create a transfer relationship between wallets."""
        cypher = """
        MATCH (from:Wallet {address: $from_address})
        MATCH (to:Wallet {address: $to_address})
        MERGE (from)-[r:TRANSFERRED {token: $token}]->(to)
        ON CREATE SET 
            r.total_value = $value,
            r.tx_count = 1,
            r.first_tx = $timestamp,
            r.last_tx = $timestamp,
            r.tx_hashes = [$tx_hash]
        ON MATCH SET
            r.total_value = r.total_value + $value,
            r.tx_count = r.tx_count + 1,
            r.last_tx = $timestamp,
            r.tx_hashes = r.tx_hashes + $tx_hash
        """
        await self.execute(
            cypher,
            {
                "from_address": from_address,
                "to_address": to_address,
                "token": token_address,
                "value": value,
                "tx_hash": tx_hash,
                "timestamp": timestamp,
            },
        )

    async def find_wallet_cluster(
        self,
        address: str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Find all wallets in the same cluster as the given address."""
        cypher = """
        MATCH (start:Wallet {address: $address})
        CALL gds.wcc.stream({
            nodeProjection: 'Wallet',
            relationshipProjection: 'TRANSFERRED'
        })
        YIELD nodeId, componentId
        WITH start, componentId as startComponent
        WHERE gds.util.asNode(nodeId) = start
        
        CALL gds.wcc.stream({
            nodeProjection: 'Wallet',
            relationshipProjection: 'TRANSFERRED'
        })
        YIELD nodeId, componentId
        WHERE componentId = startComponent
        RETURN gds.util.asNode(nodeId).address as address
        """
        return await self.query(cypher, {"address": address})

    async def trace_fund_flow(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
    ) -> list[dict[str, Any]]:
        """Find fund flow paths between two addresses."""
        cypher = """
        MATCH path = shortestPath(
            (source:Wallet {address: $source})-[:TRANSFERRED*1..$max_hops]->(target:Wallet {address: $target})
        )
        RETURN 
            [n IN nodes(path) | n.address] as addresses,
            [r IN relationships(path) | r.total_value] as values,
            length(path) as hops
        ORDER BY hops
        LIMIT 10
        """
        return await self.query(
            cypher,
            {"source": source, "target": target, "max_hops": max_hops},
        )

    async def run_louvain_clustering(self) -> list[dict[str, Any]]:
        """Run Louvain community detection algorithm."""
        # First project the graph
        await self.execute("""
            CALL gds.graph.project(
                'wallet-graph',
                'Wallet',
                'TRANSFERRED',
                {relationshipProperties: 'total_value'}
            )
        """)

        # Run Louvain
        results = await self.query("""
            CALL gds.louvain.stream('wallet-graph', {
                relationshipWeightProperty: 'total_value'
            })
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).address as address, communityId
        """)

        # Drop the projection
        await self.execute("CALL gds.graph.drop('wallet-graph')")

        return results

    async def get_pagerank(self, top_n: int = 100) -> list[dict[str, Any]]:
        """Get PageRank scores for wallets."""
        cypher = """
        CALL gds.pageRank.stream({
            nodeProjection: 'Wallet',
            relationshipProjection: 'TRANSFERRED'
        })
        YIELD nodeId, score
        RETURN gds.util.asNode(nodeId).address as address, score
        ORDER BY score DESC
        LIMIT $top_n
        """
        return await self.query(cypher, {"top_n": top_n})


# Singleton instance
_client: Neo4jClient | None = None


async def get_neo4j_client() -> Neo4jClient:
    """Get singleton Neo4j client instance."""
    global _client
    if _client is None:
        _client = Neo4jClient()
        await _client.connect()
    return _client
