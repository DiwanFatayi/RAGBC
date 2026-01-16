"""Qdrant vector database client wrapper."""

from typing import Any

from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    VectorParams,
)
import structlog

from config.settings import get_settings

logger = structlog.get_logger()


class QdrantClient:
    """
    Qdrant client wrapper with utilities for blockchain pattern storage.
    
    Provides:
    - Collection management
    - Hybrid search (dense + sparse)
    - Metadata filtering
    - Batch upsert
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        db_settings = settings.database

        self.host = host or db_settings.qdrant_host
        self.port = port or db_settings.qdrant_port
        api_key = api_key or (
            db_settings.qdrant_api_key.get_secret_value()
            if db_settings.qdrant_api_key
            else None
        )

        self._client = BaseQdrantClient(
            host=self.host,
            port=self.port,
            api_key=api_key,
        )
        self._logger = logger.bind(component="qdrant")
        self._logger.info("qdrant_connected", host=self.host, port=self.port)

    def create_collection(
        self,
        name: str,
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
        quantization: bool = True,
    ) -> None:
        """Create a new collection with optimized settings."""
        quantization_config = None
        if quantization:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance,
            ),
            quantization_config=quantization_config,
        )
        self._logger.info("collection_created", name=name, vector_size=vector_size)

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        collections = self._client.get_collections().collections
        return any(c.name == name for c in collections)

    def upsert(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
    ) -> None:
        """
        Upsert points into a collection.
        
        Each point should have:
        - id: unique identifier
        - vector: embedding vector
        - payload: metadata dictionary
        """
        point_structs = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]

        self._client.upsert(
            collection_name=collection_name,
            points=point_structs,
        )
        self._logger.debug(
            "points_upserted",
            collection=collection_name,
            count=len(points),
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 100,
        query_filter: Filter | None = None,
        with_payload: bool = True,
        score_threshold: float | None = None,
    ) -> list[Any]:
        """Execute semantic search with optional filtering."""
        return self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
            score_threshold=score_threshold,
        )

    def scroll(
        self,
        collection_name: str,
        scroll_filter: dict | None = None,
        limit: int = 100,
        offset: str | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[Any], str | None]:
        """Scroll through collection with filtering."""
        qdrant_filter = None
        if scroll_filter:
            qdrant_filter = self._build_filter(scroll_filter)

        return self._client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )

    def delete(
        self,
        collection_name: str,
        points_selector: list[str | int] | Filter,
    ) -> None:
        """Delete points by IDs or filter."""
        self._client.delete(
            collection_name=collection_name,
            points_selector=points_selector,
        )

    def get_collection_info(self, name: str) -> dict[str, Any]:
        """Get collection statistics."""
        info = self._client.get_collection(name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }

    def _build_filter(self, filter_dict: dict) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []

        for key, value in filter_dict.get("must", []):
            if isinstance(value, dict):
                if "match" in value:
                    if "any" in value["match"]:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchAny(any=value["match"]["any"]),
                            )
                        )
                    else:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value["match"]["value"]),
                            )
                        )
                elif "range" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(**value["range"]),
                        )
                    )

        return Filter(must=conditions) if conditions else None

    def close(self) -> None:
        """Close the connection."""
        self._client.close()
        self._logger.info("qdrant_disconnected")


# Singleton instance
_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    """Get singleton Qdrant client instance."""
    global _client
    if _client is None:
        _client = QdrantClient()
    return _client
