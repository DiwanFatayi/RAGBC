"""ClickHouse database client."""

from src.db.clickhouse.client import ClickHouseClient, get_clickhouse_client

__all__ = ["ClickHouseClient", "get_clickhouse_client"]
