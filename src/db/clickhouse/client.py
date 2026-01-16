"""ClickHouse client wrapper with connection pooling and utilities."""

from contextlib import contextmanager
from typing import Any, Iterator

import clickhouse_connect
from clickhouse_connect.driver import Client
import structlog

from config.settings import get_settings

logger = structlog.get_logger()


class ClickHouseClient:
    """
    ClickHouse client wrapper with connection management.
    
    Provides:
    - Connection pooling
    - Query execution with logging
    - Batch insert utilities
    - Schema management
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        settings = get_settings()
        db_settings = settings.database

        self.host = host or db_settings.clickhouse_host
        self.port = port or db_settings.clickhouse_port
        self.user = user or db_settings.clickhouse_user
        self.password = password or db_settings.clickhouse_password.get_secret_value()
        self.database = database or db_settings.clickhouse_database

        self._client: Client | None = None
        self._logger = logger.bind(component="clickhouse")

    @property
    def client(self) -> Client:
        """Get or create ClickHouse client."""
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                database=self.database,
            )
            self._logger.info(
                "clickhouse_connected",
                host=self.host,
                database=self.database,
            )
        return self._client

    def query(
        self,
        sql: str,
        parameters: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a query and return results."""
        self._logger.debug("executing_query", sql=sql[:200])
        return self.client.query(sql, parameters=parameters)

    def command(self, sql: str) -> None:
        """Execute a command (DDL, etc.) without returning results."""
        self._logger.debug("executing_command", sql=sql[:200])
        self.client.command(sql)

    def insert(
        self,
        table: str,
        data: list[dict[str, Any]],
        column_names: list[str] | None = None,
    ) -> None:
        """Insert rows into a table."""
        if not data:
            return

        if column_names is None:
            column_names = list(data[0].keys())

        rows = [[row.get(col) for col in column_names] for row in data]

        self._logger.debug(
            "inserting_rows",
            table=table,
            row_count=len(rows),
        )

        self.client.insert(
            table=table,
            data=rows,
            column_names=column_names,
        )

    def insert_df(self, table: str, df: Any) -> None:
        """Insert a pandas DataFrame."""
        self.client.insert_df(table=table, df=df)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Context manager for transaction-like behavior."""
        # ClickHouse doesn't support traditional transactions,
        # but we can use this for consistent logging/error handling
        try:
            yield
        except Exception as e:
            self._logger.error("transaction_failed", error=str(e))
            raise

    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        result = self.query(
            "SELECT 1 FROM system.tables WHERE database = %(db)s AND name = %(table)s",
            parameters={"db": self.database, "table": table},
        )
        return len(result.result_rows) > 0

    def get_row_count(self, table: str) -> int:
        """Get row count for a table."""
        result = self.query(f"SELECT count() FROM {table}")
        return result.result_rows[0][0]

    def close(self) -> None:
        """Close the connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._logger.info("clickhouse_disconnected")


# Singleton instance
_client: ClickHouseClient | None = None


def get_clickhouse_client() -> ClickHouseClient:
    """Get singleton ClickHouse client instance."""
    global _client
    if _client is None:
        _client = ClickHouseClient()
    return _client
