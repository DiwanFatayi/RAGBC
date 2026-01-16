"""
Application settings management using Pydantic Settings.
All configuration is loaded from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    # ClickHouse
    clickhouse_host: str = "localhost"
    clickhouse_port: int = 8123
    clickhouse_user: str = "default"
    clickhouse_password: SecretStr = SecretStr("")
    clickhouse_database: str = "ethereum"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_transactions: str = "transaction_patterns"
    qdrant_collection_addresses: str = "address_patterns"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("password")
    neo4j_database: str = "neo4j"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: SecretStr | None = None
    redis_db: int = 0


class EthereumSettings(BaseSettings):
    """Ethereum RPC and data source settings."""

    model_config = SettingsConfigDict(env_prefix="ETH_")

    rpc_url: str = "https://eth-mainnet.g.alchemy.com/v2/your-api-key"
    rpc_api_key: SecretStr | None = None
    chain_id: int = 1

    # CryptoHouse settings
    cryptohouse_enabled: bool = True
    cryptohouse_host: str = "github.demo.trial.altinity.cloud"
    cryptohouse_port: int = 9440
    cryptohouse_user: str = "demo"
    cryptohouse_password: SecretStr = SecretStr("demo")

    # Block processing
    blocks_per_batch: int = 100
    finality_blocks: int = 64  # Wait for finality before processing


class LLMSettings(BaseSettings):
    """LLM and embedding model settings."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    # vLLM / Ollama endpoint
    endpoint: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen3-30B-A3B-Thinking-2507"
    api_key: SecretStr | None = None

    # Generation parameters
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.95

    # Embedding model
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_device: str = "cuda"  # or "cpu"
    embedding_batch_size: int = 32

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class AgentSettings(BaseSettings):
    """Multi-agent configuration."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    # Retrieval settings
    retrieval_top_k: int = 100
    rrf_k: int = 60  # RRF constant

    # Validation settings
    max_validation_retries: int = 3
    citation_verification_enabled: bool = True

    # Graph analysis
    max_path_hops: int = 5
    cluster_algorithm: Literal["louvain", "label_propagation"] = "louvain"

    # Anomaly detection
    anomaly_z_threshold: float = 3.0
    lookback_days: int = 30

    # Timeouts
    agent_timeout_seconds: int = 300
    db_query_timeout_seconds: int = 60


class APISettings(BaseSettings):
    """FastAPI application settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: list[str] = ["*"]

    # Authentication
    jwt_secret: SecretStr = SecretStr("change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


class ObservabilitySettings(BaseSettings):
    """Monitoring and logging settings."""

    model_config = SettingsConfigDict(env_prefix="OBS_")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # Prometheus
    metrics_enabled: bool = True
    metrics_port: int = 9090

    # Tracing
    tracing_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"

    # LangSmith (optional)
    langsmith_enabled: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "insider-detection"


class Settings(BaseSettings):
    """Root settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ethereum: EthereumSettings = Field(default_factory=EthereumSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
