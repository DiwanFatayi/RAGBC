-- ClickHouse Initialization Script
-- Creates the required database schema for insider detection

-- Create database
CREATE DATABASE IF NOT EXISTS ethereum;

-- Raw transactions table
CREATE TABLE IF NOT EXISTS ethereum.transactions
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

-- Token transfers table (ERC20/721/1155)
CREATE TABLE IF NOT EXISTS ethereum.token_transfers
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

-- Blocks table
CREATE TABLE IF NOT EXISTS ethereum.blocks
(
    number UInt64,
    hash String,
    timestamp DateTime64(3),
    miner String,
    gas_used UInt64,
    gas_limit UInt64,
    transaction_count UInt32,
    base_fee_per_gas Nullable(UInt64)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (number);

-- Events table (listings, partnerships, etc.)
CREATE TABLE IF NOT EXISTS ethereum.events
(
    event_id String,
    event_type Enum8('listing' = 1, 'partnership' = 2, 'launch' = 3, 'airdrop' = 4, 'other' = 5),
    token_address Nullable(String),
    event_timestamp DateTime64(3),
    announced_timestamp Nullable(DateTime64(3)),
    description String,
    source String,
    metadata String -- JSON
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_timestamp)
ORDER BY (event_timestamp, event_id);

-- Daily address activity (materialized view)
CREATE MATERIALIZED VIEW IF NOT EXISTS ethereum.daily_address_activity
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

-- Token daily metrics (materialized view)
CREATE MATERIALIZED VIEW IF NOT EXISTS ethereum.token_daily_metrics
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (token_address, date)
AS SELECT
    token_address,
    toDate(block_timestamp) AS date,
    count() AS transfer_count,
    sum(value) AS total_volume,
    uniqExact(from_address) AS unique_senders,
    uniqExact(to_address) AS unique_receivers
FROM ethereum.token_transfers
GROUP BY token_address, date;

-- Audit log table
CREATE DATABASE IF NOT EXISTS audit;

CREATE TABLE IF NOT EXISTS audit.operations
(
    entry_id String,
    timestamp DateTime64(3),
    operation String,
    user_id String,
    input_hash String,
    output_hash String,
    citations_verified UInt32,
    citations_failed UInt32,
    confidence_score Float32,
    execution_time_ms UInt32,
    metadata String
)
ENGINE = MergeTree()
ORDER BY (timestamp, entry_id)
TTL timestamp + INTERVAL 1 YEAR;

-- Investigation results table
CREATE TABLE IF NOT EXISTS audit.investigations
(
    investigation_id String,
    user_id String,
    query String,
    started_at DateTime64(3),
    completed_at Nullable(DateTime64(3)),
    status Enum8('pending' = 1, 'running' = 2, 'completed' = 3, 'failed' = 4, 'cancelled' = 5),
    confidence_score Nullable(Float32),
    findings_count Nullable(UInt32),
    report String, -- JSON
    error Nullable(String)
)
ENGINE = MergeTree()
ORDER BY (started_at, investigation_id);
