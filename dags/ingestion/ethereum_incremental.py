"""
Incremental Ethereum Block Ingestion DAG.

Continuously ingests new blocks from Ethereum RPC into ClickHouse.
Runs every minute to maintain near real-time data freshness.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

default_args = {
    "owner": "data-team",
    "depends_on_past": True,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=10),
}

with DAG(
    "ethereum_incremental_ingestion",
    default_args=default_args,
    description="Incremental Ethereum block ingestion from RPC to ClickHouse",
    schedule_interval="*/1 * * * *",  # Every minute
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "ethereum", "incremental"],
) as dag:

    @task
    def get_sync_state() -> dict:
        """Get the current sync state from ClickHouse."""
        from src.db.clickhouse import get_clickhouse_client

        client = get_clickhouse_client()

        # Get latest processed block
        result = client.query(
            "SELECT max(block_number) as latest_block FROM ethereum.transactions"
        )
        latest_local = result.result_rows[0][0] or 0

        # Get configured start block if first run
        if latest_local == 0:
            latest_local = int(Variable.get("eth_start_block", default_var=18000000))

        return {
            "latest_local_block": latest_local,
            "batch_size": int(Variable.get("eth_batch_size", default_var=100)),
        }

    @task
    def get_chain_head() -> int:
        """Get the current chain head from RPC."""
        from web3 import Web3

        from config.settings import get_settings

        settings = get_settings()
        w3 = Web3(Web3.HTTPProvider(settings.ethereum.rpc_url))

        # Account for finality (wait for confirmed blocks)
        finality_blocks = settings.ethereum.finality_blocks
        return w3.eth.block_number - finality_blocks

    @task
    def calculate_block_range(sync_state: dict, chain_head: int) -> dict:
        """Calculate the block range to process."""
        start_block = sync_state["latest_local_block"] + 1
        batch_size = sync_state["batch_size"]

        # Don't exceed chain head
        end_block = min(start_block + batch_size - 1, chain_head)

        if start_block > end_block:
            return {"blocks_to_process": 0, "start_block": 0, "end_block": 0}

        return {
            "blocks_to_process": end_block - start_block + 1,
            "start_block": start_block,
            "end_block": end_block,
        }

    @task
    def fetch_blocks(block_range: dict) -> dict:
        """Fetch blocks and transactions from RPC."""
        if block_range["blocks_to_process"] == 0:
            return {"transactions": [], "blocks": []}

        from web3 import Web3

        from config.settings import get_settings

        settings = get_settings()
        w3 = Web3(Web3.HTTPProvider(settings.ethereum.rpc_url))

        blocks = []
        transactions = []

        for block_num in range(block_range["start_block"], block_range["end_block"] + 1):
            block = w3.eth.get_block(block_num, full_transactions=True)

            blocks.append({
                "number": block.number,
                "hash": block.hash.hex(),
                "timestamp": datetime.fromtimestamp(block.timestamp),
                "miner": block.miner,
                "gas_used": block.gasUsed,
                "gas_limit": block.gasLimit,
                "transaction_count": len(block.transactions),
                "base_fee_per_gas": getattr(block, "baseFeePerGas", None),
            })

            for tx in block.transactions:
                transactions.append({
                    "hash": tx.hash.hex(),
                    "block_number": tx.blockNumber,
                    "block_timestamp": datetime.fromtimestamp(block.timestamp),
                    "from_address": tx["from"].lower(),
                    "to_address": (tx.to or "").lower(),
                    "value": tx.value,
                    "gas_used": tx.gas,
                    "gas_price": tx.gasPrice,
                    "input": tx.input.hex() if tx.input else "",
                    "status": 1,  # Will be updated from receipt if needed
                })

        return {
            "transactions": transactions,
            "blocks": blocks,
            "block_range": block_range,
        }

    @task
    def insert_data(data: dict) -> dict:
        """Insert fetched data into ClickHouse."""
        if not data["transactions"] and not data["blocks"]:
            return {"inserted_transactions": 0, "inserted_blocks": 0}

        from src.db.clickhouse import get_clickhouse_client

        client = get_clickhouse_client()

        # Insert blocks
        if data["blocks"]:
            client.insert("ethereum.blocks", data["blocks"])

        # Insert transactions
        if data["transactions"]:
            client.insert("ethereum.transactions", data["transactions"])

        return {
            "inserted_transactions": len(data["transactions"]),
            "inserted_blocks": len(data["blocks"]),
            "block_range": data.get("block_range", {}),
        }

    @task
    def update_metrics(result: dict) -> None:
        """Update ingestion metrics."""
        from prometheus_client import Counter, Gauge

        if result["inserted_blocks"] > 0:
            # Log success
            print(
                f"Ingested {result['inserted_blocks']} blocks, "
                f"{result['inserted_transactions']} transactions"
            )

            block_range = result.get("block_range", {})
            if block_range:
                print(
                    f"Block range: {block_range.get('start_block')} - "
                    f"{block_range.get('end_block')}"
                )

    # Define task dependencies
    sync_state = get_sync_state()
    chain_head = get_chain_head()
    block_range = calculate_block_range(sync_state, chain_head)
    data = fetch_blocks(block_range)
    result = insert_data(data)
    update_metrics(result)
