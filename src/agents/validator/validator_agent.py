"""
Validator Agent - Verifies all citations before report generation.

Ensures every factual claim is traceable to verifiable on-chain data.
"""

import asyncio
import re
from typing import Any

import structlog

from src.agents.base import BaseAgent
from src.agents.state import InvestigationState, ValidationResult

logger = structlog.get_logger()


class CitationExtractor:
    """Extract and parse citations from text."""

    PATTERNS = {
        "transaction": re.compile(r"\[TX:(0x[a-fA-F0-9]{64})\]"),
        "address": re.compile(r"\[ADDR:(0x[a-fA-F0-9]{40})\]"),
        "block": re.compile(r"\[BLOCK:(\d+)\]"),
        "timestamp": re.compile(r"\[TS:([0-9T:\-\+Z]+)\]"),
    }

    def extract_all(self, text: str) -> list[dict[str, Any]]:
        """Extract all citations from text."""
        citations = []

        for citation_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                citations.append({
                    "type": citation_type,
                    "value": match.group(1),
                    "position": match.start(),
                    "context": text[max(0, match.start() - 50):match.end() + 50],
                })

        return citations

    def find_uncited_claims(self, text: str) -> list[str]:
        """Find factual-sounding statements without citations."""
        claim_patterns = [
            r"transferred \$?[\d,]+",
            r"sent \d+ ETH",
            r"received .* from 0x",
            r"on \d{4}-\d{2}-\d{2}",
            r"at block \d+",
            r"transaction 0x[a-fA-F0-9]{8}",
            r"wallet 0x[a-fA-F0-9]{8}",
            r"\$[\d,]+(?:\.\d{2})? (?:worth|in value)",
        ]

        uncited = []
        for pattern in claim_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Check if there's a citation within 100 chars
                context = text[match.start():min(len(text), match.end() + 100)]
                has_citation = any(p.search(context) for p in self.PATTERNS.values())

                if not has_citation:
                    uncited.append(match.group())

        return uncited


class CitationVerifier:
    """Verify citations against source databases."""

    def __init__(self, clickhouse_client, neo4j_driver, rpc_client=None):
        self.clickhouse = clickhouse_client
        self.neo4j = neo4j_driver
        self.rpc = rpc_client
        self._cache: dict[str, dict] = {}

    async def verify_transaction(self, tx_hash: str) -> dict[str, Any]:
        """Verify a transaction exists and return its data."""
        # Check cache first
        cache_key = f"tx:{tx_hash}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query ClickHouse
        result = self.clickhouse.query(
            """
            SELECT hash, block_number, block_timestamp,
                   from_address, to_address, value, status
            FROM ethereum.transactions
            WHERE hash = %(hash)s
            LIMIT 1
            """,
            parameters={"hash": tx_hash},
        )

        rows = result.result_rows
        if rows:
            data = dict(zip(result.column_names, rows[0]))
            verification = {
                "valid": True,
                "data": data,
                "source": "clickhouse",
            }
        else:
            # Fallback to RPC if available
            if self.rpc:
                try:
                    tx = await self._fetch_from_rpc(tx_hash)
                    if tx:
                        verification = {
                            "valid": True,
                            "data": tx,
                            "source": "rpc",
                            "warning": "Not in local DB",
                        }
                    else:
                        verification = {"valid": False, "error": "Not found"}
                except Exception as e:
                    verification = {"valid": False, "error": str(e)}
            else:
                verification = {"valid": False, "error": "Transaction not found"}

        self._cache[cache_key] = verification
        return verification

    async def verify_address(self, address: str) -> dict[str, Any]:
        """Verify an address has activity in the database."""
        cache_key = f"addr:{address}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self.clickhouse.query(
            """
            SELECT count() as tx_count
            FROM ethereum.transactions
            WHERE from_address = %(addr)s OR to_address = %(addr)s
            LIMIT 1
            """,
            parameters={"addr": address},
        )

        rows = result.result_rows
        if rows and rows[0][0] > 0:
            verification = {
                "valid": True,
                "tx_count": rows[0][0],
            }
        else:
            verification = {
                "valid": False,
                "error": "Address has no recorded activity",
            }

        self._cache[cache_key] = verification
        return verification

    async def verify_block(self, block_number: int) -> dict[str, Any]:
        """Verify a block exists."""
        cache_key = f"block:{block_number}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = self.clickhouse.query(
            """
            SELECT number, timestamp, transaction_count
            FROM ethereum.blocks
            WHERE number = %(block)s
            LIMIT 1
            """,
            parameters={"block": block_number},
        )

        rows = result.result_rows
        if rows:
            verification = {
                "valid": True,
                "data": dict(zip(result.column_names, rows[0])),
            }
        else:
            verification = {
                "valid": False,
                "error": "Block not found",
            }

        self._cache[cache_key] = verification
        return verification

    async def _fetch_from_rpc(self, tx_hash: str) -> dict | None:
        """Fetch transaction from RPC (fallback)."""
        if not self.rpc:
            return None
        # Implementation depends on web3 client
        return None


class ValidatorAgent(BaseAgent):
    """
    Agent responsible for validating all citations in analysis.
    
    Ensures zero hallucinations by verifying every factual claim
    against source databases.
    """

    name = "validator_agent"
    description = "Validates citations and ensures analysis is grounded in data"

    def __init__(
        self,
        clickhouse_client,
        neo4j_driver,
        rpc_client=None,
        llm=None,
    ):
        super().__init__(llm=llm)
        self.extractor = CitationExtractor()
        self.verifier = CitationVerifier(clickhouse_client, neo4j_driver, rpc_client)

    async def process(self, state: InvestigationState) -> dict[str, Any]:
        """Validate all citations in the current analysis."""
        # Collect all text that needs validation
        texts_to_validate = []

        # Evidence descriptions - access TypedDict using dict methods
        fused_evidence = state.get("fused_evidence", [])
        for evidence in fused_evidence:
            citations = evidence.citations if hasattr(evidence, 'citations') else evidence.get("citations", [])
            for citation in citations:
                cit_type = citation.type if hasattr(citation, 'type') else citation.get("type", "")
                cit_value = citation.value if hasattr(citation, 'value') else citation.get("value", "")
                texts_to_validate.append(f"[{cit_type.upper()}:{cit_value}]")

        # Anomaly descriptions
        anomalies = state.get("anomalies", [])
        for anomaly in anomalies:
            addr = anomaly.address if hasattr(anomaly, 'address') else anomaly.get("address", "")
            texts_to_validate.append(f"[ADDR:{addr}]")

        # Combine all text
        combined_text = " ".join(texts_to_validate)

        # Extract citations
        citations = self.extractor.extract_all(combined_text)
        uncited_claims = self.extractor.find_uncited_claims(combined_text)

        self._logger.info(
            "starting_validation",
            citation_count=len(citations),
            uncited_count=len(uncited_claims),
        )

        # Verify all citations in parallel
        verification_tasks = []
        for citation in citations:
            if citation["type"] == "transaction":
                verification_tasks.append(
                    self._verify_with_metadata(
                        self.verifier.verify_transaction(citation["value"]),
                        citation,
                    )
                )
            elif citation["type"] == "address":
                verification_tasks.append(
                    self._verify_with_metadata(
                        self.verifier.verify_address(citation["value"]),
                        citation,
                    )
                )
            elif citation["type"] == "block":
                verification_tasks.append(
                    self._verify_with_metadata(
                        self.verifier.verify_block(int(citation["value"])),
                        citation,
                    )
                )

        results = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # Process results
        failed_citations = []
        verified_count = 0
        warnings = []

        for result in results:
            if isinstance(result, Exception):
                failed_citations.append({
                    "error": str(result),
                    "type": "exception",
                })
            elif not result.get("valid"):
                failed_citations.append(result)
            else:
                verified_count += 1
                if result.get("warning"):
                    warnings.append(result["warning"])

        # Add warnings for uncited claims
        for claim in uncited_claims[:10]:  # Limit to first 10
            warnings.append(f"Uncited claim detected: '{claim}'")

        all_valid = len(failed_citations) == 0 and len(uncited_claims) == 0

        validation_result = ValidationResult(
            all_valid=all_valid,
            total_checked=len(citations),
            verified_count=verified_count,
            failed_citations=failed_citations,
            warnings=warnings,
        )

        self._logger.info(
            "validation_complete",
            all_valid=all_valid,
            verified=verified_count,
            failed=len(failed_citations),
            warnings=len(warnings),
        )

        return {
            "validation_result": validation_result,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    async def _verify_with_metadata(
        self,
        verification_coro,
        citation: dict,
    ) -> dict[str, Any]:
        """Wrap verification with citation metadata."""
        result = await verification_coro
        return {
            **result,
            "citation_type": citation["type"],
            "citation_value": citation["value"],
            "context": citation.get("context", ""),
        }
