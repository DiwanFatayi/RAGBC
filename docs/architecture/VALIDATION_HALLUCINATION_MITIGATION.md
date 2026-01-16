# Validation & Hallucination Mitigation Strategy

## Overview

This document details the specific techniques implemented to ensure all LLM outputs are grounded in verifiable on-chain data. The system employs a defense-in-depth approach with multiple validation layers.

**Core Principle:** Every factual claim in system output must be traceable to a specific transaction hash, address, or block number that exists in our databases.

---

## 1. Hallucination Taxonomy for Blockchain Analysis

### 1.1 Types of Hallucinations

| Type | Description | Risk Level | Example |
|------|-------------|------------|---------|
| **Fabricated Transaction** | LLM invents a tx hash that doesn't exist | Critical | "Transaction 0xfake123..." |
| **Incorrect Attribution** | Real tx attributed to wrong address | High | "Address A sent to B" (actually C→D) |
| **Value Fabrication** | Invented or incorrect amounts | High | "$5M transfer" (actually $500K) |
| **Temporal Fabrication** | Wrong timestamps or ordering | Medium | "Before listing" (actually after) |
| **Inference as Fact** | Speculation presented as fact | Medium | "Insider wallet" without evidence |
| **Pattern Fabrication** | Invented patterns not in data | Medium | "Coordinated activity" (random) |

### 1.2 Root Causes

1. **Training data contamination**: LLM may have seen similar cases and "remember" incorrect details
2. **Pattern completion**: LLM fills gaps with plausible-sounding but false data
3. **Aggregation errors**: Combining multiple sources introduces contradictions
4. **Context overflow**: Long contexts cause earlier evidence to be forgotten

---

## 2. Prevention Strategies

### 2.1 Prompt Engineering for Grounding

**System Prompt Template:**
```python
GROUNDED_ANALYSIS_PROMPT = """
You are a blockchain forensics analyst. Your analysis must be STRICTLY GROUNDED.

## CRITICAL RULES

1. **ONLY USE PROVIDED EVIDENCE**: You may only reference data explicitly provided in the EVIDENCE section below. Do not use any knowledge from your training data about specific transactions, addresses, or events.

2. **MANDATORY CITATIONS**: Every factual statement MUST include a citation in one of these formats:
   - Transaction: [TX:0x{64_hex_chars}]
   - Address: [ADDR:0x{40_hex_chars}]
   - Block: [BLOCK:{number}]
   - Timestamp: [TS:{ISO8601}]

3. **NO INFERENCE AS FACT**: If you infer or speculate, you MUST prefix with:
   - "Based on the pattern..." (for reasonable inferences)
   - "Potentially..." or "Possibly..." (for speculation)
   - "INSUFFICIENT EVIDENCE" (when data is lacking)

4. **VERIFICATION AWARENESS**: Your output will be automatically verified. Any citation that cannot be verified against our databases will cause the entire response to be rejected.

## EVIDENCE
{evidence_json}

## QUERY
{user_query}

## RESPONSE FORMAT
Respond in valid JSON matching this schema:
{response_schema}

Remember: It is better to say "insufficient evidence" than to fabricate data.
"""
```

### 2.2 Structured Output Enforcement

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import re

class Citation(BaseModel):
    """A verifiable citation to on-chain data."""
    
    type: Literal["transaction", "address", "block", "timestamp"]
    value: str
    context: str = Field(description="How this citation supports the claim")
    
    @field_validator("value")
    @classmethod
    def validate_citation_format(cls, v, info):
        citation_type = info.data.get("type")
        
        if citation_type == "transaction":
            if not re.match(r"^0x[a-fA-F0-9]{64}$", v):
                raise ValueError(f"Invalid transaction hash format: {v}")
        elif citation_type == "address":
            if not re.match(r"^0x[a-fA-F0-9]{40}$", v):
                raise ValueError(f"Invalid address format: {v}")
        elif citation_type == "block":
            if not v.isdigit():
                raise ValueError(f"Block number must be numeric: {v}")
        elif citation_type == "timestamp":
            try:
                datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError(f"Invalid ISO8601 timestamp: {v}")
        
        return v


class Finding(BaseModel):
    """A single finding with mandatory citations."""
    
    claim: str = Field(description="The factual claim being made")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    citations: list[Citation] = Field(min_length=1, description="Supporting citations")
    is_inference: bool = Field(default=False, description="True if this is inference, not direct observation")
    
    @field_validator("citations")
    @classmethod
    def require_citations(cls, v, info):
        if not info.data.get("is_inference", False) and len(v) == 0:
            raise ValueError("Non-inference claims require at least one citation")
        return v


class AnalysisReport(BaseModel):
    """Complete analysis report with all findings."""
    
    query: str
    summary: str
    findings: list[Finding]
    limitations: list[str]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    
    @field_validator("findings")
    @classmethod
    def validate_findings(cls, v):
        if len(v) == 0:
            raise ValueError("Report must contain at least one finding or explicit 'no findings' statement")
        return v
```

### 2.3 Temperature and Generation Settings

```python
LLM_GENERATION_CONFIG = {
    # Low temperature for factual accuracy
    "temperature": 0.1,
    
    # Disable sampling randomness for deterministic outputs
    "top_p": 0.95,
    
    # Limit response length to reduce drift
    "max_tokens": 4096,
    
    # Stop sequences to prevent runaway generation
    "stop": ["```\n\n", "---\n\n"],
    
    # Enable JSON mode for structured output
    "response_format": {"type": "json_object"},
}
```

---

## 3. Detection Strategies

### 3.1 Citation Extraction Pipeline

```python
import re
from dataclasses import dataclass

@dataclass
class ExtractedCitation:
    citation_type: str
    value: str
    position: int  # Character position in text
    surrounding_text: str  # Context for verification

class CitationExtractor:
    """Extract and parse citations from LLM output."""
    
    PATTERNS = {
        "transaction": re.compile(r"\[TX:(0x[a-fA-F0-9]{64})\]"),
        "address": re.compile(r"\[ADDR:(0x[a-fA-F0-9]{40})\]"),
        "block": re.compile(r"\[BLOCK:(\d+)\]"),
        "timestamp": re.compile(r"\[TS:([0-9T:\-\+Z]+)\]"),
    }
    
    def extract_all(self, text: str) -> list[ExtractedCitation]:
        """Extract all citations from text."""
        citations = []
        
        for citation_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                citations.append(ExtractedCitation(
                    citation_type=citation_type,
                    value=match.group(1),
                    position=match.start(),
                    surrounding_text=text[max(0, match.start()-50):match.end()+50]
                ))
        
        return citations
    
    def find_uncited_claims(self, text: str) -> list[str]:
        """Find factual-sounding statements without citations."""
        
        # Patterns that typically indicate factual claims
        claim_patterns = [
            r"transferred \$?[\d,]+",
            r"sent \d+ ETH",
            r"received .* from",
            r"on \d{4}-\d{2}-\d{2}",
            r"at block \d+",
            r"transaction 0x",
            r"wallet 0x",
        ]
        
        uncited = []
        for pattern in claim_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Check if there's a citation nearby (within 100 chars)
                context = text[match.start():min(len(text), match.end()+100)]
                has_citation = any(p.search(context) for p in self.PATTERNS.values())
                
                if not has_citation:
                    uncited.append(match.group())
        
        return uncited
```

### 3.2 Database Verification

```python
from typing import Protocol
from enum import Enum

class VerificationStatus(Enum):
    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    MISMATCH = "mismatch"
    ERROR = "error"

@dataclass
class VerificationResult:
    status: VerificationStatus
    citation: ExtractedCitation
    actual_data: dict | None
    mismatch_details: str | None

class CitationVerifier:
    """Verify citations against source databases."""
    
    def __init__(
        self,
        clickhouse_client,
        neo4j_driver,
        rpc_client  # Fallback for on-chain verification
    ):
        self.clickhouse = clickhouse_client
        self.neo4j = neo4j_driver
        self.rpc = rpc_client
    
    async def verify_transaction(
        self, 
        tx_hash: str,
        claimed_context: str
    ) -> VerificationResult:
        """Verify a transaction citation exists and context matches."""
        
        # Primary: Check ClickHouse
        result = self.clickhouse.query(
            """
            SELECT 
                hash, block_number, block_timestamp,
                from_address, to_address, value, status
            FROM ethereum.transactions 
            WHERE hash = %(hash)s
            """,
            {"hash": tx_hash}
        ).first_row_or_none()
        
        if result:
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                citation=ExtractedCitation("transaction", tx_hash, 0, claimed_context),
                actual_data=dict(result),
                mismatch_details=None
            )
        
        # Fallback: Check on-chain via RPC (slower but authoritative)
        try:
            tx = await self.rpc.get_transaction(tx_hash)
            if tx:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    citation=ExtractedCitation("transaction", tx_hash, 0, claimed_context),
                    actual_data=tx,
                    mismatch_details="Found on-chain but not in local DB"
                )
        except Exception as e:
            pass
        
        return VerificationResult(
            status=VerificationStatus.NOT_FOUND,
            citation=ExtractedCitation("transaction", tx_hash, 0, claimed_context),
            actual_data=None,
            mismatch_details=f"Transaction {tx_hash} not found in any source"
        )
    
    async def verify_address_claim(
        self,
        address: str,
        claimed_activity: dict
    ) -> VerificationResult:
        """Verify claims about an address's activity."""
        
        # Example: Verify "address received $1M in TOKEN"
        if "received_amount" in claimed_activity:
            actual = self.clickhouse.query(
                """
                SELECT sum(value) as total_received
                FROM ethereum.token_transfers
                WHERE to_address = %(addr)s
                  AND token_address = %(token)s
                  AND block_timestamp BETWEEN %(start)s AND %(end)s
                """,
                {
                    "addr": address,
                    "token": claimed_activity.get("token"),
                    "start": claimed_activity.get("start_time"),
                    "end": claimed_activity.get("end_time"),
                }
            ).first_row()
            
            claimed = claimed_activity["received_amount"]
            tolerance = 0.01  # 1% tolerance for rounding
            
            if abs(actual["total_received"] - claimed) / claimed > tolerance:
                return VerificationResult(
                    status=VerificationStatus.MISMATCH,
                    citation=ExtractedCitation("address", address, 0, str(claimed_activity)),
                    actual_data={"total_received": actual["total_received"]},
                    mismatch_details=f"Claimed {claimed}, actual {actual['total_received']}"
                )
        
        return VerificationResult(
            status=VerificationStatus.VERIFIED,
            citation=ExtractedCitation("address", address, 0, str(claimed_activity)),
            actual_data=None,
            mismatch_details=None
        )
    
    async def verify_all(
        self,
        citations: list[ExtractedCitation]
    ) -> list[VerificationResult]:
        """Verify all citations in parallel."""
        
        tasks = []
        for citation in citations:
            if citation.citation_type == "transaction":
                tasks.append(self.verify_transaction(citation.value, citation.surrounding_text))
            elif citation.citation_type == "address":
                tasks.append(self.verify_address_claim(citation.value, {}))
            elif citation.citation_type == "block":
                tasks.append(self.verify_block(int(citation.value)))
        
        return await asyncio.gather(*tasks)
```

---

## 4. Correction Strategies

### 4.1 Iterative Refinement Loop

```python
class ValidationLoop:
    """Implements iterative refinement for hallucination correction."""
    
    MAX_ITERATIONS = 3
    
    def __init__(
        self,
        llm,
        extractor: CitationExtractor,
        verifier: CitationVerifier
    ):
        self.llm = llm
        self.extractor = extractor
        self.verifier = verifier
    
    async def generate_with_validation(
        self,
        prompt: str,
        evidence: dict
    ) -> tuple[str, list[VerificationResult]]:
        """Generate response with validation loop."""
        
        for iteration in range(self.MAX_ITERATIONS):
            # Generate response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content
            
            # Extract citations
            citations = self.extractor.extract_all(response_text)
            
            # Check for uncited claims
            uncited = self.extractor.find_uncited_claims(response_text)
            
            # Verify all citations
            verification_results = await self.verifier.verify_all(citations)
            
            # Check for failures
            failures = [r for r in verification_results if r.status != VerificationStatus.VERIFIED]
            
            if not failures and not uncited:
                # All validations passed
                return response_text, verification_results
            
            # Build correction prompt
            correction_prompt = self._build_correction_prompt(
                original_response=response_text,
                failures=failures,
                uncited_claims=uncited,
                evidence=evidence
            )
            
            prompt = correction_prompt
        
        # Max iterations reached - return with warnings
        return self._add_validation_warnings(response_text, failures, uncited), verification_results
    
    def _build_correction_prompt(
        self,
        original_response: str,
        failures: list[VerificationResult],
        uncited_claims: list[str],
        evidence: dict
    ) -> str:
        """Build prompt for correction iteration."""
        
        correction_details = []
        
        for failure in failures:
            if failure.status == VerificationStatus.NOT_FOUND:
                correction_details.append(
                    f"INVALID CITATION: {failure.citation.value} does not exist. "
                    f"Remove this reference or find correct citation in evidence."
                )
            elif failure.status == VerificationStatus.MISMATCH:
                correction_details.append(
                    f"DATA MISMATCH: {failure.citation.value} - {failure.mismatch_details}. "
                    f"Correct the claim to match actual data."
                )
        
        for claim in uncited_claims:
            correction_details.append(
                f"UNCITED CLAIM: '{claim}' requires a citation. "
                f"Add [TX:], [ADDR:], or [BLOCK:] citation, or rephrase as inference."
            )
        
        return f"""
Your previous response contained validation errors that must be corrected:

ERRORS:
{chr(10).join(f"- {d}" for d in correction_details)}

ORIGINAL RESPONSE:
{original_response}

AVAILABLE EVIDENCE:
{json.dumps(evidence, indent=2)}

Please regenerate your response, correcting all errors above.
Remember: Only cite data that exists in the AVAILABLE EVIDENCE section.
"""
```

### 4.2 Confidence Scoring

```python
class ConfidenceScorer:
    """Compute confidence scores based on evidence quality."""
    
    def score_finding(self, finding: Finding, verification_results: list[VerificationResult]) -> float:
        """Score confidence for a single finding."""
        
        base_score = 1.0
        
        # Penalty for inferences
        if finding.is_inference:
            base_score *= 0.7
        
        # Penalty for few citations
        citation_count = len(finding.citations)
        if citation_count < 2:
            base_score *= 0.8
        elif citation_count >= 5:
            base_score *= 1.1  # Bonus for well-supported claims
        
        # Penalty for verification issues
        relevant_verifications = [
            v for v in verification_results 
            if v.citation.value in [c.value for c in finding.citations]
        ]
        
        for v in relevant_verifications:
            if v.status == VerificationStatus.MISMATCH:
                base_score *= 0.5
            elif v.status == VerificationStatus.NOT_FOUND:
                base_score *= 0.1  # Severe penalty
        
        # Bonus for multiple independent sources
        source_types = set(c.type for c in finding.citations)
        if len(source_types) >= 2:
            base_score *= 1.1
        
        return min(1.0, max(0.0, base_score))
    
    def score_report(self, report: AnalysisReport, verification_results: list[VerificationResult]) -> float:
        """Score overall report confidence."""
        
        if not report.findings:
            return 0.0
        
        finding_scores = [
            self.score_finding(f, verification_results) 
            for f in report.findings
        ]
        
        # Weighted average favoring lower scores (conservative)
        weights = [1.0 / (i + 1) for i in range(len(finding_scores))]
        sorted_scores = sorted(finding_scores)
        
        weighted_sum = sum(s * w for s, w in zip(sorted_scores, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum
```

---

## 5. Data Integrity Mechanisms

### 5.1 Checksum Validation

```python
import hashlib
from datetime import datetime

class DataIntegrityChecker:
    """Ensure data consistency across databases."""
    
    def __init__(self, clickhouse, qdrant, neo4j):
        self.clickhouse = clickhouse
        self.qdrant = qdrant
        self.neo4j = neo4j
    
    def compute_block_range_checksum(self, start_block: int, end_block: int) -> str:
        """Compute checksum for a block range in ClickHouse."""
        
        result = self.clickhouse.query(
            """
            SELECT 
                count() as tx_count,
                sum(cityHash64(hash)) as hash_sum,
                min(block_timestamp) as min_ts,
                max(block_timestamp) as max_ts
            FROM ethereum.transactions
            WHERE block_number BETWEEN %(start)s AND %(end)s
            """,
            {"start": start_block, "end": end_block}
        ).first_row()
        
        checksum_input = f"{result['tx_count']}:{result['hash_sum']}:{result['min_ts']}:{result['max_ts']}"
        return hashlib.sha256(checksum_input.encode()).hexdigest()
    
    def verify_cross_db_consistency(self, sample_size: int = 1000) -> dict:
        """Verify consistency between databases with sampling."""
        
        # Sample transactions from ClickHouse
        samples = self.clickhouse.query(
            f"""
            SELECT hash, from_address, to_address, block_number
            FROM ethereum.transactions
            ORDER BY rand()
            LIMIT {sample_size}
            """
        ).result_rows
        
        results = {
            "total_checked": sample_size,
            "qdrant_missing": 0,
            "neo4j_missing": 0,
            "mismatches": []
        }
        
        for tx in samples:
            tx_hash, from_addr, to_addr, block = tx
            
            # Check Qdrant
            qdrant_result = self.qdrant.scroll(
                collection_name="transaction_patterns",
                scroll_filter={"must": [{"key": "tx_hash", "match": {"value": tx_hash}}]},
                limit=1
            )
            if not qdrant_result[0]:
                results["qdrant_missing"] += 1
            
            # Check Neo4j (relationship exists)
            neo4j_result = self.neo4j.query(
                """
                MATCH (a:Wallet {address: $from})-[r:TRANSFERRED]->(b:Wallet {address: $to})
                WHERE r.tx_hash = $hash
                RETURN count(r) as count
                """,
                {"from": from_addr, "to": to_addr, "hash": tx_hash}
            )
            if neo4j_result[0]["count"] == 0:
                results["neo4j_missing"] += 1
        
        results["consistency_score"] = 1.0 - (
            (results["qdrant_missing"] + results["neo4j_missing"]) / (2 * sample_size)
        )
        
        return results
```

### 5.2 Audit Trail

```python
from pydantic import BaseModel
from datetime import datetime
import uuid

class AuditEntry(BaseModel):
    """Audit trail entry for all system operations."""
    
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operation: str  # "query", "generation", "validation", "report"
    user_id: str
    input_hash: str  # SHA256 of input
    output_hash: str  # SHA256 of output
    citations_verified: int
    citations_failed: int
    confidence_score: float
    execution_time_ms: int
    metadata: dict

class AuditLogger:
    """Log all operations for auditability."""
    
    def __init__(self, clickhouse):
        self.clickhouse = clickhouse
        self._ensure_table()
    
    def _ensure_table(self):
        self.clickhouse.execute("""
            CREATE TABLE IF NOT EXISTS audit.operations (
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
                metadata String  -- JSON
            ) ENGINE = MergeTree()
            ORDER BY (timestamp, entry_id)
            TTL timestamp + INTERVAL 1 YEAR
        """)
    
    def log(self, entry: AuditEntry):
        self.clickhouse.execute(
            """
            INSERT INTO audit.operations VALUES
            """,
            [entry.model_dump()]
        )
```

---

## 6. Monitoring & Alerting

### 6.1 Hallucination Metrics

```python
# Prometheus metrics for hallucination tracking
from prometheus_client import Counter, Histogram, Gauge

HALLUCINATION_METRICS = {
    "citations_total": Counter(
        "citations_total",
        "Total citations extracted",
        ["citation_type"]
    ),
    "citations_verified": Counter(
        "citations_verified",
        "Citations successfully verified",
        ["citation_type"]
    ),
    "citations_failed": Counter(
        "citations_failed",
        "Citations that failed verification",
        ["citation_type", "failure_reason"]
    ),
    "uncited_claims": Counter(
        "uncited_claims_total",
        "Claims without citations detected"
    ),
    "validation_iterations": Histogram(
        "validation_iterations",
        "Number of validation iterations needed",
        buckets=[1, 2, 3, 4, 5]
    ),
    "confidence_score": Histogram(
        "report_confidence_score",
        "Distribution of report confidence scores",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ),
}
```

### 6.2 Alert Rules

```yaml
# prometheus/alerts/hallucination_alerts.yml
groups:
  - name: hallucination_alerts
    rules:
      - alert: HighHallucinationRate
        expr: |
          rate(citations_failed_total[5m]) / rate(citations_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High hallucination rate detected"
          description: "More than 10% of citations are failing verification"

      - alert: LowConfidenceReports
        expr: |
          histogram_quantile(0.5, rate(report_confidence_score_bucket[1h])) < 0.6
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Median report confidence is low"
          description: "50% of reports have confidence below 0.6"

      - alert: ValidationLoopExhausted
        expr: |
          rate(validation_iterations_bucket{le="3"}[5m]) / rate(validation_iterations_count[5m]) < 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Many requests exhausting validation retries"
          description: "More than 10% of requests need max validation iterations"
```

---

## 7. Testing Strategy

### 7.1 Hallucination Test Cases

```python
import pytest

class TestHallucinationDetection:
    """Test suite for hallucination detection and mitigation."""
    
    @pytest.fixture
    def known_hallucinations(self):
        """Examples of known hallucination patterns."""
        return [
            {
                "input": "Analyze wallet 0xabc...",
                "bad_output": "Wallet sent $5M [TX:0x000...000]",  # Fake tx
                "expected_detection": "NOT_FOUND"
            },
            {
                "input": "Find insider activity for TOKEN",
                "bad_output": "Major accumulation on 2024-01-15",  # No citation
                "expected_detection": "UNCITED_CLAIM"
            },
        ]
    
    async def test_fake_transaction_detected(self, verifier):
        """Verify system detects fabricated transaction hashes."""
        fake_tx = "0x" + "0" * 64
        result = await verifier.verify_transaction(fake_tx, "test context")
        assert result.status == VerificationStatus.NOT_FOUND
    
    async def test_uncited_claim_detected(self, extractor):
        """Verify system detects claims without citations."""
        text = "The wallet transferred $1M to Binance on January 15th."
        uncited = extractor.find_uncited_claims(text)
        assert len(uncited) > 0
    
    async def test_valid_citation_passes(self, verifier, real_transaction):
        """Verify real transactions pass validation."""
        result = await verifier.verify_transaction(
            real_transaction["hash"],
            "test context"
        )
        assert result.status == VerificationStatus.VERIFIED
    
    async def test_correction_loop_fixes_hallucination(self, validation_loop):
        """Verify correction loop can fix hallucinations."""
        # Inject a prompt that might cause hallucination
        prompt = "..."
        evidence = {"transactions": [...]}  # Real evidence
        
        output, results = await validation_loop.generate_with_validation(prompt, evidence)
        
        # All citations should be verified after correction
        failures = [r for r in results if r.status != VerificationStatus.VERIFIED]
        assert len(failures) == 0
```

---

## 8. Summary: Defense-in-Depth Layers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HALLUCINATION DEFENSE LAYERS                        │
└─────────────────────────────────────────────────────────────────────────┘

Layer 1: PREVENTION (Prompt Engineering)
├── Strict grounding instructions in system prompt
├── Mandatory citation format requirements
├── Low temperature generation (0.1)
└── Structured output schemas with validation

Layer 2: DETECTION (Citation Extraction)
├── Regex-based citation parser
├── Uncited claim detector
└── Format validation (hash lengths, address checksums)

Layer 3: VERIFICATION (Database Checks)
├── Primary: ClickHouse lookup
├── Fallback: On-chain RPC verification
├── Cross-reference: Neo4j relationship check
└── Value/amount tolerance checking

Layer 4: CORRECTION (Iterative Refinement)
├── Up to 3 correction iterations
├── Specific error feedback in prompts
├── Fallback to "insufficient evidence" response
└── Warning annotations for unresolved issues

Layer 5: MONITORING (Metrics & Alerts)
├── Citation success/failure rates
├── Confidence score distribution
├── Alert on high hallucination rates
└── Audit trail for all operations

Layer 6: TESTING (Continuous Validation)
├── Known hallucination test cases
├── Regression tests for fixed issues
├── Fuzzing with adversarial prompts
└── Regular validation against known cases
```
