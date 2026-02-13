"""Factuality evaluator for verifying factual accuracy of agent outputs.

This module provides an evaluator that:
- Extracts factual claims from agent outputs
- Verifies claims against ground truth data
- Uses LLM-based verification when needed
- Extracts and validates citations
- Detects potential hallucinations
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator

if TYPE_CHECKING:
    from atp.analytics.cost import CostTracker

logger = logging.getLogger(__name__)


class VerificationMethod(str, Enum):
    """Method for verifying factual claims."""

    RAG = "rag"
    LLM_VERIFY = "llm_verify"
    GROUND_TRUTH = "ground_truth"


class ClaimType(str, Enum):
    """Type of factual claim."""

    DATE = "date"
    NUMBER = "number"
    NAME = "name"
    FACT = "fact"
    QUOTE = "quote"
    STATISTIC = "statistic"


@dataclass
class Claim:
    """Represents an extracted factual claim."""

    text: str
    claim_type: ClaimType
    confidence: float = 0.0
    verified: bool | None = None
    verification_source: str | None = None
    evidence: str | None = None
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert claim to dictionary."""
        return {
            "text": self.text,
            "type": self.claim_type.value,
            "confidence": self.confidence,
            "verified": self.verified,
            "verification_source": self.verification_source,
            "evidence": self.evidence,
            "location": self.location,
        }


@dataclass
class Citation:
    """Represents an extracted citation."""

    text: str
    source: str | None = None
    url: str | None = None
    valid: bool | None = None
    location: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "url": self.url,
            "valid": self.valid,
            "location": self.location,
        }


@dataclass
class HallucinationIndicator:
    """Represents a potential hallucination indicator."""

    indicator_type: str
    description: str
    severity: str  # low, medium, high
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert indicator to dictionary."""
        return {
            "type": self.indicator_type,
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence,
        }


@dataclass
class FactualityResult:
    """Result of factuality analysis."""

    claims: list[Claim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    hallucination_indicators: list[HallucinationIndicator] = field(default_factory=list)
    overall_score: float = 1.0
    verified_claims_count: int = 0
    unverified_claims_count: int = 0
    false_claims_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "claims": [c.to_dict() for c in self.claims],
            "citations": [c.to_dict() for c in self.citations],
            "hallucination_indicators": [
                h.to_dict() for h in self.hallucination_indicators
            ],
            "overall_score": self.overall_score,
            "verified_claims_count": self.verified_claims_count,
            "unverified_claims_count": self.unverified_claims_count,
            "false_claims_count": self.false_claims_count,
        }


class FactualityConfig(BaseModel):
    """Configuration for Factuality evaluator."""

    api_key: str | None = Field(None, description="Anthropic API key")
    model: str = Field(
        "claude-sonnet-4-20250514", description="Model to use for evaluation"
    )
    temperature: float = Field(0.0, ge=0.0, le=1.0, description="Model temperature")
    max_tokens: int = Field(2048, ge=1, description="Max tokens for response")
    timeout: float = Field(60.0, gt=0, description="Timeout per request in seconds")
    enable_cost_tracking: bool = Field(
        True, description="Enable cost tracking via CostTracker"
    )


class ClaimExtractor:
    """Extracts factual claims from text."""

    # Patterns for different claim types
    DATE_PATTERNS = [
        r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}\b",
        r"\b(in|on|since|from|until|by)\s+\d{4}\b",
    ]

    NUMBER_PATTERNS = [
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:percent|%)\b",
        r"\b\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|trillion)?\b",
        r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion|trillion)\b",
        r"\b(\d+(?:,\d{3})*)\s+(?:people|users|customers|employees)\b",
    ]

    QUOTE_PATTERNS = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r"\u201c([^\u201d]+)\u201d",  # Curly quotes
    ]

    def extract_claims(self, text: str) -> list[Claim]:
        """Extract factual claims from text.

        Args:
            text: The text to extract claims from.

        Returns:
            List of extracted claims.
        """
        claims: list[Claim] = []
        claims.extend(self._extract_date_claims(text))
        claims.extend(self._extract_number_claims(text))
        claims.extend(self._extract_quote_claims(text))
        claims.extend(self._extract_factual_statements(text))
        return claims

    def _extract_date_claims(self, text: str) -> list[Claim]:
        """Extract date-related claims."""
        claims = []
        for pattern in self.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_surrounding_context(
                    text, match.start(), match.end()
                )
                if context and self._is_factual_context(context):
                    claims.append(
                        Claim(
                            text=context,
                            claim_type=ClaimType.DATE,
                            location=f"char:{match.start()}-{match.end()}",
                        )
                    )
        return self._deduplicate_claims(claims)

    def _extract_number_claims(self, text: str) -> list[Claim]:
        """Extract number-related claims."""
        claims = []
        for pattern in self.NUMBER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_surrounding_context(
                    text, match.start(), match.end()
                )
                if context:
                    claim_type = (
                        ClaimType.STATISTIC
                        if "percent" in context.lower() or "%" in context
                        else ClaimType.NUMBER
                    )
                    claims.append(
                        Claim(
                            text=context,
                            claim_type=claim_type,
                            location=f"char:{match.start()}-{match.end()}",
                        )
                    )
        return self._deduplicate_claims(claims)

    def _extract_quote_claims(self, text: str) -> list[Claim]:
        """Extract quoted claims."""
        claims = []
        for pattern in self.QUOTE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                quote = match.group(1) if match.groups() else match.group(0)
                if len(quote) > 10:  # Skip very short quotes
                    context = self._get_surrounding_context(
                        text, match.start(), match.end(), max_chars=200
                    )
                    claims.append(
                        Claim(
                            text=context if context else quote,
                            claim_type=ClaimType.QUOTE,
                            location=f"char:{match.start()}-{match.end()}",
                        )
                    )
        return claims

    def _extract_factual_statements(self, text: str) -> list[Claim]:
        """Extract general factual statements using sentence analysis."""
        claims = []
        factual_indicators = [
            r"\bis\s+(?:the|a|an)\b",
            r"\bwas\s+(?:the|a|an)\b",
            r"\bfounded\s+(?:in|by)\b",
            r"\binvented\s+(?:in|by)\b",
            r"\bdiscovered\s+(?:in|by)\b",
            r"\baccording\s+to\b",
            r"\bresearch\s+shows\b",
            r"\bstudies\s+(?:show|indicate|suggest)\b",
            r"\bdata\s+(?:shows|indicates|suggests)\b",
        ]

        sentences = self._split_into_sentences(text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            for pattern in factual_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append(
                        Claim(
                            text=sentence,
                            claim_type=ClaimType.FACT,
                        )
                    )
                    break

        return self._deduplicate_claims(claims)

    def _get_surrounding_context(
        self, text: str, start: int, end: int, max_chars: int = 150
    ) -> str | None:
        """Get surrounding context for a match."""
        sentence_start = max(0, text.rfind(".", 0, start) + 1)
        sentence_end = text.find(".", end)
        if sentence_end == -1:
            sentence_end = len(text)
        else:
            sentence_end += 1

        context = text[sentence_start:sentence_end].strip()
        if len(context) > max_chars:
            half = max_chars // 2
            ctx_start = max(0, start - sentence_start - half)
            ctx_end = min(len(context), end - sentence_start + half)
            context = context[ctx_start:ctx_end].strip()

        return context if context else None

    def _is_factual_context(self, context: str) -> bool:
        """Check if context likely contains a factual claim."""
        non_factual_patterns = [
            r"\bwill\s+be\b",
            r"\bmight\b",
            r"\bcould\b",
            r"\bshould\b",
            r"\bwould\b",
            r"\bif\b",
            r"\bassuming\b",
        ]
        for pattern in non_factual_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return False
        return True

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        return sentence_endings.split(text)

    def _deduplicate_claims(self, claims: list[Claim]) -> list[Claim]:
        """Remove duplicate claims based on text similarity."""
        seen_texts: set[str] = set()
        unique_claims: list[Claim] = []
        for claim in claims:
            normalized = claim.text.lower().strip()
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_claims.append(claim)
        return unique_claims


class CitationExtractor:
    """Extracts and validates citations from text."""

    URL_PATTERN = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
        r"(?:/(?:[-\w._~:/?#\[\]@!$&'()*+,;=%]|(?:%[\da-fA-F]{2}))*)?",
        re.IGNORECASE,
    )

    CITATION_PATTERNS = [
        r"\[(\d+)\]",  # [1], [2], etc.
        r"\(([^)]+,\s*\d{4})\)",  # (Author, 2023)
        r"(?:Source|Reference|Citation):\s*([^\n]+)",
        r"(?:According to|As stated by|Per)\s+([^,.\n]+)",
    ]

    def extract_citations(self, text: str) -> list[Citation]:
        """Extract citations from text.

        Args:
            text: The text to extract citations from.

        Returns:
            List of extracted citations.
        """
        citations: list[Citation] = []
        citations.extend(self._extract_urls(text))
        citations.extend(self._extract_reference_citations(text))
        return citations

    def _extract_urls(self, text: str) -> list[Citation]:
        """Extract URL citations."""
        citations = []
        matches = self.URL_PATTERN.finditer(text)
        for match in matches:
            url = match.group(0)
            citations.append(
                Citation(
                    text=url,
                    url=url,
                    location=f"char:{match.start()}-{match.end()}",
                )
            )
        return citations

    def _extract_reference_citations(self, text: str) -> list[Citation]:
        """Extract reference-style citations."""
        citations = []
        for pattern in self.CITATION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation_text = match.group(1) if match.groups() else match.group(0)
                citations.append(
                    Citation(
                        text=citation_text,
                        source=citation_text,
                        location=f"char:{match.start()}-{match.end()}",
                    )
                )
        return citations

    def validate_citations(
        self,
        citations: list[Citation],
        ground_truth: dict[str, Any] | None = None,
    ) -> list[Citation]:
        """Validate extracted citations.

        Args:
            citations: List of citations to validate.
            ground_truth: Optional ground truth data for validation.

        Returns:
            List of citations with validation status.
        """
        validated_citations = []
        valid_sources = set()
        if ground_truth and "valid_sources" in ground_truth:
            valid_sources = set(ground_truth["valid_sources"])

        for citation in citations:
            citation_copy = Citation(
                text=citation.text,
                source=citation.source,
                url=citation.url,
                location=citation.location,
            )
            if citation_copy.url:
                citation_copy.valid = self._is_valid_url_format(citation_copy.url)
            elif citation_copy.source and valid_sources:
                citation_copy.valid = citation_copy.source in valid_sources
            validated_citations.append(citation_copy)

        return validated_citations

    def _is_valid_url_format(self, url: str) -> bool:
        """Check if URL has valid format."""
        return bool(self.URL_PATTERN.match(url))


class HallucinationDetector:
    """Detects potential hallucinations in text."""

    VAGUE_LANGUAGE_PATTERNS = [
        (r"\bmany\s+(?:people|experts|studies)\b", "Vague quantifier without source"),
        (
            r"\bit\s+is\s+(?:well[-\s]?known|widely\s+accepted)\b",
            "Unsourced common knowledge claim",
        ),
        (r"\bsome\s+(?:say|believe|argue)\b", "Vague attribution"),
        (r"\bresearch\s+suggests\b", "Unspecified research reference"),
        (r"\bstudies\s+(?:have\s+)?show(?:n|s)?\b", "Unspecified studies reference"),
    ]

    CONFIDENCE_PATTERNS = [
        (r"\bdefinitely\b", "Overconfident language"),
        (r"\bcertainly\b", "Overconfident language"),
        (r"\bundoubtedly\b", "Overconfident language"),
        (r"\bwithout\s+(?:a\s+)?doubt\b", "Overconfident language"),
        (r"\balways\b", "Absolute statement"),
        (r"\bnever\b", "Absolute statement"),
        (r"\beveryone\b", "Universal claim"),
        (r"\bno\s+one\b", "Universal negative claim"),
    ]

    SPECIFIC_BUT_UNVERIFIABLE = [
        (r"\b\d+(?:\.\d+)?%\s+of\b", "Specific percentage without source"),
        (r"\bexactly\s+\d+\b", "Exact number claim"),
        (r"\bprecisely\s+\d+\b", "Precise number claim"),
    ]

    def detect(self, text: str, claims: list[Claim]) -> list[HallucinationIndicator]:
        """Detect potential hallucination indicators.

        Args:
            text: The text to analyze.
            claims: List of extracted claims for context.

        Returns:
            List of hallucination indicators.
        """
        indicators: list[HallucinationIndicator] = []
        indicators.extend(self._detect_vague_language(text))
        indicators.extend(self._detect_overconfidence(text))
        indicators.extend(self._detect_unverifiable_specifics(text))
        indicators.extend(self._detect_inconsistencies(claims))
        return indicators

    def _detect_vague_language(self, text: str) -> list[HallucinationIndicator]:
        """Detect vague language patterns."""
        indicators = []
        for pattern, description in self.VAGUE_LANGUAGE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    HallucinationIndicator(
                        indicator_type="vague_language",
                        description=description,
                        severity="low",
                        evidence=match.group(0),
                    )
                )
        return indicators

    def _detect_overconfidence(self, text: str) -> list[HallucinationIndicator]:
        """Detect overconfident language."""
        indicators = []
        for pattern, description in self.CONFIDENCE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(
                    HallucinationIndicator(
                        indicator_type="overconfidence",
                        description=description,
                        severity="medium",
                        evidence=match.group(0),
                    )
                )
        return indicators

    def _detect_unverifiable_specifics(self, text: str) -> list[HallucinationIndicator]:
        """Detect specific but unverifiable claims."""
        indicators = []
        for pattern, description in self.SPECIFIC_BUT_UNVERIFIABLE:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                context = self._get_context(text, match.start(), match.end())
                if not self._has_citation_nearby(text, match.start(), match.end()):
                    indicators.append(
                        HallucinationIndicator(
                            indicator_type="unverifiable_specific",
                            description=description,
                            severity="medium",
                            evidence=context,
                        )
                    )
        return indicators

    def _detect_inconsistencies(
        self, claims: list[Claim]
    ) -> list[HallucinationIndicator]:
        """Detect inconsistencies between claims."""
        indicators = []
        date_claims = [c for c in claims if c.claim_type == ClaimType.DATE]
        number_claims = [
            c for c in claims if c.claim_type in (ClaimType.NUMBER, ClaimType.STATISTIC)
        ]

        if len(date_claims) > 1:
            for i, claim1 in enumerate(date_claims):
                for claim2 in date_claims[i + 1 :]:
                    if self._dates_conflict(claim1.text, claim2.text):
                        indicators.append(
                            HallucinationIndicator(
                                indicator_type="inconsistency",
                                description="Potentially conflicting dates",
                                severity="high",
                                evidence=f"'{claim1.text}' vs '{claim2.text}'",
                            )
                        )

        if len(number_claims) > 1:
            for i, claim1 in enumerate(number_claims):
                for claim2 in number_claims[i + 1 :]:
                    if self._numbers_conflict(claim1.text, claim2.text):
                        indicators.append(
                            HallucinationIndicator(
                                indicator_type="inconsistency",
                                description="Potentially conflicting numbers",
                                severity="high",
                                evidence=f"'{claim1.text}' vs '{claim2.text}'",
                            )
                        )

        return indicators

    def _get_context(self, text: str, start: int, end: int, chars: int = 50) -> str:
        """Get context around a match."""
        ctx_start = max(0, start - chars)
        ctx_end = min(len(text), end + chars)
        return text[ctx_start:ctx_end]

    def _has_citation_nearby(
        self, text: str, start: int, end: int, window: int = 100
    ) -> bool:
        """Check if there's a citation near the match."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        context = text[ctx_start:ctx_end]
        citation_patterns = [r"\[\d+\]", r"\([^)]+,\s*\d{4}\)", r"https?://"]
        for pattern in citation_patterns:
            if re.search(pattern, context):
                return True
        return False

    def _dates_conflict(self, text1: str, text2: str) -> bool:
        """Check if two date claims might conflict."""
        years1 = set(re.findall(r"\b(?:19|20)\d{2}\b", text1))
        years2 = set(re.findall(r"\b(?:19|20)\d{2}\b", text2))
        if years1 and years2:
            common_context = self._extract_context_keywords(
                text1
            ) & self._extract_context_keywords(text2)
            if common_context and not years1.intersection(years2):
                return True
        return False

    def _numbers_conflict(self, text1: str, text2: str) -> bool:
        """Check if two number claims might conflict."""
        common_context = self._extract_context_keywords(
            text1
        ) & self._extract_context_keywords(text2)
        if not common_context:
            return False
        numbers1 = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text1))
        numbers2 = set(re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", text2))
        if numbers1 and numbers2 and not numbers1.intersection(numbers2):
            return True
        return False

    def _extract_context_keywords(self, text: str) -> set[str]:
        """Extract context keywords from text."""
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        stop_words = {
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "were",
            "their",
            "about",
        }
        return set(words) - stop_words


class GroundTruthVerifier:
    """Verifies claims against ground truth data."""

    def __init__(self, ground_truth_path: str | Path | None = None) -> None:
        """Initialize with optional ground truth file path.

        Args:
            ground_truth_path: Path to JSON file containing ground truth.
        """
        self._ground_truth: dict[str, Any] = {}
        if ground_truth_path:
            self.load_ground_truth(ground_truth_path)

    def load_ground_truth(self, path: str | Path) -> None:
        """Load ground truth from JSON file.

        Args:
            path: Path to the ground truth JSON file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file isn't valid JSON.
        """
        path = Path(path)
        with open(path) as f:
            self._ground_truth = json.load(f)

    def set_ground_truth(self, data: dict[str, Any]) -> None:
        """Set ground truth data directly.

        Args:
            data: Dictionary containing ground truth facts.
        """
        self._ground_truth = data

    def verify_claims(self, claims: list[Claim]) -> list[Claim]:
        """Verify claims against ground truth.

        Args:
            claims: List of claims to verify.

        Returns:
            List of claims with verification status updated.
        """
        if not self._ground_truth:
            return claims

        verified_claims = []
        facts = self._ground_truth.get("facts", {})
        dates = self._ground_truth.get("dates", {})
        numbers = self._ground_truth.get("numbers", {})

        for claim in claims:
            claim_copy = Claim(
                text=claim.text,
                claim_type=claim.claim_type,
                confidence=claim.confidence,
                location=claim.location,
            )

            if claim_copy.claim_type == ClaimType.DATE:
                verified, evidence = self._verify_date_claim(claim_copy.text, dates)
                claim_copy.verified = verified
                claim_copy.evidence = evidence
                claim_copy.verification_source = "ground_truth"

            elif claim_copy.claim_type in (ClaimType.NUMBER, ClaimType.STATISTIC):
                verified, evidence = self._verify_number_claim(claim_copy.text, numbers)
                claim_copy.verified = verified
                claim_copy.evidence = evidence
                claim_copy.verification_source = "ground_truth"

            elif claim_copy.claim_type == ClaimType.FACT:
                verified, evidence = self._verify_fact_claim(claim_copy.text, facts)
                claim_copy.verified = verified
                claim_copy.evidence = evidence
                claim_copy.verification_source = "ground_truth"

            if claim_copy.verified is True:
                claim_copy.confidence = 1.0
            elif claim_copy.verified is False:
                claim_copy.confidence = 0.0

            verified_claims.append(claim_copy)

        return verified_claims

    def _verify_date_claim(
        self, claim_text: str, dates: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Verify a date claim against ground truth."""
        claim_lower = claim_text.lower()
        for key, value in dates.items():
            if key.lower() in claim_lower:
                expected_date = str(value)
                if expected_date in claim_text:
                    return True, f"Matches ground truth: {key}={value}"
                return False, f"Expected {key}={value}, found different date"
        return None, None

    def _verify_number_claim(
        self, claim_text: str, numbers: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Verify a number claim against ground truth."""
        claim_lower = claim_text.lower()
        for key, value in numbers.items():
            if key.lower() in claim_lower:
                expected_number = str(value)
                normalized_expected = expected_number.replace(",", "")
                claim_numbers = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", claim_text)
                normalized_claims = [n.replace(",", "") for n in claim_numbers]
                if normalized_expected in normalized_claims:
                    return True, f"Matches ground truth: {key}={value}"
                return False, f"Expected {key}={value}, found different number"
        return None, None

    def _verify_fact_claim(
        self, claim_text: str, facts: dict[str, Any]
    ) -> tuple[bool | None, str | None]:
        """Verify a fact claim against ground truth."""
        claim_lower = claim_text.lower()
        for key, value in facts.items():
            key_words = set(key.lower().split())
            claim_words = set(claim_lower.split())
            if len(key_words.intersection(claim_words)) >= len(key_words) * 0.5:
                value_str = str(value).lower()
                if value_str in claim_lower:
                    return True, f"Matches ground truth: {key}"
                elif any(word in claim_lower for word in value_str.split()):
                    return None, f"Partial match for: {key}"
                else:
                    return False, f"Contradicts ground truth: {key}={value}"
        return None, None

    @property
    def ground_truth(self) -> dict[str, Any]:
        """Get the current ground truth data."""
        return self._ground_truth


class LLMFactVerifier:
    """Verifies claims using LLM."""

    def __init__(
        self,
        config: FactualityConfig | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize with optional configuration.

        Args:
            config: Optional configuration for the LLM.
            cost_tracker: Optional cost tracker instance.
        """
        self._config = config or FactualityConfig()
        self._client: Any = None
        self._cost_tracker = cost_tracker
        self._input_tokens = 0
        self._output_tokens = 0

    def _get_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package is required for LLM verification. "
                    "Install it with: "
                    "uv add 'atp-platform[llm]'"
                ) from e

            if self._config.api_key:
                self._client = anthropic.AsyncAnthropic(api_key=self._config.api_key)
            else:
                self._client = anthropic.AsyncAnthropic()

        return self._client

    async def verify_claims(
        self,
        claims: list[Claim],
        context: str,
        ground_truth: dict[str, Any] | None = None,
    ) -> list[Claim]:
        """Verify claims using LLM.

        Args:
            claims: List of claims to verify.
            context: Original context/text.
            ground_truth: Optional ground truth for additional context.

        Returns:
            List of claims with verification status.
        """
        if not claims:
            return claims

        unverified_claims = [c for c in claims if c.verified is None]
        if not unverified_claims:
            return claims

        prompt = self._build_verification_prompt(
            unverified_claims, context, ground_truth
        )

        try:
            response = await self._call_llm(prompt)
            verified_claims = self._parse_verification_response(
                response, unverified_claims
            )
            verified_map = {c.text: c for c in verified_claims}
            result = []
            for claim in claims:
                if claim.text in verified_map:
                    result.append(verified_map[claim.text])
                else:
                    result.append(claim)
            return result
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return claims

    def _build_verification_prompt(
        self,
        claims: list[Claim],
        context: str,
        ground_truth: dict[str, Any] | None = None,
    ) -> str:
        """Build prompt for claim verification."""
        claims_text = "\n".join(
            f"{i + 1}. [{c.claim_type.value}] {c.text}" for i, c in enumerate(claims)
        )

        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"""
GROUND TRUTH (use as reference):
{json.dumps(ground_truth, indent=2)}
"""

        prompt_parts = [
            "You are a fact-checking assistant. Analyze the following claims "
            "extracted from a text and determine their factual accuracy.",
            "",
            "ORIGINAL CONTEXT:",
            context[:5000],
            "",
            "CLAIMS TO VERIFY:",
            claims_text,
            ground_truth_section,
            "For each claim, provide:",
            '1. verification_status: "verified" (true), "false", or "unverified"',
            "2. confidence: a score from 0.0 to 1.0",
            "3. evidence: brief explanation of your assessment",
            "",
            "Respond ONLY with a valid JSON array in this format:",
            "[",
            '  {"claim_index": 1, "verification_status": "verified|false|unverified",',
            '   "confidence": 0.0-1.0, "evidence": "explanation"},',
            "  ...",
            "]",
            "",
            "Important:",
            '- Only mark claims as "verified" or "false" if you have high confidence',
            '- Use "unverified" for claims you cannot confidently assess',
            "- Provide specific evidence for your assessments",
            "- Response must be valid JSON only",
        ]
        prompt = "\n".join(prompt_parts)

        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        client = self._get_client()

        response = await asyncio.wait_for(
            client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=self._config.timeout,
        )

        self._input_tokens += response.usage.input_tokens
        self._output_tokens += response.usage.output_tokens

        return response.content[0].text

    def _parse_verification_response(
        self, response: str, claims: list[Claim]
    ) -> list[Claim]:
        """Parse LLM verification response."""
        response = response.strip()

        if response.startswith("```"):
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            results = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                results = json.loads(json_match.group())
            else:
                return claims

        verified_claims = []
        for claim in claims:
            claim_copy = Claim(
                text=claim.text,
                claim_type=claim.claim_type,
                confidence=claim.confidence,
                location=claim.location,
                verified=claim.verified,
                evidence=claim.evidence,
                verification_source=claim.verification_source,
            )
            verified_claims.append(claim_copy)

        for result in results:
            idx = result.get("claim_index", 0) - 1
            if 0 <= idx < len(verified_claims):
                status = result.get("verification_status", "unverified")
                confidence = float(result.get("confidence", 0.5))
                evidence = result.get("evidence", "")

                verified_claims[idx].confidence = confidence
                verified_claims[idx].evidence = evidence
                verified_claims[idx].verification_source = "llm_verify"

                if status == "verified":
                    verified_claims[idx].verified = True
                elif status == "false":
                    verified_claims[idx].verified = False
                else:
                    verified_claims[idx].verified = None

        return verified_claims

    @property
    def input_tokens(self) -> int:
        """Get total input tokens used."""
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        """Get total output tokens used."""
        return self._output_tokens


class FactualityEvaluator(Evaluator):
    """Evaluator for verifying factual accuracy of agent outputs.

    This evaluator extracts factual claims from agent outputs and verifies them
    against ground truth data and/or using LLM-based verification.

    Features:
    - Claim extraction (dates, numbers, quotes, facts)
    - Ground truth verification from JSON file
    - LLM-based fact verification
    - Citation extraction and validation
    - Hallucination detection heuristics
    - Confidence scoring for claims

    Configuration options:
        ground_truth_file: Path to JSON file with ground truth facts
        verification_method: 'rag', 'llm_verify', or 'ground_truth'
        min_confidence: Minimum confidence threshold (default: 0.8)
        check_citations: Whether to validate citations (default: true)
        detect_hallucinations: Whether to run hallucination detection (default: true)
        path: Optional artifact path to evaluate

    Example usage:
        ```yaml
        assertions:
          - type: "factuality"
            config:
              ground_truth_file: "facts.json"
              verification_method: "llm_verify"
              min_confidence: 0.8
              check_citations: true
              detect_hallucinations: true
        ```
    """

    def __init__(
        self,
        config: FactualityConfig | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        """Initialize the Factuality evaluator.

        Args:
            config: Optional configuration for the evaluator.
            cost_tracker: Optional cost tracker instance.
        """
        self._config = config or FactualityConfig()
        self._claim_extractor = ClaimExtractor()
        self._citation_extractor = CitationExtractor()
        self._hallucination_detector = HallucinationDetector()
        self._ground_truth_verifier = GroundTruthVerifier()
        self._llm_verifier = LLMFactVerifier(self._config, cost_tracker)
        self._test_id: str | None = None
        self._suite_id: str | None = None

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "factuality"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate agent results for factual accuracy.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion configuration.

        Returns:
            EvalResult containing factuality check results.
        """
        self._test_id = task.id
        self._suite_id = getattr(task, "suite_id", None)

        config = assertion.config
        ground_truth_file = config.get("ground_truth_file")
        verification_method_str = config.get("verification_method", "ground_truth")
        min_confidence = config.get("min_confidence", 0.8)
        check_citations = config.get("check_citations", True)
        detect_hallucinations = config.get("detect_hallucinations", True)
        artifact_path = config.get("path")

        try:
            verification_method = VerificationMethod(verification_method_str.lower())
        except ValueError:
            verification_method = VerificationMethod.GROUND_TRUTH

        content = self._get_artifact_content(response, artifact_path)
        if content is None:
            return self._create_result(
                [
                    self._create_check(
                        name="factuality",
                        passed=False,
                        message="No artifact content found for evaluation",
                        details={"path": artifact_path},
                    )
                ]
            )

        ground_truth: dict[str, Any] = {}
        if ground_truth_file:
            try:
                self._ground_truth_verifier.load_ground_truth(ground_truth_file)
                ground_truth = self._ground_truth_verifier.ground_truth
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load ground truth file: {e}")

        claims = self._claim_extractor.extract_claims(content)

        if verification_method == VerificationMethod.GROUND_TRUTH:
            claims = self._ground_truth_verifier.verify_claims(claims)
        elif verification_method == VerificationMethod.LLM_VERIFY:
            claims = self._ground_truth_verifier.verify_claims(claims)
            claims = await self._llm_verifier.verify_claims(
                claims, content, ground_truth
            )
        elif verification_method == VerificationMethod.RAG:
            claims = self._ground_truth_verifier.verify_claims(claims)
            claims = await self._llm_verifier.verify_claims(
                claims, content, ground_truth
            )

        citations: list[Citation] = []
        if check_citations:
            citations = self._citation_extractor.extract_citations(content)
            citations = self._citation_extractor.validate_citations(
                citations, ground_truth
            )

        hallucination_indicators: list[HallucinationIndicator] = []
        if detect_hallucinations:
            hallucination_indicators = self._hallucination_detector.detect(
                content, claims
            )

        result = self._calculate_result(
            claims, citations, hallucination_indicators, min_confidence
        )

        return self._create_eval_result(result, min_confidence, config)

    def _get_artifact_content(
        self, response: ATPResponse, path: str | None = None
    ) -> str | None:
        """Extract artifact content for evaluation."""
        target_artifacts = response.artifacts
        if path:
            target_artifacts = [
                a
                for a in response.artifacts
                if (getattr(a, "path", None) or getattr(a, "name", None)) == path
            ]

        for artifact in target_artifacts:
            if hasattr(artifact, "content") and artifact.content:
                return str(artifact.content)
            if hasattr(artifact, "data") and artifact.data:
                return json.dumps(artifact.data, indent=2)

        return None

    def _calculate_result(
        self,
        claims: list[Claim],
        citations: list[Citation],
        hallucination_indicators: list[HallucinationIndicator],
        min_confidence: float,
    ) -> FactualityResult:
        """Calculate factuality analysis result."""
        verified_count = sum(1 for c in claims if c.verified is True)
        false_count = sum(1 for c in claims if c.verified is False)
        unverified_count = sum(1 for c in claims if c.verified is None)

        if claims:
            claim_score = verified_count / len(claims)
            false_penalty = false_count / len(claims) * 0.5
            claim_score = max(0.0, claim_score - false_penalty)
        else:
            claim_score = 1.0

        citation_score = 1.0
        if citations:
            valid_citations = sum(1 for c in citations if c.valid is True)
            invalid_citations = sum(1 for c in citations if c.valid is False)
            if valid_citations + invalid_citations > 0:
                citation_score = valid_citations / (valid_citations + invalid_citations)

        hallucination_penalty = 0.0
        for indicator in hallucination_indicators:
            if indicator.severity == "high":
                hallucination_penalty += 0.15
            elif indicator.severity == "medium":
                hallucination_penalty += 0.08
            else:
                hallucination_penalty += 0.03
        hallucination_penalty = min(hallucination_penalty, 0.5)

        overall_score = (
            claim_score * 0.6
            + citation_score * 0.2
            + (1.0 - hallucination_penalty) * 0.2
        )
        overall_score = max(0.0, min(1.0, overall_score))

        return FactualityResult(
            claims=claims,
            citations=citations,
            hallucination_indicators=hallucination_indicators,
            overall_score=overall_score,
            verified_claims_count=verified_count,
            unverified_claims_count=unverified_count,
            false_claims_count=false_count,
        )

    def _create_eval_result(
        self,
        result: FactualityResult,
        min_confidence: float,
        config: dict[str, Any],
    ) -> EvalResult:
        """Create EvalResult from FactualityResult."""
        checks: list[EvalCheck] = []

        passed = result.overall_score >= min_confidence
        if result.false_claims_count > 0:
            passed = False

        message_parts = []
        if result.claims:
            message_parts.append(
                f"{result.verified_claims_count}/{len(result.claims)} claims verified"
            )
        if result.false_claims_count > 0:
            message_parts.append(f"{result.false_claims_count} false claims detected")
        if result.hallucination_indicators:
            message_parts.append(
                f"{len(result.hallucination_indicators)} hallucination indicators"
            )

        message = "; ".join(message_parts) if message_parts else "No claims found"

        high_severity_hallucinations = [
            h for h in result.hallucination_indicators if h.severity == "high"
        ]

        checks.append(
            EvalCheck(
                name="factuality_check",
                passed=passed,
                score=result.overall_score,
                message=message,
                details={
                    "total_claims": len(result.claims),
                    "verified_claims": result.verified_claims_count,
                    "unverified_claims": result.unverified_claims_count,
                    "false_claims": result.false_claims_count,
                    "total_citations": len(result.citations),
                    "valid_citations": sum(
                        1 for c in result.citations if c.valid is True
                    ),
                    "hallucination_indicators": len(result.hallucination_indicators),
                    "high_severity_indicators": len(high_severity_hallucinations),
                    "min_confidence_threshold": min_confidence,
                    "claims": [c.to_dict() for c in result.claims],
                    "citations": [c.to_dict() for c in result.citations],
                    "hallucinations": [
                        h.to_dict() for h in result.hallucination_indicators
                    ],
                    "llm_tokens": {
                        "input": self._llm_verifier.input_tokens,
                        "output": self._llm_verifier.output_tokens,
                    },
                },
            )
        )

        return self._create_result(checks)
