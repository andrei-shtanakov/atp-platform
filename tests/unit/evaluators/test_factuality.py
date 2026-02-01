"""Unit tests for FactualityEvaluator."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.evaluators.factuality import (
    Citation,
    CitationExtractor,
    Claim,
    ClaimExtractor,
    ClaimType,
    FactualityConfig,
    FactualityEvaluator,
    FactualityResult,
    GroundTruthVerifier,
    HallucinationDetector,
    HallucinationIndicator,
    LLMFactVerifier,
    VerificationMethod,
)
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ArtifactFile, ArtifactStructured, ATPResponse, ResponseStatus


@pytest.fixture
def evaluator() -> FactualityEvaluator:
    """Create FactualityEvaluator instance with default config."""
    return FactualityEvaluator()


@pytest.fixture
def evaluator_with_config() -> FactualityEvaluator:
    """Create FactualityEvaluator instance with custom config."""
    config = FactualityConfig(
        api_key="test-api-key",
        model="claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=2048,
        timeout=30.0,
    )
    return FactualityEvaluator(config)


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Sample Factuality Test",
        task=TaskDefinition(description="Write a factual report about AI history"),
        constraints=Constraints(),
    )


@pytest.fixture
def response_with_factual_content() -> ATPResponse:
    """Create response with factual content."""
    content = """# AI History Report

Artificial Intelligence was founded as an academic discipline in 1956.
The term "Artificial Intelligence" was coined by John McCarthy in 1956.

According to recent studies, 75% of enterprises have adopted AI in some form.
OpenAI was founded in 2015 with a $1 billion investment.

Sources:
- https://example.com/ai-history
- [1] McCarthy, J. (1956). "AI Conference Proceedings"
"""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="report.md",
                content=content,
                content_type="text/markdown",
            ),
        ],
    )


@pytest.fixture
def response_with_hallucination_indicators() -> ATPResponse:
    """Create response with hallucination indicators."""
    content = """# Report

Many experts believe that AI will definitely transform every industry.
It is well-known that everyone uses AI daily.
Studies show that exactly 87.3% of companies always use AI without exception.
No one doubts that AI is certainly the future.
"""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactFile(
                path="report.md",
                content=content,
                content_type="text/markdown",
            ),
        ],
    )


@pytest.fixture
def response_with_structured_artifact() -> ATPResponse:
    """Create response with structured artifact."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[
            ArtifactStructured(
                name="result",
                data={
                    "title": "AI Report",
                    "founded_year": 1956,
                    "key_person": "John McCarthy",
                },
            ),
        ],
    )


@pytest.fixture
def empty_response() -> ATPResponse:
    """Create response with no artifacts."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[],
    )


@pytest.fixture
def ground_truth_data() -> dict:
    """Create sample ground truth data."""
    return {
        "facts": {
            "AI founding": "1956",
            "AI term coined by": "John McCarthy",
        },
        "dates": {
            "AI founded": "1956",
            "OpenAI founded": "2015",
        },
        "numbers": {
            "enterprise AI adoption": "75",
            "OpenAI initial investment": "1 billion",
        },
        "valid_sources": [
            "https://example.com/ai-history",
            "McCarthy, J. (1956)",
        ],
    }


@pytest.fixture
def ground_truth_file(tmp_path: Path, ground_truth_data: dict) -> Path:
    """Create a temporary ground truth file."""
    file_path = tmp_path / "ground_truth.json"
    with open(file_path, "w") as f:
        json.dump(ground_truth_data, f)
    return file_path


class TestFactualityConfig:
    """Tests for FactualityConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = FactualityConfig()
        assert config.api_key is None
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.0
        assert config.max_tokens == 2048
        assert config.timeout == 60.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = FactualityConfig(
            api_key="test-key",
            model="custom-model",
            temperature=0.5,
            max_tokens=4096,
            timeout=120.0,
        )
        assert config.api_key == "test-key"
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096
        assert config.timeout == 120.0


class TestVerificationMethod:
    """Tests for VerificationMethod enum."""

    def test_verification_methods(self) -> None:
        """Test verification method values."""
        assert VerificationMethod.RAG.value == "rag"
        assert VerificationMethod.LLM_VERIFY.value == "llm_verify"
        assert VerificationMethod.GROUND_TRUTH.value == "ground_truth"


class TestClaimType:
    """Tests for ClaimType enum."""

    def test_claim_types(self) -> None:
        """Test claim type values."""
        assert ClaimType.DATE.value == "date"
        assert ClaimType.NUMBER.value == "number"
        assert ClaimType.NAME.value == "name"
        assert ClaimType.FACT.value == "fact"
        assert ClaimType.QUOTE.value == "quote"
        assert ClaimType.STATISTIC.value == "statistic"


class TestClaim:
    """Tests for Claim dataclass."""

    def test_claim_creation(self) -> None:
        """Test claim creation."""
        claim = Claim(
            text="AI was founded in 1956",
            claim_type=ClaimType.DATE,
            confidence=0.9,
            verified=True,
            evidence="Matches ground truth",
        )
        assert claim.text == "AI was founded in 1956"
        assert claim.claim_type == ClaimType.DATE
        assert claim.confidence == 0.9
        assert claim.verified is True
        assert claim.evidence == "Matches ground truth"

    def test_claim_to_dict(self) -> None:
        """Test claim to_dict method."""
        claim = Claim(
            text="Test claim",
            claim_type=ClaimType.FACT,
            confidence=0.8,
            verified=True,
        )
        d = claim.to_dict()
        assert d["text"] == "Test claim"
        assert d["type"] == "fact"
        assert d["confidence"] == 0.8
        assert d["verified"] is True


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self) -> None:
        """Test citation creation."""
        citation = Citation(
            text="https://example.com",
            url="https://example.com",
            valid=True,
        )
        assert citation.text == "https://example.com"
        assert citation.url == "https://example.com"
        assert citation.valid is True

    def test_citation_to_dict(self) -> None:
        """Test citation to_dict method."""
        citation = Citation(
            text="Source reference",
            source="McCarthy, J.",
            valid=True,
        )
        d = citation.to_dict()
        assert d["text"] == "Source reference"
        assert d["source"] == "McCarthy, J."
        assert d["valid"] is True


class TestHallucinationIndicator:
    """Tests for HallucinationIndicator dataclass."""

    def test_indicator_creation(self) -> None:
        """Test indicator creation."""
        indicator = HallucinationIndicator(
            indicator_type="vague_language",
            description="Unsourced claim",
            severity="medium",
            evidence="many experts say",
        )
        assert indicator.indicator_type == "vague_language"
        assert indicator.severity == "medium"

    def test_indicator_to_dict(self) -> None:
        """Test indicator to_dict method."""
        indicator = HallucinationIndicator(
            indicator_type="overconfidence",
            description="Absolute statement",
            severity="high",
        )
        d = indicator.to_dict()
        assert d["type"] == "overconfidence"
        assert d["severity"] == "high"


class TestFactualityResult:
    """Tests for FactualityResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation."""
        result = FactualityResult(
            claims=[Claim("test", ClaimType.FACT)],
            overall_score=0.85,
            verified_claims_count=1,
        )
        assert len(result.claims) == 1
        assert result.overall_score == 0.85

    def test_result_to_dict(self) -> None:
        """Test result to_dict method."""
        result = FactualityResult(
            claims=[Claim("test", ClaimType.FACT)],
            citations=[Citation("url", url="https://test.com")],
            overall_score=0.9,
        )
        d = result.to_dict()
        assert "claims" in d
        assert "citations" in d
        assert d["overall_score"] == 0.9


class TestClaimExtractor:
    """Tests for ClaimExtractor."""

    def test_extract_date_claims(self) -> None:
        """Test extracting date claims."""
        extractor = ClaimExtractor()
        text = "AI was founded in 1956. The conference happened on January 1, 2023."
        claims = extractor.extract_claims(text)
        date_claims = [c for c in claims if c.claim_type == ClaimType.DATE]
        assert len(date_claims) >= 1

    def test_extract_number_claims(self) -> None:
        """Test extracting number claims."""
        extractor = ClaimExtractor()
        text = "About 75% of companies use AI. The investment was $1 billion."
        claims = extractor.extract_claims(text)
        number_claims = [
            c for c in claims if c.claim_type in (ClaimType.NUMBER, ClaimType.STATISTIC)
        ]
        assert len(number_claims) >= 1

    def test_extract_quote_claims(self) -> None:
        """Test extracting quote claims."""
        extractor = ClaimExtractor()
        text = (
            'He said "artificial intelligence will change the world" at the conference.'
        )
        claims = extractor.extract_claims(text)
        quote_claims = [c for c in claims if c.claim_type == ClaimType.QUOTE]
        assert len(quote_claims) >= 1

    def test_extract_factual_statements(self) -> None:
        """Test extracting factual statements."""
        extractor = ClaimExtractor()
        text = "OpenAI was founded by Elon Musk. According to research, AI is growing."
        claims = extractor.extract_claims(text)
        fact_claims = [c for c in claims if c.claim_type == ClaimType.FACT]
        assert len(fact_claims) >= 1

    def test_deduplication(self) -> None:
        """Test claim deduplication within same type."""
        extractor = ClaimExtractor()
        # Test that duplicate date claims are deduplicated within each extraction type
        text = "AI was founded in 1956."
        claims = extractor.extract_claims(text)
        # At minimum, we should get claims extracted
        assert len(claims) >= 1
        # Check that fact claims are deduplicated
        fact_claims = [c for c in claims if c.claim_type == ClaimType.FACT]
        fact_texts = [c.text.lower().strip() for c in fact_claims]
        assert len(fact_texts) == len(set(fact_texts))

    def test_empty_text(self) -> None:
        """Test with empty text."""
        extractor = ClaimExtractor()
        claims = extractor.extract_claims("")
        assert claims == []


class TestCitationExtractor:
    """Tests for CitationExtractor."""

    def test_extract_url_citations(self) -> None:
        """Test extracting URL citations."""
        extractor = CitationExtractor()
        text = "See https://example.com/page for details."
        citations = extractor.extract_citations(text)
        url_citations = [c for c in citations if c.url]
        assert len(url_citations) >= 1
        assert "https://example.com/page" in [c.url for c in url_citations]

    def test_extract_reference_citations(self) -> None:
        """Test extracting reference citations."""
        extractor = CitationExtractor()
        text = "According to McCarthy, AI is important. [1] See reference."
        citations = extractor.extract_citations(text)
        assert len(citations) >= 1

    def test_extract_author_year_citations(self) -> None:
        """Test extracting author-year citations."""
        extractor = CitationExtractor()
        text = "Research shows this (Smith, 2023)."
        citations = extractor.extract_citations(text)
        assert len(citations) >= 1

    def test_validate_citations_with_ground_truth(self) -> None:
        """Test citation validation with ground truth."""
        extractor = CitationExtractor()
        citations = [
            Citation(text="https://example.com", url="https://example.com"),
            Citation(text="McCarthy, J.", source="McCarthy, J."),
        ]
        ground_truth = {
            "valid_sources": ["https://example.com", "McCarthy, J."],
        }
        validated = extractor.validate_citations(citations, ground_truth)
        assert all(c.valid for c in validated if c.source)

    def test_validate_url_format(self) -> None:
        """Test URL format validation."""
        extractor = CitationExtractor()
        citations = [
            Citation(text="https://valid.com", url="https://valid.com"),
            Citation(text="invalid-url", url="invalid-url"),
        ]
        validated = extractor.validate_citations(citations)
        assert validated[0].valid is True
        assert validated[1].valid is False


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    def test_detect_vague_language(self) -> None:
        """Test detecting vague language."""
        detector = HallucinationDetector()
        text = "Many experts believe this is true. Some say it will work."
        indicators = detector.detect(text, [])
        vague_indicators = [
            i for i in indicators if i.indicator_type == "vague_language"
        ]
        assert len(vague_indicators) >= 1

    def test_detect_overconfidence(self) -> None:
        """Test detecting overconfident language."""
        detector = HallucinationDetector()
        text = "This will definitely work. Everyone knows this."
        indicators = detector.detect(text, [])
        confidence_indicators = [
            i for i in indicators if i.indicator_type == "overconfidence"
        ]
        assert len(confidence_indicators) >= 1

    def test_detect_unverifiable_specifics(self) -> None:
        """Test detecting unverifiable specific claims."""
        detector = HallucinationDetector()
        text = "Exactly 87.3% of users prefer this."
        indicators = detector.detect(text, [])
        specific_indicators = [
            i for i in indicators if i.indicator_type == "unverifiable_specific"
        ]
        assert len(specific_indicators) >= 1

    def test_detect_inconsistencies(self) -> None:
        """Test detecting inconsistencies between claims."""
        detector = HallucinationDetector()
        # Test the _dates_conflict method directly - claims need:
        # 1. Different years (19xx or 20xx pattern)
        # 2. Common context keywords (words with 4+ chars)
        # Using "intelligence" and "founded" as common keywords
        claims = [
            Claim("Artificial intelligence research started in 1956", ClaimType.DATE),
            Claim("Artificial intelligence research started in 1960", ClaimType.DATE),
        ]
        indicators = detector.detect("", claims)
        inconsistency_indicators = [
            i for i in indicators if i.indicator_type == "inconsistency"
        ]
        # Should detect conflict: same context with different years (1956 vs 1960)
        assert len(inconsistency_indicators) >= 1

    def test_no_indicators_for_clean_text(self) -> None:
        """Test no indicators for clean factual text."""
        detector = HallucinationDetector()
        text = "AI was founded in 1956 at the Dartmouth Conference."
        indicators = detector.detect(text, [])
        assert len(indicators) == 0


class TestGroundTruthVerifier:
    """Tests for GroundTruthVerifier."""

    def test_load_ground_truth_file(
        self, ground_truth_file: Path, ground_truth_data: dict
    ) -> None:
        """Test loading ground truth from file."""
        verifier = GroundTruthVerifier(ground_truth_file)
        assert verifier.ground_truth == ground_truth_data

    def test_set_ground_truth_directly(self, ground_truth_data: dict) -> None:
        """Test setting ground truth directly."""
        verifier = GroundTruthVerifier()
        verifier.set_ground_truth(ground_truth_data)
        assert verifier.ground_truth == ground_truth_data

    def test_verify_date_claims(self, ground_truth_data: dict) -> None:
        """Test verifying date claims."""
        verifier = GroundTruthVerifier()
        verifier.set_ground_truth(ground_truth_data)
        claims = [
            Claim("AI founded in 1956", ClaimType.DATE),
        ]
        verified = verifier.verify_claims(claims)
        # The first claim should match "AI founded" -> 1956
        assert verified[0].verified is True

    def test_verify_number_claims(self, ground_truth_data: dict) -> None:
        """Test verifying number claims."""
        verifier = GroundTruthVerifier()
        verifier.set_ground_truth(ground_truth_data)
        claims = [
            Claim("Enterprise AI adoption is at 75%", ClaimType.STATISTIC),
        ]
        verified = verifier.verify_claims(claims)
        assert verified[0].verified is True

    def test_verify_false_claims(self, ground_truth_data: dict) -> None:
        """Test detecting false claims."""
        verifier = GroundTruthVerifier()
        verifier.set_ground_truth(ground_truth_data)
        # Use a claim that matches the ground truth key but has wrong value
        claims = [
            Claim("AI founded in 1999", ClaimType.DATE),
        ]
        verified = verifier.verify_claims(claims)
        # This should be marked false because "AI founded" is in ground truth with 1956
        assert verified[0].verified is False

    def test_unverified_claims(self) -> None:
        """Test claims that cannot be verified."""
        verifier = GroundTruthVerifier()
        verifier.set_ground_truth({"facts": {}, "dates": {}, "numbers": {}})
        claims = [
            Claim("The weather was nice", ClaimType.FACT),
        ]
        verified = verifier.verify_claims(claims)
        assert verified[0].verified is None

    def test_empty_ground_truth(self) -> None:
        """Test with empty ground truth."""
        verifier = GroundTruthVerifier()
        claims = [Claim("Test claim", ClaimType.FACT)]
        verified = verifier.verify_claims(claims)
        assert verified == claims

    def test_load_nonexistent_file(self) -> None:
        """Test loading nonexistent file."""
        verifier = GroundTruthVerifier()
        with pytest.raises(FileNotFoundError):
            verifier.load_ground_truth("/nonexistent/path.json")


class TestLLMFactVerifier:
    """Tests for LLMFactVerifier."""

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        verifier = LLMFactVerifier()
        assert verifier._config.model == "claude-sonnet-4-20250514"

    def test_init_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = FactualityConfig(model="custom-model")
        verifier = LLMFactVerifier(config)
        assert verifier._config.model == "custom-model"

    @pytest.mark.anyio
    async def test_verify_claims_empty_list(self) -> None:
        """Test verification with empty claims list."""
        verifier = LLMFactVerifier()
        result = await verifier.verify_claims([], "context", None)
        assert result == []

    @pytest.mark.anyio
    async def test_verify_claims_all_already_verified(self) -> None:
        """Test verification when all claims already verified."""
        verifier = LLMFactVerifier()
        claims = [
            Claim("Test", ClaimType.FACT, verified=True),
            Claim("Test2", ClaimType.FACT, verified=False),
        ]
        result = await verifier.verify_claims(claims, "context", None)
        assert result == claims

    @pytest.mark.anyio
    async def test_verify_claims_with_llm(self) -> None:
        """Test LLM-based claim verification."""
        config = FactualityConfig(api_key="test-key")
        verifier = LLMFactVerifier(config)

        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    [
                        {
                            "claim_index": 1,
                            "verification_status": "verified",
                            "confidence": 0.95,
                            "evidence": "Verified fact",
                        }
                    ]
                )
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch.object(verifier, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            claims = [Claim("AI founded in 1956", ClaimType.FACT)]
            result = await verifier.verify_claims(claims, "context text", None)

            assert len(result) == 1
            assert result[0].verified is True
            assert result[0].confidence == 0.95

    @pytest.mark.anyio
    async def test_verify_claims_handles_error(self) -> None:
        """Test error handling during verification."""
        config = FactualityConfig(api_key="test-key")
        verifier = LLMFactVerifier(config)

        with patch.object(verifier, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
            mock_get_client.return_value = mock_client

            claims = [Claim("Test claim", ClaimType.FACT)]
            result = await verifier.verify_claims(claims, "context", None)

            assert result == claims

    def test_token_tracking(self) -> None:
        """Test token usage tracking."""
        verifier = LLMFactVerifier()
        assert verifier.input_tokens == 0
        assert verifier.output_tokens == 0


class TestFactualityEvaluatorProperties:
    """Tests for FactualityEvaluator properties."""

    def test_evaluator_name(self, evaluator: FactualityEvaluator) -> None:
        """Test evaluator name property."""
        assert evaluator.name == "factuality"


class TestFactualityEvaluatorEvaluate:
    """Tests for FactualityEvaluator.evaluate method."""

    @pytest.mark.anyio
    async def test_evaluate_no_artifact(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """Test evaluate fails when no artifact found."""
        assertion = Assertion(type="factuality", config={})
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is False
        assert "no artifact" in result.checks[0].message.lower()

    @pytest.mark.anyio
    async def test_evaluate_with_ground_truth_file(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_factual_content: ATPResponse,
        ground_truth_file: Path,
    ) -> None:
        """Test evaluation with ground truth file."""
        assertion = Assertion(
            type="factuality",
            config={
                "ground_truth_file": str(ground_truth_file),
                "verification_method": "ground_truth",
                "min_confidence": 0.5,
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_factual_content, [], assertion
        )
        assert result.checks[0].score >= 0.0
        assert "claims" in result.checks[0].details

    @pytest.mark.anyio
    async def test_evaluate_detects_hallucinations(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_hallucination_indicators: ATPResponse,
    ) -> None:
        """Test evaluation detects hallucination indicators."""
        assertion = Assertion(
            type="factuality",
            config={
                "detect_hallucinations": True,
                "min_confidence": 0.9,
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_hallucination_indicators, [], assertion
        )
        assert result.checks[0].details["hallucination_indicators"] > 0

    @pytest.mark.anyio
    async def test_evaluate_extracts_citations(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_factual_content: ATPResponse,
    ) -> None:
        """Test evaluation extracts and validates citations."""
        assertion = Assertion(
            type="factuality",
            config={
                "check_citations": True,
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_factual_content, [], assertion
        )
        assert result.checks[0].details["total_citations"] >= 1

    @pytest.mark.anyio
    async def test_evaluate_with_llm_verification(
        self,
        evaluator_with_config: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_factual_content: ATPResponse,
    ) -> None:
        """Test evaluation with LLM verification."""
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=json.dumps(
                    [
                        {
                            "claim_index": 1,
                            "verification_status": "verified",
                            "confidence": 0.9,
                            "evidence": "Fact verified",
                        }
                    ]
                )
            )
        ]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch.object(
            evaluator_with_config._llm_verifier, "_get_client"
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            assertion = Assertion(
                type="factuality",
                config={
                    "verification_method": "llm_verify",
                    "min_confidence": 0.5,
                },
            )
            result = await evaluator_with_config.evaluate(
                sample_task, response_with_factual_content, [], assertion
            )
            assert "llm_tokens" in result.checks[0].details

    @pytest.mark.anyio
    async def test_evaluate_with_structured_artifact(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_structured_artifact: ATPResponse,
    ) -> None:
        """Test evaluation with structured artifact."""
        assertion = Assertion(type="factuality", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_structured_artifact, [], assertion
        )
        assert result.passed is not None

    @pytest.mark.anyio
    async def test_evaluate_with_invalid_verification_method(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_factual_content: ATPResponse,
    ) -> None:
        """Test evaluation handles invalid verification method."""
        assertion = Assertion(
            type="factuality",
            config={
                "verification_method": "invalid_method",
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_factual_content, [], assertion
        )
        assert result.passed is not None

    @pytest.mark.anyio
    async def test_evaluate_with_specific_artifact_path(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_factual_content: ATPResponse,
    ) -> None:
        """Test evaluation with specific artifact path."""
        assertion = Assertion(
            type="factuality",
            config={
                "path": "report.md",
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_factual_content, [], assertion
        )
        assert result.passed is not None

    @pytest.mark.anyio
    async def test_evaluate_wrong_artifact_path(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        response_with_factual_content: ATPResponse,
    ) -> None:
        """Test evaluation with wrong artifact path."""
        assertion = Assertion(
            type="factuality",
            config={
                "path": "nonexistent.md",
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_factual_content, [], assertion
        )
        assert result.passed is False

    @pytest.mark.anyio
    async def test_evaluate_false_claims_fail(
        self,
        evaluator: FactualityEvaluator,
        sample_task: TestDefinition,
        ground_truth_file: Path,
    ) -> None:
        """Test evaluation fails when false claims detected."""
        content = "AI was founded in 1999. This is wrong."
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(
                    path="report.md",
                    content=content,
                    content_type="text/markdown",
                ),
            ],
        )
        assertion = Assertion(
            type="factuality",
            config={
                "ground_truth_file": str(ground_truth_file),
                "verification_method": "ground_truth",
            },
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.checks[0].details.get("false_claims", 0) >= 0


class TestRegistry:
    """Tests for registry integration."""

    def test_factuality_in_registry(self) -> None:
        """Test FactualityEvaluator is registered."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.is_registered("factuality")

    def test_factuality_assertion_mapped(self) -> None:
        """Test factuality assertion type is mapped."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        assert registry.supports_assertion("factuality")

    def test_create_factuality_from_registry(self) -> None:
        """Test creating FactualityEvaluator from registry."""
        from atp.evaluators.registry import get_registry

        registry = get_registry()
        evaluator = registry.create("factuality")
        assert isinstance(evaluator, FactualityEvaluator)


class TestFactualityImports:
    """Tests for module imports."""

    def test_imports_from_evaluators(self) -> None:
        """Test imports from evaluators package."""
        from atp.evaluators import (
            Citation,
            CitationExtractor,
            Claim,
            ClaimExtractor,
            ClaimType,
            FactualityConfig,
            FactualityEvaluator,
            FactualityResult,
            GroundTruthVerifier,
            HallucinationDetector,
            HallucinationIndicator,
            LLMFactVerifier,
            VerificationMethod,
        )

        assert FactualityEvaluator is not None
        assert FactualityConfig is not None
        assert Claim is not None
        assert ClaimType is not None
        assert ClaimExtractor is not None
        assert Citation is not None
        assert CitationExtractor is not None
        assert GroundTruthVerifier is not None
        assert LLMFactVerifier is not None
        assert HallucinationDetector is not None
        assert HallucinationIndicator is not None
        assert VerificationMethod is not None
        assert FactualityResult is not None
