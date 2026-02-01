"""Unit tests for StyleEvaluator."""

import pytest

from atp.evaluators.style import (
    StyleConfig,
    StyleEvaluator,
    StyleMetrics,
    TextAnalyzer,
    ToneType,
)
from atp.loader.models import Assertion, Constraints, TaskDefinition, TestDefinition
from atp.protocol import ArtifactFile, ArtifactStructured, ATPResponse, ResponseStatus


@pytest.fixture
def evaluator() -> StyleEvaluator:
    """Create StyleEvaluator instance."""
    return StyleEvaluator()


@pytest.fixture
def sample_task() -> TestDefinition:
    """Create a sample test definition."""
    return TestDefinition(
        id="test-001",
        name="Style Test",
        task=TaskDefinition(description="Test task"),
        constraints=Constraints(),
    )


@pytest.fixture
def professional_text() -> str:
    """Create professional-tone text."""
    return (
        "The analysis demonstrates that implementing this solution will "
        "facilitate optimal performance. Furthermore, the recommendation "
        "indicates comprehensive evaluation of the parameters. "
        "Accordingly, we should establish appropriate measures to ensure "
        "successful implementation of the proposed objective."
    )


@pytest.fixture
def casual_text() -> str:
    """Create casual-tone text."""
    return (
        "Hey, so basically this is pretty cool stuff! It's really easy "
        "to use and definitely works great. Yeah, just grab the thing "
        "and it'll totally do what you need. Super simple, right?"
    )


@pytest.fixture
def formal_text() -> str:
    """Create formal-tone text."""
    return (
        "Pursuant to the aforementioned regulations, the parties shall "
        "hereby agree to the terms set forth herein. Notwithstanding "
        "any provisions to the contrary, the obligations henceforth "
        "constitute binding agreement. Therefore, the undersigned shall "
        "be bound thereby."
    )


@pytest.fixture
def friendly_text() -> str:
    """Create friendly-tone text."""
    return (
        "I'm so glad you reached out! We absolutely love helping our "
        "customers and appreciate your support. I hope this solution "
        "works perfectly for you. Thanks so much for choosing us, and "
        "please let me know if there's anything else I can help with!"
    )


@pytest.fixture
def response_with_professional_text(professional_text: str) -> ATPResponse:
    """Create response with professional text content."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="output.txt", content=professional_text)],
    )


@pytest.fixture
def response_with_casual_text(casual_text: str) -> ATPResponse:
    """Create response with casual text content."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="output.txt", content=casual_text)],
    )


@pytest.fixture
def response_with_friendly_text(friendly_text: str) -> ATPResponse:
    """Create response with friendly text content."""
    return ATPResponse(
        task_id="test-001",
        status=ResponseStatus.COMPLETED,
        artifacts=[ArtifactFile(path="output.txt", content=friendly_text)],
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
def complex_text() -> str:
    """Create text with complex sentences for readability testing."""
    return (
        "The implementation of sophisticated algorithms necessitates "
        "comprehensive understanding of computational complexity theory. "
        "Furthermore, the optimization of performance metrics requires "
        "meticulous analysis of algorithmic efficiency. The concomitant "
        "improvement in processing speed demonstrates the efficacy of "
        "the proposed methodology. Subsequently, the experimental results "
        "corroborate our initial hypotheses regarding performance enhancement."
    )


@pytest.fixture
def simple_text() -> str:
    """Create text with simple sentences for readability testing."""
    return (
        "This is a short text. It has simple words. The sentences are easy. "
        "You can read it fast. It is not hard at all. Simple is good."
    )


@pytest.fixture
def passive_voice_text() -> str:
    """Create text with passive voice sentences."""
    return (
        "The report was written by the team. The analysis was conducted "
        "by the experts. The results were verified by the reviewers. "
        "The conclusions were drawn from the data. The recommendations "
        "are being implemented by the staff."
    )


class TestTextAnalyzer:
    """Tests for TextAnalyzer class."""

    def test_sentence_splitting(self) -> None:
        """Test sentence splitting handles various cases."""
        text = "Hello world. This is a test! Is it working? Yes it is."
        analyzer = TextAnalyzer(text)
        assert len(analyzer.sentences) == 4

    def test_sentence_splitting_with_abbreviations(self) -> None:
        """Test sentence splitting handles abbreviations."""
        text = "Dr. Smith went to the store. Mr. Jones followed."
        analyzer = TextAnalyzer(text)
        assert len(analyzer.sentences) == 2

    def test_word_extraction(self) -> None:
        """Test word extraction from text."""
        text = "Hello, world! This is a test."
        analyzer = TextAnalyzer(text)
        assert "hello" in analyzer.words
        assert "world" in analyzer.words
        assert len(analyzer.words) == 6

    def test_syllable_counting(self) -> None:
        """Test syllable counting in words."""
        analyzer = TextAnalyzer("")
        assert analyzer._syllables_in_word("hello") == 2
        assert analyzer._syllables_in_word("world") == 1
        assert analyzer._syllables_in_word("beautiful") == 3
        assert analyzer._syllables_in_word("the") == 1
        assert analyzer._syllables_in_word("a") == 1

    def test_flesch_kincaid_grade_simple_text(self, simple_text: str) -> None:
        """Test Flesch-Kincaid grade for simple text."""
        analyzer = TextAnalyzer(simple_text)
        grade = analyzer.calculate_flesch_kincaid_grade()
        # Simple text should have low grade level
        assert grade < 5.0

    def test_flesch_kincaid_grade_complex_text(self, complex_text: str) -> None:
        """Test Flesch-Kincaid grade for complex text."""
        analyzer = TextAnalyzer(complex_text)
        grade = analyzer.calculate_flesch_kincaid_grade()
        # Complex text should have higher grade level
        assert grade > 10.0

    def test_flesch_reading_ease_simple_text(self, simple_text: str) -> None:
        """Test Flesch Reading Ease for simple text."""
        analyzer = TextAnalyzer(simple_text)
        score = analyzer.calculate_flesch_reading_ease()
        # Simple text should have high reading ease (70+)
        assert score > 70.0

    def test_flesch_reading_ease_complex_text(self, complex_text: str) -> None:
        """Test Flesch Reading Ease for complex text."""
        analyzer = TextAnalyzer(complex_text)
        score = analyzer.calculate_flesch_reading_ease()
        # Complex text should have lower reading ease
        assert score < 50.0

    def test_smog_grade(self, complex_text: str) -> None:
        """Test SMOG grade calculation."""
        analyzer = TextAnalyzer(complex_text)
        grade = analyzer.calculate_smog_grade()
        assert grade > 0.0

    def test_coleman_liau_grade(self, complex_text: str) -> None:
        """Test Coleman-Liau grade calculation."""
        analyzer = TextAnalyzer(complex_text)
        grade = analyzer.calculate_coleman_liau_grade()
        assert grade > 0.0

    def test_passive_voice_detection(self, passive_voice_text: str) -> None:
        """Test passive voice detection."""
        analyzer = TextAnalyzer(passive_voice_text)
        percentage = analyzer.calculate_passive_voice_percentage()
        # Most sentences in passive_voice_text are passive
        assert percentage > 50.0

    def test_passive_voice_active_text(self, casual_text: str) -> None:
        """Test passive voice detection with active voice text."""
        analyzer = TextAnalyzer(casual_text)
        percentage = analyzer.calculate_passive_voice_percentage()
        # Casual text should have minimal passive voice
        assert percentage < 30.0

    def test_sentence_length_calculation(self) -> None:
        """Test sentence length metrics."""
        text = "This is short. This sentence is a bit longer than the first one."
        analyzer = TextAnalyzer(text)
        avg, max_len = analyzer.calculate_sentence_lengths()
        assert avg > 0.0
        assert max_len > 0

    def test_tone_scores_professional(self, professional_text: str) -> None:
        """Test tone scores for professional text."""
        analyzer = TextAnalyzer(professional_text)
        scores = analyzer.calculate_tone_scores()
        # Professional text should score high for professional tone
        assert scores[ToneType.PROFESSIONAL.value] >= 0.5

    def test_tone_scores_casual(self, casual_text: str) -> None:
        """Test tone scores for casual text."""
        analyzer = TextAnalyzer(casual_text)
        scores = analyzer.calculate_tone_scores()
        # Casual text should score high for casual tone
        assert scores[ToneType.CASUAL.value] >= 0.5

    def test_get_metrics_returns_all_fields(self, professional_text: str) -> None:
        """Test that get_metrics returns all required fields."""
        analyzer = TextAnalyzer(professional_text)
        metrics = analyzer.get_metrics()

        assert isinstance(metrics, StyleMetrics)
        assert metrics.flesch_kincaid_grade >= 0.0
        assert 0.0 <= metrics.flesch_reading_ease <= 100.0
        assert metrics.smog_grade >= 0.0
        assert metrics.coleman_liau_grade >= 0.0
        assert 0.0 <= metrics.passive_voice_percentage <= 100.0
        assert metrics.avg_sentence_length >= 0.0
        assert metrics.max_sentence_length >= 0
        assert metrics.total_sentences >= 0
        assert metrics.total_words >= 0
        assert len(metrics.tone_scores) == 4


class TestStyleConfig:
    """Tests for StyleConfig model."""

    def test_default_values(self) -> None:
        """Test StyleConfig default values."""
        config = StyleConfig()
        assert config.expected_tone is None
        assert config.tone_threshold == 0.7
        assert config.max_flesch_kincaid_grade is None
        assert config.forbidden_words == []
        assert config.required_words == []

    def test_custom_values(self) -> None:
        """Test StyleConfig with custom values."""
        config = StyleConfig(
            expected_tone=ToneType.PROFESSIONAL,
            tone_threshold=0.8,
            max_flesch_kincaid_grade=12.0,
            max_passive_voice_percentage=20.0,
            forbidden_words=["awesome", "cool"],
        )
        assert config.expected_tone == ToneType.PROFESSIONAL
        assert config.tone_threshold == 0.8
        assert config.max_flesch_kincaid_grade == 12.0
        assert config.max_passive_voice_percentage == 20.0
        assert config.forbidden_words == ["awesome", "cool"]


class TestStyleEvaluatorName:
    """Tests for evaluator name property."""

    def test_evaluator_name(self, evaluator: StyleEvaluator) -> None:
        """Test evaluator name is 'style'."""
        assert evaluator.name == "style"


class TestStyleEvaluatorTone:
    """Tests for tone evaluation."""

    @pytest.mark.anyio
    async def test_tone_professional_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        response_with_professional_text: ATPResponse,
    ) -> None:
        """Test professional tone detection passes."""
        assertion = Assertion(
            type="tone",
            config={
                "expected_tone": "professional",
                "tone_threshold": 0.5,
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_professional_text, [], assertion
        )
        assert result.passed is True
        assert result.checks[0].name == "tone"

    @pytest.mark.anyio
    async def test_tone_mismatch_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        response_with_casual_text: ATPResponse,
    ) -> None:
        """Test tone mismatch fails."""
        assertion = Assertion(
            type="tone",
            config={
                "expected_tone": "formal",
                "tone_threshold": 0.7,
            },
        )
        result = await evaluator.evaluate(
            sample_task, response_with_casual_text, [], assertion
        )
        # Casual text should not match formal tone well
        assert result.checks[0].name == "tone"
        assert "expected_tone" in result.checks[0].details
        assert result.checks[0].details["expected_tone"] == "formal"

    @pytest.mark.anyio
    async def test_tone_missing_config(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        response_with_professional_text: ATPResponse,
    ) -> None:
        """Test tone evaluation fails without expected_tone."""
        assertion = Assertion(type="tone", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_professional_text, [], assertion
        )
        assert result.passed is False
        assert "expected_tone" in result.checks[0].message


class TestStyleEvaluatorReadability:
    """Tests for readability evaluation."""

    @pytest.mark.anyio
    async def test_flesch_kincaid_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test Flesch-Kincaid grade check passes."""
        simple_text = "This is easy. Short words help. Read it fast."
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=simple_text)],
        )
        assertion = Assertion(
            type="readability",
            config={"max_flesch_kincaid_grade": 10.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        # Find the Flesch-Kincaid check
        fk_check = next(
            (c for c in result.checks if c.name == "flesch_kincaid_grade"), None
        )
        assert fk_check is not None
        assert fk_check.passed is True

    @pytest.mark.anyio
    async def test_flesch_kincaid_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        complex_text: str,
    ) -> None:
        """Test Flesch-Kincaid grade check fails for complex text."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=complex_text)],
        )
        assertion = Assertion(
            type="readability",
            config={"max_flesch_kincaid_grade": 5.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        fk_check = next(
            (c for c in result.checks if c.name == "flesch_kincaid_grade"), None
        )
        assert fk_check is not None
        assert fk_check.passed is False

    @pytest.mark.anyio
    async def test_flesch_reading_ease_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        simple_text: str,
    ) -> None:
        """Test Flesch Reading Ease check passes for simple text."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=simple_text)],
        )
        assertion = Assertion(
            type="readability",
            config={"min_flesch_reading_ease": 60.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        fre_check = next(
            (c for c in result.checks if c.name == "flesch_reading_ease"), None
        )
        assert fre_check is not None
        assert fre_check.passed is True

    @pytest.mark.anyio
    async def test_smog_grade_check(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        complex_text: str,
    ) -> None:
        """Test SMOG grade check."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=complex_text)],
        )
        assertion = Assertion(
            type="readability",
            config={"max_smog_grade": 20.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        smog_check = next((c for c in result.checks if c.name == "smog_grade"), None)
        assert smog_check is not None
        assert "actual" in smog_check.details

    @pytest.mark.anyio
    async def test_coleman_liau_check(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        complex_text: str,
    ) -> None:
        """Test Coleman-Liau grade check."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=complex_text)],
        )
        assertion = Assertion(
            type="readability",
            config={"max_coleman_liau_grade": 20.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        cl_check = next(
            (c for c in result.checks if c.name == "coleman_liau_grade"), None
        )
        assert cl_check is not None
        assert "actual" in cl_check.details


class TestStyleEvaluatorPassiveVoice:
    """Tests for passive voice evaluation."""

    @pytest.mark.anyio
    async def test_passive_voice_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        casual_text: str,
    ) -> None:
        """Test passive voice check passes for active text."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=casual_text)],
        )
        assertion = Assertion(
            type="passive_voice",
            config={"max_passive_voice_percentage": 30.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is True
        assert result.checks[0].name == "passive_voice"

    @pytest.mark.anyio
    async def test_passive_voice_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        passive_voice_text: str,
    ) -> None:
        """Test passive voice check fails for passive text."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=passive_voice_text)],
        )
        assertion = Assertion(
            type="passive_voice",
            config={"max_passive_voice_percentage": 20.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is False
        assert "exceeds" in result.checks[0].message

    @pytest.mark.anyio
    async def test_passive_voice_missing_config(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        casual_text: str,
    ) -> None:
        """Test passive voice fails without threshold config."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=casual_text)],
        )
        assertion = Assertion(type="passive_voice", config={})
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is False
        assert "must be specified" in result.checks[0].message


class TestStyleEvaluatorSentenceLength:
    """Tests for sentence length evaluation."""

    @pytest.mark.anyio
    async def test_avg_sentence_length_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        simple_text: str,
    ) -> None:
        """Test average sentence length check passes."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=simple_text)],
        )
        assertion = Assertion(
            type="sentence_length",
            config={"max_avg_sentence_length": 20.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        avg_check = next(
            (c for c in result.checks if c.name == "avg_sentence_length"), None
        )
        assert avg_check is not None
        assert avg_check.passed is True

    @pytest.mark.anyio
    async def test_avg_sentence_length_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        complex_text: str,
    ) -> None:
        """Test average sentence length check fails for long sentences."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=complex_text)],
        )
        assertion = Assertion(
            type="sentence_length",
            config={"max_avg_sentence_length": 5.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        avg_check = next(
            (c for c in result.checks if c.name == "avg_sentence_length"), None
        )
        assert avg_check is not None
        assert avg_check.passed is False

    @pytest.mark.anyio
    async def test_min_sentence_length_check(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        simple_text: str,
    ) -> None:
        """Test minimum average sentence length check."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=simple_text)],
        )
        assertion = Assertion(
            type="sentence_length",
            config={"min_avg_sentence_length": 2.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        min_check = next(
            (c for c in result.checks if c.name == "min_avg_sentence_length"), None
        )
        assert min_check is not None
        assert min_check.passed is True

    @pytest.mark.anyio
    async def test_max_single_sentence_length(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test maximum single sentence length check."""
        text = "Short. This is a very long sentence with many words."
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=text)],
        )
        assertion = Assertion(
            type="sentence_length",
            config={"max_sentence_length": 5},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        max_check = next(
            (c for c in result.checks if c.name == "max_sentence_length"), None
        )
        assert max_check is not None
        assert max_check.passed is False


class TestStyleEvaluatorRules:
    """Tests for custom style rules evaluation."""

    @pytest.mark.anyio
    async def test_forbidden_words_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        professional_text: str,
    ) -> None:
        """Test forbidden words check passes when words not present."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=professional_text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"forbidden_words": ["awesome", "cool", "lol"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        fw_check = next((c for c in result.checks if c.name == "forbidden_words"), None)
        assert fw_check is not None
        assert fw_check.passed is True

    @pytest.mark.anyio
    async def test_forbidden_words_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        casual_text: str,
    ) -> None:
        """Test forbidden words check fails when words present."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=casual_text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"forbidden_words": ["cool", "pretty"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        fw_check = next((c for c in result.checks if c.name == "forbidden_words"), None)
        assert fw_check is not None
        assert fw_check.passed is False
        assert "cool" in fw_check.details["found"]

    @pytest.mark.anyio
    async def test_required_words_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        professional_text: str,
    ) -> None:
        """Test required words check passes when words present."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=professional_text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"required_words": ["analysis", "implementation"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        rw_check = next((c for c in result.checks if c.name == "required_words"), None)
        assert rw_check is not None
        assert rw_check.passed is True

    @pytest.mark.anyio
    async def test_required_words_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        casual_text: str,
    ) -> None:
        """Test required words check fails when words missing."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=casual_text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"required_words": ["pursuant", "furthermore"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        rw_check = next((c for c in result.checks if c.name == "required_words"), None)
        assert rw_check is not None
        assert rw_check.passed is False
        assert "pursuant" in rw_check.details["missing"]

    @pytest.mark.anyio
    async def test_forbidden_patterns_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        professional_text: str,
    ) -> None:
        """Test forbidden patterns check passes when patterns not matched."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=professional_text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"forbidden_patterns": [r"lol+", r"!!+"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        fp_check = next(
            (c for c in result.checks if c.name == "forbidden_patterns"), None
        )
        assert fp_check is not None
        assert fp_check.passed is True

    @pytest.mark.anyio
    async def test_forbidden_patterns_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test forbidden patterns check fails when patterns matched."""
        text = "This is exciting!! And lol so funny!!!"
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"forbidden_patterns": [r"!!+", r"lol"]},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        fp_check = next(
            (c for c in result.checks if c.name == "forbidden_patterns"), None
        )
        assert fp_check is not None
        assert fp_check.passed is False

    @pytest.mark.anyio
    async def test_required_patterns_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test required patterns check passes when patterns matched."""
        text = "Please contact us at support@example.com for assistance."
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"required_patterns": [r"\S+@\S+\.\S+"]},  # Email pattern
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        rp_check = next(
            (c for c in result.checks if c.name == "required_patterns"), None
        )
        assert rp_check is not None
        assert rp_check.passed is True

    @pytest.mark.anyio
    async def test_required_patterns_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        professional_text: str,
    ) -> None:
        """Test required patterns check fails when patterns not matched."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=professional_text)],
        )
        assertion = Assertion(
            type="style_rules",
            config={"required_patterns": [r"\S+@\S+\.\S+"]},  # Email pattern
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        rp_check = next(
            (c for c in result.checks if c.name == "required_patterns"), None
        )
        assert rp_check is not None
        assert rp_check.passed is False


class TestStyleEvaluatorComprehensive:
    """Tests for comprehensive style evaluation."""

    @pytest.mark.anyio
    async def test_style_all_checks_pass(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        professional_text: str,
    ) -> None:
        """Test comprehensive style evaluation with all checks passing."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=professional_text)],
        )
        assertion = Assertion(
            type="style",
            config={
                "expected_tone": "professional",
                "tone_threshold": 0.5,
                "max_flesch_kincaid_grade": 20.0,
                "max_passive_voice_percentage": 50.0,
                "max_avg_sentence_length": 30.0,
            },
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        # Should have metrics check plus configured checks
        assert len(result.checks) >= 4
        assert result.checks[0].name == "style_metrics"

    @pytest.mark.anyio
    async def test_style_mixed_pass_fail(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        casual_text: str,
    ) -> None:
        """Test style evaluation with mixed pass/fail checks."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content=casual_text)],
        )
        assertion = Assertion(
            type="style",
            config={
                "expected_tone": "professional",  # Will fail
                "tone_threshold": 0.8,
                "max_flesch_kincaid_grade": 20.0,  # Will pass
            },
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        tone_check = next((c for c in result.checks if c.name == "tone"), None)
        fk_check = next(
            (c for c in result.checks if c.name == "flesch_kincaid_grade"), None
        )
        assert tone_check is not None
        assert fk_check is not None

    @pytest.mark.anyio
    async def test_style_no_text(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        empty_response: ATPResponse,
    ) -> None:
        """Test style evaluation with no text content."""
        assertion = Assertion(type="style", config={})
        result = await evaluator.evaluate(sample_task, empty_response, [], assertion)
        assert result.passed is False
        assert result.checks[0].name == "no_text"


class TestStyleEvaluatorUnknownType:
    """Tests for unknown assertion types."""

    @pytest.mark.anyio
    async def test_unknown_assertion_type(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
        response_with_professional_text: ATPResponse,
    ) -> None:
        """Test unknown assertion type returns failure."""
        assertion = Assertion(type="unknown_style_type", config={})
        result = await evaluator.evaluate(
            sample_task, response_with_professional_text, [], assertion
        )
        assert result.passed is False
        assert "unknown" in result.checks[0].message.lower()


class TestStyleEvaluatorEdgeCases:
    """Edge case tests for StyleEvaluator."""

    @pytest.mark.anyio
    async def test_empty_text(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test handling of empty text content."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content="")],
        )
        assertion = Assertion(type="style", config={})
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.passed is False

    @pytest.mark.anyio
    async def test_single_word(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test handling of single word text."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[ArtifactFile(path="output.txt", content="Hello")],
        )
        assertion = Assertion(
            type="readability",
            config={"max_flesch_kincaid_grade": 20.0},
        )
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        # Should not crash with minimal text
        assert len(result.checks) >= 1

    @pytest.mark.anyio
    async def test_artifact_with_data_dict(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test extracting text from artifact with data dict."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactStructured(
                    name="output",
                    data={"text": "This is the text content from data field."},
                )
            ],
        )
        assertion = Assertion(type="style", config={})
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        # Should extract text from data.text field
        assert result.checks[0].name == "style_metrics"
        assert result.checks[0].passed is True

    @pytest.mark.anyio
    async def test_multiple_artifacts(
        self,
        evaluator: StyleEvaluator,
        sample_task: TestDefinition,
    ) -> None:
        """Test combining text from multiple artifacts."""
        response = ATPResponse(
            task_id="test-001",
            status=ResponseStatus.COMPLETED,
            artifacts=[
                ArtifactFile(path="part1.txt", content="First part of the text."),
                ArtifactFile(path="part2.txt", content="Second part of the text."),
            ],
        )
        assertion = Assertion(type="style", config={})
        result = await evaluator.evaluate(sample_task, response, [], assertion)
        assert result.checks[0].name == "style_metrics"
        # Combined text should have more words
        assert result.checks[0].details["total_words"] >= 8
