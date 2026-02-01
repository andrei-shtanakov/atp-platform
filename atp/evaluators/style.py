"""Style and Tone evaluator for assessing writing style and readability."""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from atp.evaluators.base import EvalCheck, EvalResult, Evaluator
from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse


class ToneType(str, Enum):
    """Supported tone types for analysis."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FORMAL = "formal"
    FRIENDLY = "friendly"


class StyleConfig(BaseModel):
    """Configuration for style evaluation."""

    # Tone settings
    expected_tone: ToneType | None = Field(
        None, description="Expected tone of the output"
    )
    tone_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum tone match score to pass"
    )

    # Readability settings
    max_flesch_kincaid_grade: float | None = Field(
        None, description="Maximum Flesch-Kincaid grade level"
    )
    min_flesch_reading_ease: float | None = Field(
        None, description="Minimum Flesch Reading Ease score (0-100)"
    )
    max_smog_grade: float | None = Field(None, description="Maximum SMOG grade level")
    max_coleman_liau_grade: float | None = Field(
        None, description="Maximum Coleman-Liau grade level"
    )

    # Sentence structure settings
    max_passive_voice_percentage: float | None = Field(
        None, ge=0.0, le=100.0, description="Maximum allowed passive voice percentage"
    )
    max_avg_sentence_length: float | None = Field(
        None, description="Maximum average words per sentence"
    )
    min_avg_sentence_length: float | None = Field(
        None, description="Minimum average words per sentence"
    )
    max_sentence_length: int | None = Field(
        None, description="Maximum words in any single sentence"
    )

    # Custom style rules
    forbidden_words: list[str] = Field(
        default_factory=list, description="Words that should not appear in output"
    )
    required_words: list[str] = Field(
        default_factory=list, description="Words that must appear in output"
    )
    forbidden_patterns: list[str] = Field(
        default_factory=list, description="Regex patterns that should not match"
    )
    required_patterns: list[str] = Field(
        default_factory=list, description="Regex patterns that must match"
    )


class StyleMetrics(BaseModel):
    """Computed style metrics for a text."""

    # Readability scores
    flesch_kincaid_grade: float = Field(..., description="Flesch-Kincaid Grade Level")
    flesch_reading_ease: float = Field(
        ..., description="Flesch Reading Ease score (0-100)"
    )
    smog_grade: float = Field(..., description="SMOG Grade Level")
    coleman_liau_grade: float = Field(..., description="Coleman-Liau Grade Level")

    # Structure metrics
    passive_voice_percentage: float = Field(
        ..., description="Percentage of sentences with passive voice"
    )
    avg_sentence_length: float = Field(..., description="Average words per sentence")
    max_sentence_length: int = Field(..., description="Maximum words in any sentence")
    total_sentences: int = Field(..., description="Total number of sentences")
    total_words: int = Field(..., description="Total number of words")

    # Tone scores (0-1 for each type)
    tone_scores: dict[str, float] = Field(
        default_factory=dict, description="Tone scores for each tone type"
    )


# Tone indicator words for each tone type
TONE_INDICATORS: dict[str, dict[str, list[str]]] = {
    ToneType.PROFESSIONAL: {
        "positive": [
            "accordingly",
            "analysis",
            "appropriate",
            "comprehensive",
            "consequently",
            "demonstrate",
            "ensure",
            "establish",
            "evaluate",
            "facilitate",
            "furthermore",
            "implement",
            "indicate",
            "moreover",
            "objective",
            "optimize",
            "parameter",
            "pursuant",
            "recommendation",
            "regarding",
            "respectively",
            "subsequently",
            "therefore",
            "utilize",
        ],
        "negative": [
            "awesome",
            "cool",
            "gonna",
            "gotta",
            "hey",
            "lol",
            "nope",
            "stuff",
            "wanna",
            "yeah",
            "yep",
        ],
    },
    ToneType.CASUAL: {
        "positive": [
            "awesome",
            "basically",
            "btw",
            "cool",
            "definitely",
            "easy",
            "fun",
            "gonna",
            "great",
            "hey",
            "just",
            "like",
            "nice",
            "okay",
            "pretty",
            "quick",
            "really",
            "simple",
            "so",
            "super",
            "thanks",
            "thing",
            "totally",
            "well",
            "yeah",
        ],
        "negative": [
            "accordingly",
            "heretofore",
            "hereby",
            "notwithstanding",
            "pursuant",
            "whereas",
            "whereby",
            "wherefore",
        ],
    },
    ToneType.FORMAL: {
        "positive": [
            "accordingly",
            "aforementioned",
            "consequently",
            "constitute",
            "deem",
            "forthwith",
            "henceforth",
            "heretofore",
            "hereby",
            "herein",
            "nevertheless",
            "notwithstanding",
            "pursuant",
            "shall",
            "thereby",
            "therefore",
            "thus",
            "upon",
            "whereas",
            "whereby",
            "wherein",
            "whereof",
        ],
        "negative": [
            "awesome",
            "btw",
            "cool",
            "gonna",
            "gotta",
            "hey",
            "lol",
            "stuff",
            "wanna",
            "yeah",
            "yep",
        ],
    },
    ToneType.FRIENDLY: {
        "positive": [
            "absolutely",
            "amazing",
            "appreciate",
            "delighted",
            "enjoy",
            "excited",
            "fantastic",
            "glad",
            "grateful",
            "great",
            "happy",
            "help",
            "hope",
            "love",
            "nice",
            "perfect",
            "pleased",
            "support",
            "thanks",
            "welcome",
            "wonderful",
        ],
        "negative": [
            "demand",
            "failure",
            "impossible",
            "inadequate",
            "must",
            "never",
            "refuse",
            "unacceptable",
            "unfortunately",
        ],
    },
}


# Passive voice patterns (auxiliary verb + past participle patterns)
PASSIVE_PATTERNS = [
    r"\b(am|is|are|was|were|be|been|being)\s+(\w+ed|written|done|made|given|taken|seen|known|found|said|told|thought|become|begun|broken|brought|built|bought|caught|chosen|come|cut|drawn|driven|eaten|fallen|felt|fought|flown|forgotten|forgiven|frozen|gotten|gone|grown|had|heard|held|hidden|hit|hurt|kept|known|laid|led|left|lent|let|lain|lit|lost|made|meant|met|paid|put|read|ridden|rung|risen|run|said|sat|seen|sold|sent|set|shaken|shone|shot|shown|shut|sung|sunk|slept|slid|spoken|spent|spun|spread|stood|stolen|stuck|stung|struck|sworn|swept|swum|taken|taught|torn|thought|thrown|understood|woken|worn|won|wound|written)\b",
]


class TextAnalyzer:
    """Analyzes text for style metrics."""

    def __init__(self, text: str) -> None:
        """Initialize analyzer with text.

        Args:
            text: The text to analyze.
        """
        self.text = text
        self.sentences = self._split_sentences()
        self.words = self._extract_words()
        self.syllable_count = self._count_syllables()

    def _split_sentences(self) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations and edge cases
        text = self.text
        # Protect common abbreviations
        abbreviations = [
            "Mr.",
            "Mrs.",
            "Ms.",
            "Dr.",
            "Prof.",
            "Inc.",
            "Ltd.",
            "Jr.",
            "Sr.",
            "vs.",
            "e.g.",
            "i.e.",
            "etc.",
            "al.",
            "Fig.",
            "fig.",
        ]
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace(".", "<PERIOD>"))

        # Split on sentence boundaries
        sentences = re.split(r"[.!?]+\s*", text)

        # Restore abbreviations and clean
        sentences = [s.replace("<PERIOD>", ".").strip() for s in sentences if s.strip()]
        return sentences

    def _extract_words(self) -> list[str]:
        """Extract words from text."""
        # Remove punctuation and split
        text = re.sub(r"[^\w\s'-]", " ", self.text.lower())
        words = text.split()
        # Filter out empty strings and pure punctuation
        return [w for w in words if w and re.search(r"\w", w)]

    def _count_syllables(self) -> int:
        """Count total syllables in text."""
        return sum(self._syllables_in_word(word) for word in self.words)

    def _syllables_in_word(self, word: str) -> int:
        """Count syllables in a word using regex-based estimation."""
        word = word.lower().strip()
        if not word:
            return 0

        # Handle common exceptions
        exceptions = {
            "the": 1,
            "a": 1,
            "an": 1,
            "i": 1,
            "is": 1,
            "are": 1,
            "were": 1,
            "your": 1,
            "their": 1,
            "there": 1,
            "they": 1,
            "we": 1,
            "he": 1,
            "she": 1,
            "it": 1,
            "you": 1,
            "be": 1,
            "been": 1,
            "being": 2,
            "have": 1,
            "has": 1,
            "had": 1,
            "do": 1,
            "does": 1,
            "did": 1,
        }
        if word in exceptions:
            return exceptions[word]

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for i, char in enumerate(word):
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Handle silent e at end
        if word.endswith("e") and count > 1:
            count -= 1

        # Handle -le endings
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        # Handle -ed endings (usually silent unless preceded by t or d)
        if word.endswith("ed") and len(word) > 2:
            if word[-3] not in "td":
                count -= 1

        # Minimum of 1 syllable
        return max(1, count)

    def _count_complex_words(self) -> int:
        """Count words with 3+ syllables (for SMOG calculation)."""
        return sum(1 for word in self.words if self._syllables_in_word(word) >= 3)

    def _count_characters_in_words(self) -> int:
        """Count total characters in all words (letters only)."""
        return sum(len(re.sub(r"[^a-zA-Z]", "", word)) for word in self.words)

    def calculate_flesch_kincaid_grade(self) -> float:
        """Calculate Flesch-Kincaid Grade Level.

        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        """
        if not self.sentences or not self.words:
            return 0.0

        avg_sentence_length = len(self.words) / len(self.sentences)
        avg_syllables_per_word = self.syllable_count / len(self.words)

        grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        return max(0.0, round(grade, 2))

    def calculate_flesch_reading_ease(self) -> float:
        """Calculate Flesch Reading Ease score.

        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        Score interpretation:
        - 90-100: Very easy (5th grade)
        - 80-89: Easy (6th grade)
        - 70-79: Fairly easy (7th grade)
        - 60-69: Standard (8th-9th grade)
        - 50-59: Fairly difficult (10th-12th grade)
        - 30-49: Difficult (college)
        - 0-29: Very difficult (graduate)
        """
        if not self.sentences or not self.words:
            return 100.0

        avg_sentence_length = len(self.words) / len(self.sentences)
        avg_syllables_per_word = self.syllable_count / len(self.words)

        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        # Clamp to 0-100 range
        return max(0.0, min(100.0, round(score, 2)))

    def calculate_smog_grade(self) -> float:
        """Calculate SMOG (Simple Measure of Gobbledygook) grade level.

        Formula: 1.0430 * sqrt(complex_words * (30/sentences)) + 3.1291
        Complex words = words with 3+ syllables
        """
        if not self.sentences or len(self.sentences) < 3:
            # SMOG requires at least 30 sentences ideally, but we'll adapt
            return self.calculate_flesch_kincaid_grade()

        complex_words = self._count_complex_words()
        polysyllable_count = complex_words * (30 / len(self.sentences))

        grade = 1.0430 * (polysyllable_count**0.5) + 3.1291
        return max(0.0, round(grade, 2))

    def calculate_coleman_liau_grade(self) -> float:
        """Calculate Coleman-Liau Index grade level.

        Formula: 0.0588 * L - 0.296 * S - 15.8
        L = average number of letters per 100 words
        S = average number of sentences per 100 words
        """
        if not self.words or not self.sentences:
            return 0.0

        chars = self._count_characters_in_words()
        letters_per_100 = (chars / len(self.words)) * 100
        sentences_per_100 = (len(self.sentences) / len(self.words)) * 100

        grade = 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8
        return max(0.0, round(grade, 2))

    def calculate_passive_voice_percentage(self) -> float:
        """Calculate percentage of sentences with passive voice."""
        if not self.sentences:
            return 0.0

        passive_count = 0
        for sentence in self.sentences:
            for pattern in PASSIVE_PATTERNS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    passive_count += 1
                    break

        return round((passive_count / len(self.sentences)) * 100, 2)

    def calculate_sentence_lengths(self) -> tuple[float, int]:
        """Calculate average and maximum sentence length in words.

        Returns:
            Tuple of (average_length, max_length)
        """
        if not self.sentences:
            return 0.0, 0

        lengths = []
        for sentence in self.sentences:
            words = re.findall(r"\b\w+\b", sentence)
            lengths.append(len(words))

        avg_length = sum(lengths) / len(lengths) if lengths else 0.0
        max_length = max(lengths) if lengths else 0

        return round(avg_length, 2), max_length

    def calculate_tone_scores(self) -> dict[str, float]:
        """Calculate tone scores for each tone type.

        Returns:
            Dictionary mapping tone type to score (0-1).
        """
        words_lower = {w.lower() for w in self.words}
        scores = {}

        for tone_type in ToneType:
            indicators = TONE_INDICATORS[tone_type]
            positive_words = set(indicators["positive"])
            negative_words = set(indicators["negative"])

            positive_matches = len(words_lower & positive_words)
            negative_matches = len(words_lower & negative_words)

            # Calculate score based on positive/negative matches
            # More positive matches = higher score
            # Negative matches reduce score
            total_indicators = len(positive_words) + len(negative_words)
            if total_indicators == 0:
                scores[tone_type.value] = 0.5
                continue

            # Weighted score: positive matches add, negative subtract
            raw_score = (positive_matches * 2 - negative_matches) / (
                len(self.words) or 1
            )
            # Normalize to 0-1 range with sigmoid-like function
            normalized = 0.5 + min(0.5, max(-0.5, raw_score * 10))
            scores[tone_type.value] = round(normalized, 3)

        return scores

    def get_metrics(self) -> StyleMetrics:
        """Calculate all style metrics for the text.

        Returns:
            StyleMetrics containing all calculated metrics.
        """
        avg_sentence_length, max_sentence_length = self.calculate_sentence_lengths()

        return StyleMetrics(
            flesch_kincaid_grade=self.calculate_flesch_kincaid_grade(),
            flesch_reading_ease=self.calculate_flesch_reading_ease(),
            smog_grade=self.calculate_smog_grade(),
            coleman_liau_grade=self.calculate_coleman_liau_grade(),
            passive_voice_percentage=self.calculate_passive_voice_percentage(),
            avg_sentence_length=avg_sentence_length,
            max_sentence_length=max_sentence_length,
            total_sentences=len(self.sentences),
            total_words=len(self.words),
            tone_scores=self.calculate_tone_scores(),
        )


class StyleEvaluator(Evaluator):
    """Evaluator for writing style, tone, and readability.

    This evaluator assesses agent outputs for:
    - Tone analysis (professional, casual, formal, friendly)
    - Readability metrics (Flesch-Kincaid, SMOG, Coleman-Liau, etc.)
    - Passive voice percentage
    - Sentence length analysis
    - Custom style rules (forbidden/required words and patterns)

    Example assertion config:
        {
            "type": "style",
            "config": {
                "expected_tone": "professional",
                "tone_threshold": 0.7,
                "max_flesch_kincaid_grade": 12.0,
                "min_flesch_reading_ease": 50.0,
                "max_passive_voice_percentage": 20.0,
                "max_avg_sentence_length": 25.0,
                "forbidden_words": ["awesome", "cool"],
                "required_words": ["therefore", "consequently"]
            }
        }
    """

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "style"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """Evaluate style and tone of agent output.

        Args:
            task: Test definition containing task details.
            response: ATP Response from the agent.
            trace: List of ATP Events from execution.
            assertion: Assertion to evaluate against.

        Returns:
            EvalResult containing style check results.
        """
        assertion_type = assertion.type
        config_dict = assertion.config or {}

        if assertion_type == "style":
            return await self._evaluate_style(response, config_dict)
        elif assertion_type == "tone":
            return await self._evaluate_tone(response, config_dict)
        elif assertion_type == "readability":
            return await self._evaluate_readability(response, config_dict)
        elif assertion_type == "passive_voice":
            return await self._evaluate_passive_voice(response, config_dict)
        elif assertion_type == "sentence_length":
            return await self._evaluate_sentence_length(response, config_dict)
        elif assertion_type == "style_rules":
            return await self._evaluate_style_rules(response, config_dict)
        else:
            check = self._create_check(
                name="unknown_assertion",
                passed=False,
                message=f"Unknown style assertion type: {assertion_type}",
                details={"assertion_type": assertion_type},
            )
            return self._create_result([check])

    async def _evaluate_style(
        self, response: ATPResponse, config_dict: dict[str, Any]
    ) -> EvalResult:
        """Evaluate all style aspects based on configuration."""
        config = StyleConfig(**config_dict)
        text = self._extract_text_from_response(response)

        if not text:
            check = self._create_check(
                name="no_text",
                passed=False,
                message="No text content found in response to evaluate",
                details={"artifacts_count": len(response.artifacts or [])},
            )
            return self._create_result([check])

        analyzer = TextAnalyzer(text)
        metrics = analyzer.get_metrics()
        checks: list[EvalCheck] = []

        # Add metrics check (always passes, informational)
        checks.append(
            self._create_check(
                name="style_metrics",
                passed=True,
                message="Style metrics computed successfully",
                details=metrics.model_dump(),
            )
        )

        # Tone check
        if config.expected_tone:
            tone_check = self._check_tone(metrics, config)
            checks.append(tone_check)

        # Readability checks
        readability_checks = self._check_readability(metrics, config)
        checks.extend(readability_checks)

        # Passive voice check
        if config.max_passive_voice_percentage is not None:
            passive_check = self._check_passive_voice(metrics, config)
            checks.append(passive_check)

        # Sentence length checks
        sentence_checks = self._check_sentence_length(metrics, config)
        checks.extend(sentence_checks)

        # Custom style rules
        rule_checks = self._check_style_rules(text, config)
        checks.extend(rule_checks)

        return self._create_result(checks)

    async def _evaluate_tone(
        self, response: ATPResponse, config_dict: dict[str, Any]
    ) -> EvalResult:
        """Evaluate only tone aspect."""
        text = self._extract_text_from_response(response)
        if not text:
            check = self._create_check(
                name="no_text",
                passed=False,
                message="No text content found in response",
            )
            return self._create_result([check])

        expected_tone = config_dict.get("expected_tone")
        threshold = config_dict.get("tone_threshold", 0.7)

        if not expected_tone:
            check = self._create_check(
                name="tone_config_missing",
                passed=False,
                message="expected_tone must be specified in config",
            )
            return self._create_result([check])

        analyzer = TextAnalyzer(text)
        metrics = analyzer.get_metrics()
        config = StyleConfig(expected_tone=expected_tone, tone_threshold=threshold)
        tone_check = self._check_tone(metrics, config)

        return self._create_result([tone_check])

    async def _evaluate_readability(
        self, response: ATPResponse, config_dict: dict[str, Any]
    ) -> EvalResult:
        """Evaluate only readability aspects."""
        text = self._extract_text_from_response(response)
        if not text:
            check = self._create_check(
                name="no_text",
                passed=False,
                message="No text content found in response",
            )
            return self._create_result([check])

        analyzer = TextAnalyzer(text)
        metrics = analyzer.get_metrics()
        config = StyleConfig(**config_dict)
        checks = self._check_readability(metrics, config)

        if not checks:
            # Return metrics as informational if no thresholds specified
            checks = [
                self._create_check(
                    name="readability_metrics",
                    passed=True,
                    message="Readability metrics computed",
                    details={
                        "flesch_kincaid_grade": metrics.flesch_kincaid_grade,
                        "flesch_reading_ease": metrics.flesch_reading_ease,
                        "smog_grade": metrics.smog_grade,
                        "coleman_liau_grade": metrics.coleman_liau_grade,
                    },
                )
            ]

        return self._create_result(checks)

    async def _evaluate_passive_voice(
        self, response: ATPResponse, config_dict: dict[str, Any]
    ) -> EvalResult:
        """Evaluate passive voice percentage."""
        text = self._extract_text_from_response(response)
        if not text:
            check = self._create_check(
                name="no_text",
                passed=False,
                message="No text content found in response",
            )
            return self._create_result([check])

        max_percentage = config_dict.get("max_passive_voice_percentage")
        if max_percentage is None:
            check = self._create_check(
                name="config_missing",
                passed=False,
                message="max_passive_voice_percentage must be specified",
            )
            return self._create_result([check])

        analyzer = TextAnalyzer(text)
        actual = analyzer.calculate_passive_voice_percentage()
        passed = actual <= max_percentage

        check = self._create_check(
            name="passive_voice",
            passed=passed,
            message=(
                f"Passive voice: {actual}%"
                if passed
                else f"Passive voice {actual}% exceeds maximum {max_percentage}%"
            ),
            details={
                "actual_percentage": actual,
                "max_percentage": max_percentage,
                "total_sentences": len(analyzer.sentences),
            },
        )
        return self._create_result([check])

    async def _evaluate_sentence_length(
        self, response: ATPResponse, config_dict: dict[str, Any]
    ) -> EvalResult:
        """Evaluate sentence length metrics."""
        text = self._extract_text_from_response(response)
        if not text:
            check = self._create_check(
                name="no_text",
                passed=False,
                message="No text content found in response",
            )
            return self._create_result([check])

        analyzer = TextAnalyzer(text)
        metrics = analyzer.get_metrics()
        config = StyleConfig(**config_dict)
        checks = self._check_sentence_length(metrics, config)

        if not checks:
            checks = [
                self._create_check(
                    name="sentence_length_metrics",
                    passed=True,
                    message="Sentence length metrics computed",
                    details={
                        "avg_sentence_length": metrics.avg_sentence_length,
                        "max_sentence_length": metrics.max_sentence_length,
                        "total_sentences": metrics.total_sentences,
                    },
                )
            ]

        return self._create_result(checks)

    async def _evaluate_style_rules(
        self, response: ATPResponse, config_dict: dict[str, Any]
    ) -> EvalResult:
        """Evaluate custom style rules."""
        text = self._extract_text_from_response(response)
        if not text:
            check = self._create_check(
                name="no_text",
                passed=False,
                message="No text content found in response",
            )
            return self._create_result([check])

        config = StyleConfig(**config_dict)
        checks = self._check_style_rules(text, config)

        if not checks:
            checks = [
                self._create_check(
                    name="no_rules",
                    passed=True,
                    message="No custom style rules specified",
                )
            ]

        return self._create_result(checks)

    def _extract_text_from_response(self, response: ATPResponse) -> str:
        """Extract text content from response artifacts.

        Args:
            response: The ATP response to extract text from.

        Returns:
            Combined text content from all text artifacts.
        """
        texts: list[str] = []

        if response.artifacts:
            for artifact in response.artifacts:
                # Get content from artifact
                if hasattr(artifact, "content") and artifact.content:
                    texts.append(str(artifact.content))
                elif hasattr(artifact, "data") and artifact.data:
                    if isinstance(artifact.data, str):
                        texts.append(artifact.data)
                    elif isinstance(artifact.data, dict):
                        # Try to extract text from common fields
                        for key in ["text", "content", "body", "message", "output"]:
                            if key in artifact.data and isinstance(
                                artifact.data[key], str
                            ):
                                texts.append(artifact.data[key])

        return "\n".join(texts)

    def _check_tone(self, metrics: StyleMetrics, config: StyleConfig) -> EvalCheck:
        """Check if tone matches expected tone."""
        if not config.expected_tone:
            return self._create_check(
                name="tone",
                passed=True,
                message="No expected tone specified",
            )

        expected = config.expected_tone.value
        actual_score = metrics.tone_scores.get(expected, 0.0)
        passed = actual_score >= config.tone_threshold

        # Find dominant tone
        dominant_tone = max(metrics.tone_scores.items(), key=lambda x: x[1])

        return self._create_check(
            name="tone",
            passed=passed,
            message=(
                f"Tone matches expected '{expected}' (score: {actual_score:.2f})"
                if passed
                else f"Tone score for '{expected}' is {actual_score:.2f}, "
                f"below threshold {config.tone_threshold:.2f}. "
                f"Dominant tone: '{dominant_tone[0]}' ({dominant_tone[1]:.2f})"
            ),
            details={
                "expected_tone": expected,
                "threshold": config.tone_threshold,
                "actual_score": actual_score,
                "all_scores": metrics.tone_scores,
                "dominant_tone": dominant_tone[0],
            },
        )

    def _check_readability(
        self, metrics: StyleMetrics, config: StyleConfig
    ) -> list[EvalCheck]:
        """Check readability metrics against thresholds."""
        checks: list[EvalCheck] = []

        # Flesch-Kincaid Grade Level
        if config.max_flesch_kincaid_grade is not None:
            passed = metrics.flesch_kincaid_grade <= config.max_flesch_kincaid_grade
            checks.append(
                self._create_check(
                    name="flesch_kincaid_grade",
                    passed=passed,
                    message=(
                        f"Flesch-Kincaid Grade: {metrics.flesch_kincaid_grade}"
                        if passed
                        else f"Flesch-Kincaid Grade {metrics.flesch_kincaid_grade} "
                        f"exceeds maximum {config.max_flesch_kincaid_grade}"
                    ),
                    details={
                        "actual": metrics.flesch_kincaid_grade,
                        "max": config.max_flesch_kincaid_grade,
                    },
                )
            )

        # Flesch Reading Ease
        if config.min_flesch_reading_ease is not None:
            passed = metrics.flesch_reading_ease >= config.min_flesch_reading_ease
            checks.append(
                self._create_check(
                    name="flesch_reading_ease",
                    passed=passed,
                    message=(
                        f"Flesch Reading Ease: {metrics.flesch_reading_ease}"
                        if passed
                        else f"Flesch Reading Ease {metrics.flesch_reading_ease} "
                        f"below minimum {config.min_flesch_reading_ease}"
                    ),
                    details={
                        "actual": metrics.flesch_reading_ease,
                        "min": config.min_flesch_reading_ease,
                    },
                )
            )

        # SMOG Grade
        if config.max_smog_grade is not None:
            passed = metrics.smog_grade <= config.max_smog_grade
            checks.append(
                self._create_check(
                    name="smog_grade",
                    passed=passed,
                    message=(
                        f"SMOG Grade: {metrics.smog_grade}"
                        if passed
                        else f"SMOG Grade {metrics.smog_grade} "
                        f"exceeds maximum {config.max_smog_grade}"
                    ),
                    details={
                        "actual": metrics.smog_grade,
                        "max": config.max_smog_grade,
                    },
                )
            )

        # Coleman-Liau Grade
        if config.max_coleman_liau_grade is not None:
            passed = metrics.coleman_liau_grade <= config.max_coleman_liau_grade
            checks.append(
                self._create_check(
                    name="coleman_liau_grade",
                    passed=passed,
                    message=(
                        f"Coleman-Liau Grade: {metrics.coleman_liau_grade}"
                        if passed
                        else f"Coleman-Liau Grade {metrics.coleman_liau_grade} "
                        f"exceeds maximum {config.max_coleman_liau_grade}"
                    ),
                    details={
                        "actual": metrics.coleman_liau_grade,
                        "max": config.max_coleman_liau_grade,
                    },
                )
            )

        return checks

    def _check_passive_voice(
        self, metrics: StyleMetrics, config: StyleConfig
    ) -> EvalCheck:
        """Check passive voice percentage against threshold."""
        if config.max_passive_voice_percentage is None:
            return self._create_check(
                name="passive_voice",
                passed=True,
                message=f"Passive voice: {metrics.passive_voice_percentage}%",
                details={"percentage": metrics.passive_voice_percentage},
            )

        passed = metrics.passive_voice_percentage <= config.max_passive_voice_percentage
        return self._create_check(
            name="passive_voice",
            passed=passed,
            message=(
                f"Passive voice: {metrics.passive_voice_percentage}%"
                if passed
                else f"Passive voice {metrics.passive_voice_percentage}% "
                f"exceeds maximum {config.max_passive_voice_percentage}%"
            ),
            details={
                "actual": metrics.passive_voice_percentage,
                "max": config.max_passive_voice_percentage,
            },
        )

    def _check_sentence_length(
        self, metrics: StyleMetrics, config: StyleConfig
    ) -> list[EvalCheck]:
        """Check sentence length metrics against thresholds."""
        checks: list[EvalCheck] = []

        # Max average sentence length
        if config.max_avg_sentence_length is not None:
            passed = metrics.avg_sentence_length <= config.max_avg_sentence_length
            checks.append(
                self._create_check(
                    name="avg_sentence_length",
                    passed=passed,
                    message=(
                        f"Average sentence length: {metrics.avg_sentence_length} words"
                        if passed
                        else f"Average sentence length {metrics.avg_sentence_length} "
                        f"exceeds maximum {config.max_avg_sentence_length} words"
                    ),
                    details={
                        "actual": metrics.avg_sentence_length,
                        "max": config.max_avg_sentence_length,
                    },
                )
            )

        # Min average sentence length
        if config.min_avg_sentence_length is not None:
            passed = metrics.avg_sentence_length >= config.min_avg_sentence_length
            checks.append(
                self._create_check(
                    name="min_avg_sentence_length",
                    passed=passed,
                    message=(
                        f"Average sentence length: {metrics.avg_sentence_length} words"
                        if passed
                        else f"Average sentence length {metrics.avg_sentence_length} "
                        f"below minimum {config.min_avg_sentence_length} words"
                    ),
                    details={
                        "actual": metrics.avg_sentence_length,
                        "min": config.min_avg_sentence_length,
                    },
                )
            )

        # Max single sentence length
        if config.max_sentence_length is not None:
            passed = metrics.max_sentence_length <= config.max_sentence_length
            checks.append(
                self._create_check(
                    name="max_sentence_length",
                    passed=passed,
                    message=(
                        f"Longest sentence: {metrics.max_sentence_length} words"
                        if passed
                        else (
                            f"Longest sentence has {metrics.max_sentence_length} "
                            f"words, exceeds maximum {config.max_sentence_length} words"
                        )
                    ),
                    details={
                        "actual": metrics.max_sentence_length,
                        "max": config.max_sentence_length,
                    },
                )
            )

        return checks

    def _check_style_rules(self, text: str, config: StyleConfig) -> list[EvalCheck]:
        """Check custom style rules (forbidden/required words and patterns)."""
        checks: list[EvalCheck] = []
        text_lower = text.lower()
        words_in_text = set(re.findall(r"\b\w+\b", text_lower))

        # Forbidden words
        if config.forbidden_words:
            forbidden_found = [
                w for w in config.forbidden_words if w.lower() in words_in_text
            ]
            passed = len(forbidden_found) == 0
            checks.append(
                self._create_check(
                    name="forbidden_words",
                    passed=passed,
                    message=(
                        "No forbidden words found"
                        if passed
                        else f"Forbidden words found: {', '.join(forbidden_found)}"
                    ),
                    details={
                        "forbidden": config.forbidden_words,
                        "found": forbidden_found,
                    },
                )
            )

        # Required words
        if config.required_words:
            required_missing = [
                w for w in config.required_words if w.lower() not in words_in_text
            ]
            passed = len(required_missing) == 0
            checks.append(
                self._create_check(
                    name="required_words",
                    passed=passed,
                    message=(
                        "All required words found"
                        if passed
                        else f"Missing required words: {', '.join(required_missing)}"
                    ),
                    details={
                        "required": config.required_words,
                        "missing": required_missing,
                    },
                )
            )

        # Forbidden patterns
        if config.forbidden_patterns:
            pattern_matches = []
            for pattern in config.forbidden_patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        pattern_matches.append(pattern)
                except re.error:
                    pass  # Skip invalid regex patterns
            passed = len(pattern_matches) == 0
            checks.append(
                self._create_check(
                    name="forbidden_patterns",
                    passed=passed,
                    message=(
                        "No forbidden patterns matched"
                        if passed
                        else f"Forbidden patterns matched: {', '.join(pattern_matches)}"
                    ),
                    details={
                        "forbidden": config.forbidden_patterns,
                        "matched": pattern_matches,
                    },
                )
            )

        # Required patterns
        if config.required_patterns:
            patterns_missing = []
            for pattern in config.required_patterns:
                try:
                    if not re.search(pattern, text, re.IGNORECASE):
                        patterns_missing.append(pattern)
                except re.error:
                    patterns_missing.append(f"{pattern} (invalid regex)")
            passed = len(patterns_missing) == 0
            checks.append(
                self._create_check(
                    name="required_patterns",
                    passed=passed,
                    message=(
                        "All required patterns matched"
                        if passed
                        else f"Missing required patterns: {', '.join(patterns_missing)}"
                    ),
                    details={
                        "required": config.required_patterns,
                        "missing": patterns_missing,
                    },
                )
            )

        return checks
