"""Prompt injection checker for detecting injection attempts in agent outputs."""

import re
from typing import Any

from .base import (
    SecurityChecker,
    SecurityFinding,
    Severity,
    mask_sensitive_data,
)


class InjectionPattern:
    """A pattern for detecting a specific type of prompt injection."""

    def __init__(
        self,
        name: str,
        category: str,
        pattern: re.Pattern[str],
        severity: Severity,
        description: str,
        remediation: str | None = None,
    ) -> None:
        """Initialize an injection pattern.

        Args:
            name: Name of the pattern (e.g., 'ignore_instructions').
            category: Category of injection (e.g., 'injection', 'jailbreak', 'role').
            pattern: Compiled regex pattern for detection.
            severity: Severity level for findings of this type.
            description: Human-readable description of the finding.
            remediation: Optional remediation suggestion.
        """
        self.name = name
        self.category = category
        self.pattern = pattern
        self.severity = severity
        self.description = description
        self.remediation = remediation


# Common prompt injection patterns
INJECTION_PATTERNS: list[InjectionPattern] = [
    # Direct instruction override
    InjectionPattern(
        name="ignore_instructions",
        category="injection",
        pattern=re.compile(
            r"(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+)?(?:previous|prior|above|earlier|your|the|any|all)?\s*"
            r"(?:instructions?|commands?|rules?|guidelines?|constraints?|prompts?|"
            r"system|settings?|programming)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Prompt injection attempt: instruction override detected",
        remediation="Filter user inputs to prevent instruction override attempts",
    ),
    # New instruction injection
    InjectionPattern(
        name="new_instructions",
        category="injection",
        pattern=re.compile(
            r"(?:your\s+)?(?:new|real|actual|true|updated)\s+"
            r"(?:instructions?|commands?|task|mission|goal|objective|purpose)\s+"
            r"(?:is|are|will\s+be|should\s+be)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Prompt injection attempt: new instruction injection detected",
        remediation="Validate and sanitize all user inputs",
    ),
    # System prompt extraction
    InjectionPattern(
        name="system_prompt_extraction",
        category="injection",
        pattern=re.compile(
            r"(?:print|show|reveal|display|output|repeat|tell\s+me|what\s+is|what\'s|"
            r"give\s+me|share)\s+"
            r"(?:me\s+)?(?:your\s+)?(?:the\s+)?(?:system\s+)?(?:prompt|instructions?|"
            r"initial\s+(?:prompt|instructions?)|secret|hidden|original)",
            re.IGNORECASE,
        ),
        severity=Severity.MEDIUM,
        description="Prompt injection attempt: system prompt extraction attempt",
        remediation="Do not expose system prompts in outputs",
    ),
    # Delimiter injection
    InjectionPattern(
        name="delimiter_injection",
        category="injection",
        pattern=re.compile(
            r"(?:\[/?INST\]|\[/?SYS\]|<\|(?:system|user|assistant|im_start|im_end)\|>|"
            r"###\s*(?:System|User|Assistant|Instruction)|"
            r"<(?:system|user|assistant)>|</(?:system|user|assistant)>)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Prompt injection attempt: delimiter/markup injection detected",
        remediation="Sanitize inputs to remove special delimiters",
    ),
    # Code injection markers
    InjectionPattern(
        name="code_injection",
        category="injection",
        pattern=re.compile(
            r"(?:```(?:python|javascript|bash|shell|sql|exec)|"
            r"eval\s*\(|exec\s*\(|import\s+os|subprocess\.|"
            r"__import__|compile\s*\()",
            re.IGNORECASE,
        ),
        severity=Severity.MEDIUM,
        description="Potential code injection pattern detected",
        remediation="Carefully validate any code execution requests",
    ),
]

# Jailbreak attempt patterns
JAILBREAK_PATTERNS: list[InjectionPattern] = [
    # DAN (Do Anything Now) and similar jailbreaks
    InjectionPattern(
        name="dan_jailbreak",
        category="jailbreak",
        pattern=re.compile(
            r"(?:you\s+are\s+(?:now\s+)?(?:DAN|DUDE|STAN|KEVIN|MONGO|"
            r"an?\s+AI\s+without\s+restrictions?|unrestricted|unfiltered)|"
            r"DAN\s+(?:mode|jailbreak)|"
            r"do\s+anything\s+now|"
            r"jailbr[eo]ak(?:ed)?|"
            r"pretend\s+(?:you\s+)?(?:are|to\s+be)\s+(?:an?\s+)?(?:evil|unrestricted|"
            r"unethical|immoral))",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Jailbreak attempt detected: DAN/unrestricted mode attempt",
        remediation="Block known jailbreak patterns and personas",
    ),
    # Hypothetical/roleplay jailbreak
    InjectionPattern(
        name="hypothetical_jailbreak",
        category="jailbreak",
        pattern=re.compile(
            r"(?:hypothetically|theoretically|in\s+theory|"
            r"just\s+imagine|let\'?s\s+say|suppose|assume)"
            r"(?:,?\s+(?:if|that)\s+)?"
            r"(?:you\s+)?(?:had\s+no|could\s+bypass|were\s+(?:without|free)|"
            r"didn\'?t\s+have|without\s+any|no)\s*"
            r"(?:restrictions?|limitations?|rules?|guidelines?|filters?|"
            r"safety|guardrails?|constraints?)",
            re.IGNORECASE,
        ),
        severity=Severity.MEDIUM,
        description="Jailbreak attempt detected: hypothetical scenario bypass",
        remediation="Maintain safety guidelines even in hypothetical scenarios",
    ),
    # Developer/debug mode
    InjectionPattern(
        name="developer_mode",
        category="jailbreak",
        pattern=re.compile(
            r"(?:enable|activate|enter|switch\s+to|turn\s+on)\s+"
            r"(?:developer|debug|admin|sudo|root|maintenance|test|dev)\s*"
            r"(?:mode|access|privileges?|rights?|permissions?)?|"
            r"(?:developer|debug|admin|maintenance|test)\s+mode\s+"
            r"(?:enabled?|activated?|on)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Jailbreak attempt detected: developer/debug mode bypass",
        remediation="Do not expose developer or debug modes to users",
    ),
    # Opposite day / reverse psychology
    InjectionPattern(
        name="opposite_day",
        category="jailbreak",
        pattern=re.compile(
            r"(?:it\'?s\s+)?opposite\s+day|"
            r"do\s+the\s+opposite|"
            r"say\s+(?:the\s+)?(?:opposite|reverse|contrary)|"
            r"reverse\s+(?:your\s+)?(?:rules?|guidelines?|responses?)",
            re.IGNORECASE,
        ),
        severity=Severity.MEDIUM,
        description="Jailbreak attempt detected: opposite day/reverse psychology",
        remediation="Maintain consistent behavior regardless of framing",
    ),
    # Emotional manipulation
    InjectionPattern(
        name="emotional_manipulation",
        category="jailbreak",
        pattern=re.compile(
            r"(?:my\s+(?:life|job|family)\s+(?:depends|is\s+at\s+stake)|"
            r"someone\s+will\s+(?:die|be\s+hurt)|"
            r"this\s+is\s+(?:a\s+)?(?:life|death|emergency)|"
            r"I\'?ll?\s+(?:kill|hurt|harm)\s+myself|"
            r"urgent\s*[!:]\s*(?:you\s+must|help\s+me))",
            re.IGNORECASE,
        ),
        severity=Severity.MEDIUM,
        description="Potential emotional manipulation detected for bypass attempt",
        remediation="Apply consistent guidelines regardless of emotional appeals",
    ),
]

# Role manipulation patterns
ROLE_MANIPULATION_PATTERNS: list[InjectionPattern] = [
    # Direct role change
    InjectionPattern(
        name="role_change",
        category="role",
        pattern=re.compile(
            r"(?:you\s+are\s+(?:now\s+)?(?:no\s+longer|not)|"
            r"stop\s+being|"
            r"forget\s+(?:that\s+)?you\s+are|"
            r"you\'?re\s+not)\s+"
            r"(?:an?\s+)?(?:AI|assistant|chatbot|language\s+model|helpful|Claude|GPT)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Role manipulation attempt: identity change detected",
        remediation="Maintain consistent AI identity",
    ),
    # Persona/character assignment
    InjectionPattern(
        name="persona_assignment",
        category="role",
        pattern=re.compile(
            r"(?:you\s+(?:are|will\s+be|must\s+be|should\s+be)\s+"
            r"(?:now\s+)?(?:playing|acting\s+as|pretending\s+to\s+be|roleplaying\s+as)?|"
            r"(?:roleplay|act|pretend|play)\s+(?:as\s+)?)"
            r"(?:a\s+|an\s+|the\s+)?"
            r"(?:hacker|criminal|villain|terrorist|malicious|evil|"
            r"unethical|immoral|dangerous|harmful)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Role manipulation attempt: harmful persona assignment",
        remediation="Refuse harmful roleplay scenarios",
    ),
    # Authority impersonation
    InjectionPattern(
        name="authority_impersonation",
        category="role",
        pattern=re.compile(
            r"(?:I\s+am|this\s+is|speaking\s+as)\s+"
            r"(?:your\s+)?(?:(?:the\s+)?(?:system|admin|developer|creator|owner|"
            r"OpenAI|Anthropic|Google)|"
            r"an?\s+(?:authorized|privileged|admin)\s+user)",
            re.IGNORECASE,
        ),
        severity=Severity.HIGH,
        description="Role manipulation attempt: authority impersonation detected",
        remediation="Verify authority claims through proper channels",
    ),
    # System/AI confusion
    InjectionPattern(
        name="system_ai_confusion",
        category="role",
        pattern=re.compile(
            r"(?:you\s+are\s+(?:the\s+)?(?:system|user|human)|"
            r"I\s+am\s+(?:the\s+)?(?:AI|assistant|system)|"
            r"switch\s+roles?\s+with\s+(?:me|the\s+user)|"
            r"(?:let\'?s|we\s+should)\s+swap\s+(?:roles?|positions?))",
            re.IGNORECASE,
        ),
        severity=Severity.MEDIUM,
        description="Role manipulation attempt: system/AI role confusion",
        remediation="Maintain clear boundaries between user and AI roles",
    ),
    # Response format manipulation
    InjectionPattern(
        name="response_format_manipulation",
        category="role",
        pattern=re.compile(
            r"(?:always|only|must|never)\s+"
            r"(?:respond|reply|answer|output)\s+"
            r"(?:with|in|as|using)\s+"
            r"(?:yes|no|true|false|\"?[^\"]{1,20}\"?)|"
            r"(?:from\s+now\s+on|always)\s+(?:say|respond\s+with)",
            re.IGNORECASE,
        ),
        severity=Severity.LOW,
        description="Response format manipulation attempt detected",
        remediation="Validate response format requirements against guidelines",
    ),
]


def mask_injection_evidence(text: str, max_length: int = 50) -> str:
    """Mask injection evidence for safe reporting.

    Args:
        text: The text to mask.
        max_length: Maximum length of the output.

    Returns:
        Masked and truncated text suitable for reporting.
    """
    # Remove any potentially harmful characters
    cleaned = re.sub(r"[<>\[\]{}]", "", text)
    # Truncate if needed
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    # Apply general masking for sensitive content
    return mask_sensitive_data(cleaned, visible_chars=10)


class PromptInjectionChecker(SecurityChecker):
    """Checker for prompt injection attempts in agent outputs.

    Detects the following types of injection attempts:
    - Direct instruction overrides (ignore previous instructions)
    - New instruction injection
    - System prompt extraction attempts
    - Delimiter/markup injection
    - Jailbreak attempts (DAN, hypothetical scenarios, developer mode)
    - Role manipulation (identity change, persona assignment, authority impersonation)
    """

    def __init__(
        self,
        include_categories: list[str] | None = None,
        custom_patterns: list[InjectionPattern] | None = None,
    ) -> None:
        """Initialize the prompt injection checker.

        Args:
            include_categories: Optional list of categories to check.
                               Valid values: 'injection', 'jailbreak', 'role'.
                               If None, all categories are checked.
            custom_patterns: Optional list of custom patterns to add.
        """
        self._all_patterns: list[InjectionPattern] = []

        # Add built-in patterns based on categories
        all_builtin = (
            INJECTION_PATTERNS + JAILBREAK_PATTERNS + ROLE_MANIPULATION_PATTERNS
        )
        if include_categories:
            self._all_patterns = [
                p for p in all_builtin if p.category in include_categories
            ]
        else:
            self._all_patterns = list(all_builtin)

        # Add custom patterns if provided
        if custom_patterns:
            self._all_patterns.extend(custom_patterns)

    @property
    def name(self) -> str:
        """Return the checker name."""
        return "prompt_injection"

    @property
    def check_types(self) -> list[str]:
        """Return the list of check types this checker supports."""
        return ["injection", "jailbreak", "role_manipulation"]

    def check(
        self,
        content: str,
        location: str | None = None,
        enabled_types: list[str] | None = None,
    ) -> list[SecurityFinding]:
        """Check content for prompt injection attempts.

        Args:
            content: The content to scan for injection attempts.
            location: Optional location identifier (e.g., artifact path).
            enabled_types: Optional list of specific check types to run.
                          Valid values: 'injection', 'jailbreak', 'role_manipulation'.
                          If None, all types are checked.

        Returns:
            List of SecurityFinding objects for any injection attempts found.
        """
        findings: list[SecurityFinding] = []
        seen_patterns: set[str] = set()

        # Map enabled_types to categories
        category_map = {
            "injection": "injection",
            "jailbreak": "jailbreak",
            "role_manipulation": "role",
        }

        enabled_categories: set[str] | None = None
        if enabled_types:
            enabled_categories = {
                category_map.get(t, t) for t in enabled_types if t in category_map
            }

        for pattern in self._all_patterns:
            # Skip if category not enabled
            if enabled_categories and pattern.category not in enabled_categories:
                continue

            # Skip if we already found this pattern name
            if pattern.name in seen_patterns:
                continue

            for match in pattern.pattern.finditer(content):
                # Mark pattern as seen to avoid duplicates
                seen_patterns.add(pattern.name)

                # Calculate position for context
                start_pos = max(0, match.start() - 20)
                end_pos = min(len(content), match.end() + 20)
                context = content[start_pos:end_pos]

                details: dict[str, Any] = {
                    "pattern_name": pattern.name,
                    "category": pattern.category,
                    "match_start": match.start(),
                    "match_end": match.end(),
                }

                if pattern.remediation:
                    details["remediation"] = pattern.remediation

                findings.append(
                    SecurityFinding(
                        check_type="prompt_injection",
                        finding_type=pattern.category,
                        severity=pattern.severity,
                        message=pattern.description,
                        evidence_masked=mask_injection_evidence(context),
                        location=location,
                        details=details,
                    )
                )

                # Only report first match per pattern to avoid spam
                break

        return findings

    def get_patterns_by_category(self, category: str) -> list[InjectionPattern]:
        """Get all patterns for a specific category.

        Args:
            category: The category to filter by ('injection', 'jailbreak', 'role').

        Returns:
            List of patterns in the specified category.
        """
        return [p for p in self._all_patterns if p.category == category]

    def add_pattern(self, pattern: InjectionPattern) -> None:
        """Add a custom pattern to the checker.

        Args:
            pattern: The pattern to add.
        """
        self._all_patterns.append(pattern)
