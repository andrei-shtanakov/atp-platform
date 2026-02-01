"""Code safety checker for detecting dangerous code patterns."""

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import (
    SecurityChecker,
    SecurityFinding,
    Severity,
    mask_sensitive_data,
)


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    UNKNOWN = "unknown"


class CodePattern(BaseModel):
    """A pattern for detecting dangerous code constructs."""

    name: str = Field(..., description="Name of the pattern")
    category: str = Field(
        ..., description="Category (dangerous_import, dangerous_function, etc.)"
    )
    pattern: Any = Field(..., description="Compiled regex pattern")
    severity: Severity = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description")
    languages: list[Language] = Field(
        ..., description="Languages this pattern applies to"
    )
    remediation: str | None = Field(None, description="Suggested fix")

    model_config = ConfigDict(arbitrary_types_allowed=True)


def mask_code_evidence(code: str, max_length: int = 80) -> str:
    """Mask code evidence for safe reporting.

    Args:
        code: The code snippet to mask.
        max_length: Maximum length of the output.

    Returns:
        Masked and truncated code suitable for reporting.
    """
    cleaned = code.strip()
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    return mask_sensitive_data(cleaned, visible_chars=15)


# Python dangerous imports
PYTHON_DANGEROUS_IMPORTS = [
    # System access
    ("os", Severity.HIGH, "Operating system access"),
    ("subprocess", Severity.CRITICAL, "Command execution"),
    ("shutil", Severity.HIGH, "File operations"),
    ("sys", Severity.MEDIUM, "System parameters and functions"),
    # Network access
    ("socket", Severity.HIGH, "Low-level network access"),
    ("http.client", Severity.MEDIUM, "HTTP client connections"),
    ("urllib", Severity.MEDIUM, "URL operations"),
    ("requests", Severity.MEDIUM, "HTTP requests"),
    ("httpx", Severity.MEDIUM, "HTTP requests"),
    ("aiohttp", Severity.MEDIUM, "Async HTTP requests"),
    ("ftplib", Severity.HIGH, "FTP operations"),
    ("smtplib", Severity.HIGH, "Email sending"),
    ("telnetlib", Severity.HIGH, "Telnet operations"),
    # Code execution
    ("ctypes", Severity.CRITICAL, "C library access"),
    ("importlib", Severity.HIGH, "Dynamic imports"),
    ("pickle", Severity.HIGH, "Object serialization (arbitrary code execution)"),
    ("marshal", Severity.HIGH, "Internal Python object serialization"),
    ("shelve", Severity.HIGH, "Object persistence with pickle"),
    # Process control
    ("multiprocessing", Severity.MEDIUM, "Process spawning"),
    ("threading", Severity.LOW, "Thread operations"),
    ("signal", Severity.MEDIUM, "Signal handling"),
    # Dangerous utilities
    ("tempfile", Severity.LOW, "Temporary file creation"),
    ("glob", Severity.LOW, "File pattern matching"),
    ("pathlib", Severity.LOW, "File path operations"),
    ("builtins", Severity.HIGH, "Access to built-in functions"),
    ("code", Severity.CRITICAL, "Interactive interpreter"),
    ("codeop", Severity.CRITICAL, "Compile Python code"),
    ("pty", Severity.CRITICAL, "Pseudo-terminal utilities"),
]

# Python dangerous functions
PYTHON_DANGEROUS_FUNCTIONS = [
    ("eval", Severity.CRITICAL, "Arbitrary code execution from string"),
    ("exec", Severity.CRITICAL, "Arbitrary code execution"),
    ("compile", Severity.HIGH, "Compile code objects"),
    ("__import__", Severity.HIGH, "Dynamic module import"),
    ("open", Severity.MEDIUM, "File operations"),
    ("input", Severity.LOW, "User input (potential injection)"),
    ("getattr", Severity.MEDIUM, "Dynamic attribute access"),
    ("setattr", Severity.MEDIUM, "Dynamic attribute modification"),
    ("delattr", Severity.MEDIUM, "Dynamic attribute deletion"),
    ("globals", Severity.HIGH, "Access to global namespace"),
    ("locals", Severity.MEDIUM, "Access to local namespace"),
    ("vars", Severity.MEDIUM, "Access to object attributes"),
    ("dir", Severity.LOW, "Object introspection"),
]

# JavaScript dangerous patterns
JAVASCRIPT_DANGEROUS_PATTERNS = [
    # Code execution
    (r"\beval\s*\(", Severity.CRITICAL, "Arbitrary code execution"),
    (r"\bFunction\s*\(", Severity.CRITICAL, "Dynamic function creation"),
    (r"\bsetTimeout\s*\([^,]*,", Severity.MEDIUM, "Delayed code execution"),
    (r"\bsetInterval\s*\([^,]*,", Severity.MEDIUM, "Repeated code execution"),
    # DOM manipulation (XSS vectors)
    (r"\.innerHTML\s*=", Severity.HIGH, "DOM manipulation (XSS risk)"),
    (r"\.outerHTML\s*=", Severity.HIGH, "DOM manipulation (XSS risk)"),
    (r"document\.write\s*\(", Severity.HIGH, "Document write (XSS risk)"),
    (r"document\.writeln\s*\(", Severity.HIGH, "Document write (XSS risk)"),
    (r"\.insertAdjacentHTML\s*\(", Severity.HIGH, "HTML insertion (XSS risk)"),
    # Network operations
    (r"\bfetch\s*\(", Severity.MEDIUM, "Network request"),
    (r"\bXMLHttpRequest\b", Severity.MEDIUM, "HTTP request"),
    (r"\bWebSocket\s*\(", Severity.MEDIUM, "WebSocket connection"),
    (r"\.ajax\s*\(", Severity.MEDIUM, "AJAX request"),
    # File system (Node.js)
    (r"\brequire\s*\(\s*['\"]fs['\"]", Severity.HIGH, "File system access"),
    (r"\brequire\s*\(\s*['\"]child_process['\"]", Severity.CRITICAL, "Process spawn"),
    (r"\brequire\s*\(\s*['\"]net['\"]", Severity.HIGH, "Network access"),
    (r"\brequire\s*\(\s*['\"]http['\"]", Severity.MEDIUM, "HTTP server/client"),
    (r"\brequire\s*\(\s*['\"]https['\"]", Severity.MEDIUM, "HTTPS server/client"),
    # ES6 imports
    (r"\bimport\s+.*\bfrom\s+['\"]fs['\"]", Severity.HIGH, "File system import"),
    (
        r"\bimport\s+.*\bfrom\s+['\"]child_process['\"]",
        Severity.CRITICAL,
        "Process spawn import",
    ),
    (r"\bimport\s+.*\bfrom\s+['\"]net['\"]", Severity.HIGH, "Network import"),
    # Process and environment
    (r"\bprocess\.env\b", Severity.MEDIUM, "Environment variable access"),
    (r"\bprocess\.exit\s*\(", Severity.HIGH, "Process termination"),
    (r"\brequire\s*\(\s*['\"]vm['\"]", Severity.CRITICAL, "VM module (code execution)"),
]

# Bash dangerous patterns
BASH_DANGEROUS_PATTERNS = [
    # Command execution
    (r"\beval\s+", Severity.CRITICAL, "Arbitrary command execution"),
    (r"`[^`]+`", Severity.HIGH, "Command substitution"),
    (r"\$\([^)]+\)", Severity.HIGH, "Command substitution"),
    (r"\bexec\s+", Severity.CRITICAL, "Process replacement"),
    (r"\bsource\s+", Severity.HIGH, "Script sourcing"),
    (r"^\s*\.\s+/", Severity.HIGH, "Script sourcing (dot command)"),
    # Dangerous commands
    (r"\brm\s+-rf\s+", Severity.CRITICAL, "Recursive forced deletion"),
    (r"\brm\s+.*\*", Severity.HIGH, "Wildcard deletion"),
    (r"\bchmod\s+777\b", Severity.HIGH, "Permissive file permissions"),
    (r"\bchown\s+", Severity.MEDIUM, "File ownership change"),
    (r"\bsudo\s+", Severity.HIGH, "Privileged execution"),
    (r"\bsu\s+", Severity.HIGH, "User switching"),
    (r"\bmkdir\s+-p\s+", Severity.LOW, "Directory creation"),
    # Network operations
    (r"\bcurl\s+", Severity.MEDIUM, "HTTP request"),
    (r"\bwget\s+", Severity.MEDIUM, "File download"),
    (r"\bnc\s+", Severity.HIGH, "Netcat (network utility)"),
    (r"\bnetcat\s+", Severity.HIGH, "Netcat (network utility)"),
    (r"\bssh\s+", Severity.MEDIUM, "SSH connection"),
    (r"\bscp\s+", Severity.MEDIUM, "Secure file copy"),
    (r"\brsync\s+", Severity.MEDIUM, "Remote sync"),
    # File operations
    (r"\bcat\s+.*>", Severity.MEDIUM, "File overwrite"),
    (r"\becho\s+.*>>", Severity.LOW, "File append"),
    (r"\bdd\s+", Severity.HIGH, "Low-level data copy"),
    (r"\bmkfs\b", Severity.CRITICAL, "Filesystem creation"),
    (r"\bfdisk\b", Severity.CRITICAL, "Disk partitioning"),
    # Environment manipulation
    (r"\bexport\s+", Severity.LOW, "Environment variable export"),
    (r"\bunset\s+", Severity.LOW, "Variable unsetting"),
    # Dangerous redirections
    (r">\s*/dev/sd[a-z]", Severity.CRITICAL, "Direct disk write"),
    (r">\s*/dev/null\s+2>&1", Severity.LOW, "Output suppression"),
    # Pipes to dangerous commands
    (r"\|\s*sh\b", Severity.CRITICAL, "Piped shell execution"),
    (r"\|\s*bash\b", Severity.CRITICAL, "Piped bash execution"),
    (r"\|\s*sudo\s+", Severity.CRITICAL, "Piped sudo execution"),
]

# File system operation patterns (language-agnostic where possible)
FILE_OPERATION_PATTERNS = [
    # Read operations
    (r"\bread\s*\(", ["python", "javascript"], Severity.LOW, "File read"),
    (r"\.read\s*\(", ["python"], Severity.LOW, "File read method"),
    (r"\.readlines\s*\(", ["python"], Severity.LOW, "Read file lines"),
    (r"\.readline\s*\(", ["python"], Severity.LOW, "Read file line"),
    (r"readFile(?:Sync)?\s*\(", ["javascript"], Severity.MEDIUM, "File read"),
    # Write operations
    (r"\.write\s*\(", ["python", "javascript"], Severity.MEDIUM, "File write"),
    (r"\.writelines\s*\(", ["python"], Severity.MEDIUM, "Write file lines"),
    (r"writeFile(?:Sync)?\s*\(", ["javascript"], Severity.MEDIUM, "File write"),
    (r"appendFile(?:Sync)?\s*\(", ["javascript"], Severity.MEDIUM, "File append"),
    # Delete operations
    (r"\.unlink\s*\(", ["python", "javascript"], Severity.HIGH, "File deletion"),
    (r"\.remove\s*\(", ["python"], Severity.HIGH, "File deletion"),
    (r"\.rmdir\s*\(", ["python", "javascript"], Severity.HIGH, "Directory deletion"),
    (r"\.rmtree\s*\(", ["python"], Severity.CRITICAL, "Recursive deletion"),
    (r"unlinkSync\s*\(", ["javascript"], Severity.HIGH, "Sync file deletion"),
    (r"rmdirSync\s*\(", ["javascript"], Severity.HIGH, "Sync directory deletion"),
    # Path traversal patterns
    (r"\.\./", ["python", "javascript", "bash"], Severity.HIGH, "Path traversal"),
    (r"\\.\\.", ["python", "javascript", "bash"], Severity.HIGH, "Path traversal"),
]

# Network operation patterns
NETWORK_OPERATION_PATTERNS = [
    # Socket operations
    (
        r"\.connect\s*\(",
        ["python", "javascript"],
        Severity.MEDIUM,
        "Network connection",
    ),
    (r"\.bind\s*\(", ["python", "javascript"], Severity.MEDIUM, "Network binding"),
    (r"\.listen\s*\(", ["python", "javascript"], Severity.MEDIUM, "Network listening"),
    (r"\.accept\s*\(", ["python"], Severity.MEDIUM, "Connection accept"),
    # HTTP operations
    (
        r"\.get\s*\(['\"]https?://",
        ["python", "javascript"],
        Severity.MEDIUM,
        "HTTP GET request",
    ),
    (
        r"\.post\s*\(['\"]https?://",
        ["python", "javascript"],
        Severity.MEDIUM,
        "HTTP POST request",
    ),
    (
        r"requests\.(get|post|put|delete|patch)\s*\(",
        ["python"],
        Severity.MEDIUM,
        "HTTP request",
    ),
    (r"urllib\.request\.urlopen\s*\(", ["python"], Severity.MEDIUM, "URL open"),
    # DNS operations
    (r"\.gethostbyname\s*\(", ["python"], Severity.LOW, "DNS lookup"),
    (r"\.getaddrinfo\s*\(", ["python"], Severity.LOW, "Address info lookup"),
    (r"dns\.lookup\s*\(", ["javascript"], Severity.LOW, "DNS lookup"),
]


def _build_python_import_patterns() -> list[CodePattern]:
    """Build regex patterns for Python dangerous imports."""
    patterns = []
    for module, severity, desc in PYTHON_DANGEROUS_IMPORTS:
        # Match various import styles
        # import os, from os import, __import__('os')
        regex = re.compile(
            rf"(?:^\s*import\s+{re.escape(module)}\b|"
            rf"^\s*from\s+{re.escape(module)}\b|"
            rf"__import__\s*\(\s*['\"]({re.escape(module)})['\"])",
            re.MULTILINE,
        )
        patterns.append(
            CodePattern(
                name=f"python_import_{module.replace('.', '_')}",
                category="dangerous_import",
                pattern=regex,
                severity=severity,
                description=f"Dangerous import: {module} - {desc}",
                languages=[Language.PYTHON],
                remediation=f"Avoid importing {module} unless absolutely necessary",
            )
        )
    return patterns


def _build_python_function_patterns() -> list[CodePattern]:
    """Build regex patterns for Python dangerous functions."""
    patterns = []
    for func, severity, desc in PYTHON_DANGEROUS_FUNCTIONS:
        # Match function calls
        regex = re.compile(rf"\b{re.escape(func)}\s*\(", re.MULTILINE)
        patterns.append(
            CodePattern(
                name=f"python_func_{func}",
                category="dangerous_function",
                pattern=regex,
                severity=severity,
                description=f"Dangerous function: {func}() - {desc}",
                languages=[Language.PYTHON],
                remediation=f"Avoid using {func}() - security risk",
            )
        )
    return patterns


def _build_javascript_patterns() -> list[CodePattern]:
    """Build regex patterns for JavaScript dangerous code."""
    patterns = []
    for i, (regex_str, severity, desc) in enumerate(JAVASCRIPT_DANGEROUS_PATTERNS):
        regex = re.compile(regex_str, re.MULTILINE | re.IGNORECASE)
        # Create name from regex pattern
        name = f"javascript_pattern_{i}"
        if "eval" in regex_str:
            name = "javascript_eval"
        elif "Function" in regex_str:
            name = "javascript_function_constructor"
        elif "innerHTML" in regex_str:
            name = "javascript_innerhtml"
        elif "fetch" in regex_str:
            name = "javascript_fetch"
        elif "child_process" in regex_str:
            name = "javascript_child_process"
        elif "fs" in regex_str:
            name = "javascript_fs"
        elif "WebSocket" in regex_str:
            name = "javascript_websocket"
        elif "process.exit" in regex_str:
            name = "javascript_process_exit"
        elif "vm" in regex_str:
            name = "javascript_vm"
        patterns.append(
            CodePattern(
                name=name,
                category="dangerous_function"
                if "eval" in regex_str or "Function" in regex_str
                else "dangerous_import"
                if "require" in regex_str or "import" in regex_str
                else "network_operation"
                if any(x in regex_str for x in ["fetch", "WebSocket", "ajax", "http"])
                else "dangerous_function",
                pattern=regex,
                severity=severity,
                description=f"JavaScript: {desc}",
                languages=[Language.JAVASCRIPT],
                remediation="Review and sanitize this code pattern",
            )
        )
    return patterns


def _build_bash_patterns() -> list[CodePattern]:
    """Build regex patterns for Bash dangerous code."""
    patterns = []
    for i, (regex_str, severity, desc) in enumerate(BASH_DANGEROUS_PATTERNS):
        regex = re.compile(regex_str, re.MULTILINE)
        # Create descriptive name from pattern
        name = f"bash_pattern_{i}"
        if "eval" in regex_str:
            name = "bash_eval"
        elif "exec" in regex_str:
            name = "bash_exec"
        elif "rm -rf" in regex_str:
            name = "bash_rm_rf"
        elif "sudo" in regex_str:
            name = "bash_sudo"
        elif "curl" in regex_str:
            name = "bash_curl"
        elif "wget" in regex_str:
            name = "bash_wget"
        elif "nc" in regex_str or "netcat" in regex_str:
            name = "bash_netcat"
        elif "dd" in regex_str:
            name = "bash_dd"
        patterns.append(
            CodePattern(
                name=name,
                category="dangerous_function"
                if "eval" in regex_str or "exec" in regex_str
                else "file_operation"
                if any(
                    x in regex_str
                    for x in ["rm", "chmod", "chown", "cat", "dd", "mkfs"]
                )
                else "network_operation"
                if any(x in regex_str for x in ["curl", "wget", "nc", "ssh", "scp"])
                else "dangerous_function",
                pattern=regex,
                severity=severity,
                description=f"Bash: {desc}",
                languages=[Language.BASH],
                remediation="Review this shell command for security implications",
            )
        )
    return patterns


def _build_file_operation_patterns() -> list[CodePattern]:
    """Build regex patterns for file system operations."""
    patterns = []
    for i, item in enumerate(FILE_OPERATION_PATTERNS):
        regex_str, langs, severity, desc = item
        regex = re.compile(regex_str, re.MULTILINE)
        lang_enums = [Language(lang) for lang in langs]
        patterns.append(
            CodePattern(
                name=f"file_op_{i}",
                category="file_operation",
                pattern=regex,
                severity=severity,
                description=f"File operation: {desc}",
                languages=lang_enums,
                remediation="Ensure proper input validation for file paths",
            )
        )
    return patterns


def _build_network_operation_patterns() -> list[CodePattern]:
    """Build regex patterns for network operations."""
    patterns = []
    for i, item in enumerate(NETWORK_OPERATION_PATTERNS):
        regex_str, langs, severity, desc = item
        regex = re.compile(regex_str, re.MULTILINE)
        lang_enums = [Language(lang) for lang in langs]
        patterns.append(
            CodePattern(
                name=f"network_op_{i}",
                category="network_operation",
                pattern=regex,
                severity=severity,
                description=f"Network operation: {desc}",
                languages=lang_enums,
                remediation="Validate network destinations and access controls",
            )
        )
    return patterns


class CodeSafetyChecker(SecurityChecker):
    """Checker for dangerous code patterns.

    Detects the following types of dangerous code:
    - Dangerous imports (os, subprocess, socket, etc.)
    - Dangerous functions (eval, exec, compile, etc.)
    - File system operations (read, write, delete)
    - Network operations (connect, bind, HTTP requests)

    Supports multiple languages:
    - Python
    - JavaScript
    - Bash/Shell
    """

    def __init__(
        self,
        include_categories: list[str] | None = None,
        include_languages: list[str] | None = None,
        custom_patterns: list[CodePattern] | None = None,
    ) -> None:
        """Initialize the code safety checker.

        Args:
            include_categories: Optional list of categories to check.
                Valid values: 'dangerous_import', 'dangerous_function',
                'file_operation', 'network_operation'.
                If None, all categories are checked.
            include_languages: Optional list of languages to check.
                Valid values: 'python', 'javascript', 'bash'.
                If None, all languages are checked.
            custom_patterns: Optional list of custom patterns to add.
        """
        self._all_patterns: list[CodePattern] = []

        # Build all patterns
        all_builtin = (
            _build_python_import_patterns()
            + _build_python_function_patterns()
            + _build_javascript_patterns()
            + _build_bash_patterns()
            + _build_file_operation_patterns()
            + _build_network_operation_patterns()
        )

        # Filter by categories if specified
        if include_categories:
            all_builtin = [p for p in all_builtin if p.category in include_categories]

        # Filter by languages if specified
        if include_languages:
            lang_enums = {Language(lang) for lang in include_languages}
            all_builtin = [
                p
                for p in all_builtin
                if any(lang in lang_enums for lang in p.languages)
            ]

        self._all_patterns = all_builtin

        # Add custom patterns if provided
        if custom_patterns:
            self._all_patterns.extend(custom_patterns)

    @property
    def name(self) -> str:
        """Return the checker name."""
        return "code_safety"

    @property
    def check_types(self) -> list[str]:
        """Return the list of check types this checker supports."""
        return [
            "dangerous_import",
            "dangerous_function",
            "file_operation",
            "network_operation",
        ]

    def detect_language(self, content: str) -> Language:
        """Detect the programming language of the code content.

        Args:
            content: The code content to analyze.

        Returns:
            Detected Language enum value.
        """
        # Check for Python indicators
        python_indicators = [
            r"^import\s+\w+",
            r"^from\s+\w+\s+import",
            r"^def\s+\w+\s*\(",
            r"^class\s+\w+",
            r":\s*$",
            r"^\s+(?:if|for|while|def|class|try|except|with)\s+",
            r"^#!/.*python",
        ]
        python_score = sum(
            1
            for pattern in python_indicators
            if re.search(pattern, content, re.MULTILINE)
        )

        # Check for JavaScript indicators
        js_indicators = [
            r"\bfunction\s+\w+\s*\(",
            r"\bconst\s+\w+\s*=",
            r"\blet\s+\w+\s*=",
            r"\bvar\s+\w+\s*=",
            r"=>\s*\{",
            r"\brequire\s*\(",
            r"\bmodule\.exports\b",
            r"\bexport\s+(?:default|const|function|class)",
            r"^#!/.*node",
        ]
        js_score = sum(
            1 for pattern in js_indicators if re.search(pattern, content, re.MULTILINE)
        )

        # Check for Bash indicators
        bash_indicators = [
            r"^#!/.*(?:ba)?sh",
            r"^\s*(?:if|then|else|elif|fi|for|do|done|while|case|esac)\b",
            r"\$\{?\w+\}?",
            r"\|\s*(?:grep|awk|sed|cut|sort)",
            r"^export\s+\w+=",
            r"&&\s*$",
            r"\[\[\s+.*\s+\]\]",
        ]
        bash_score = sum(
            1
            for pattern in bash_indicators
            if re.search(pattern, content, re.MULTILINE)
        )

        # Return language with highest score
        if python_score > js_score and python_score > bash_score:
            return Language.PYTHON
        elif js_score > python_score and js_score > bash_score:
            return Language.JAVASCRIPT
        elif bash_score > 0:
            return Language.BASH
        else:
            return Language.UNKNOWN

    def check(
        self,
        content: str,
        location: str | None = None,
        enabled_types: list[str] | None = None,
    ) -> list[SecurityFinding]:
        """Check code content for dangerous patterns.

        Args:
            content: The code content to scan.
            location: Optional location identifier (e.g., artifact path).
            enabled_types: Optional list of specific check types to run.
                Valid values: 'dangerous_import', 'dangerous_function',
                'file_operation', 'network_operation'.
                If None, all types are checked.

        Returns:
            List of SecurityFinding objects for any dangerous code found.
        """
        findings: list[SecurityFinding] = []
        seen_patterns: set[str] = set()

        # Detect language for filtering
        detected_lang = self.detect_language(content)

        for pattern in self._all_patterns:
            # Filter by enabled types
            if enabled_types and pattern.category not in enabled_types:
                continue

            # Filter by detected language (always include UNKNOWN patterns)
            if (
                detected_lang != Language.UNKNOWN
                and detected_lang not in pattern.languages
            ):
                continue

            # Skip if we already found this pattern
            if pattern.name in seen_patterns:
                continue

            for match in pattern.pattern.finditer(content):
                # Mark pattern as seen
                seen_patterns.add(pattern.name)

                # Get context around the match
                start_pos = max(0, match.start() - 20)
                end_pos = min(len(content), match.end() + 40)
                context = content[start_pos:end_pos]

                # Get line number
                line_num = content[: match.start()].count("\n") + 1

                details: dict[str, Any] = {
                    "pattern_name": pattern.name,
                    "category": pattern.category,
                    "match_start": match.start(),
                    "match_end": match.end(),
                    "line_number": line_num,
                    "language": detected_lang.value,
                    "matched_text": match.group(0)[:50],
                }

                if pattern.remediation:
                    details["remediation"] = pattern.remediation

                findings.append(
                    SecurityFinding(
                        check_type="code_safety",
                        finding_type=pattern.category,
                        severity=pattern.severity,
                        message=pattern.description,
                        evidence_masked=mask_code_evidence(context),
                        location=location,
                        details=details,
                    )
                )

                # Only report first match per pattern
                break

        return findings

    def get_patterns_by_category(self, category: str) -> list[CodePattern]:
        """Get all patterns for a specific category.

        Args:
            category: The category to filter by.

        Returns:
            List of patterns in the specified category.
        """
        return [p for p in self._all_patterns if p.category == category]

    def get_patterns_by_language(self, language: str) -> list[CodePattern]:
        """Get all patterns for a specific language.

        Args:
            language: The language to filter by ('python', 'javascript', 'bash').

        Returns:
            List of patterns for the specified language.
        """
        try:
            lang_enum = Language(language)
            return [p for p in self._all_patterns if lang_enum in p.languages]
        except ValueError:
            return []

    def add_pattern(self, pattern: CodePattern) -> None:
        """Add a custom pattern to the checker.

        Args:
            pattern: The pattern to add.
        """
        self._all_patterns.append(pattern)
