"""Dashboard services layer.

This module provides service classes that encapsulate business logic
for the ATP Dashboard, separating it from route handlers.

Services are injected into routes via FastAPI dependency injection.
"""

from atp.dashboard.v2.services.agent_service import AgentService
from atp.dashboard.v2.services.comparison_service import ComparisonService
from atp.dashboard.v2.services.export_service import ExportService
from atp.dashboard.v2.services.github_import import (
    GitHubFile,
    ImportResult,
    ParsedGitHubURL,
    import_from_github,
    parse_github_url,
    validate_suite_yaml,
)
from atp.dashboard.v2.services.test_service import TestService

__all__ = [
    "AgentService",
    "ComparisonService",
    "ExportService",
    "GitHubFile",
    "ImportResult",
    "ParsedGitHubURL",
    "TestService",
    "import_from_github",
    "parse_github_url",
    "validate_suite_yaml",
]
