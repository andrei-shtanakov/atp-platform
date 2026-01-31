"""Dashboard services layer.

This module provides service classes that encapsulate business logic
for the ATP Dashboard, separating it from route handlers.

Services are injected into routes via FastAPI dependency injection.
"""

from atp.dashboard.v2.services.agent_service import AgentService
from atp.dashboard.v2.services.comparison_service import ComparisonService
from atp.dashboard.v2.services.export_service import ExportService
from atp.dashboard.v2.services.test_service import TestService

__all__ = [
    "AgentService",
    "ComparisonService",
    "ExportService",
    "TestService",
]
