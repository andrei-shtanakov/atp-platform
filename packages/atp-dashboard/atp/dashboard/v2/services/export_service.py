"""Export service for data export operations.

This module provides the ExportService class that encapsulates all
business logic related to exporting data in various formats.
"""

import csv
import io
import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from atp.dashboard.models import (
    SuiteDefinition,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.schemas import (
    AgentConfigCreate,
    AssertionCreate,
    ConstraintsCreate,
    ScoringWeightsCreate,
    SuiteDefinitionResponse,
    TaskCreate,
    TestDefaultsCreate,
    TestResponse,
    YAMLExportResponse,
)


class ExportService:
    """Service for export operations.

    This service encapsulates all business logic related to
    exporting data in various formats (CSV, JSON, YAML).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the export service.

        Args:
            session: Database session for queries.
        """
        self._session = session

    async def export_suite_to_yaml(self, suite_id: int) -> YAMLExportResponse | None:
        """Export a suite definition to YAML format.

        Args:
            suite_id: Suite definition ID.

        Returns:
            YAMLExportResponse with YAML content, or None if not found
            or suite has no tests.
        """
        suite_def = await self._session.get(SuiteDefinition, suite_id)
        if suite_def is None:
            return None

        if not suite_def.tests_json:
            return None

        # Import lazily to avoid circular dependencies
        from atp.generator.core import TestSuiteData
        from atp.generator.writer import YAMLWriter
        from atp.loader.models import (
            AgentConfig,
            Assertion,
            Constraints,
            ScoringWeights,
            TaskDefinition,
            TestDefaults,
            TestDefinition,
        )

        # Build TestDefaults
        defaults_data = suite_def.defaults_json
        defaults = TestDefaults(
            runs_per_test=defaults_data.get("runs_per_test", 1),
            timeout_seconds=defaults_data.get("timeout_seconds", 300),
            scoring=ScoringWeights(**defaults_data.get("scoring", {})),
            constraints=(
                Constraints(**defaults_data.get("constraints", {}))
                if defaults_data.get("constraints")
                else None
            ),
        )

        # Build agents
        agents = [
            AgentConfig(
                name=a["name"],
                type=a.get("type"),
                config=a.get("config", {}),
            )
            for a in suite_def.agents_json
        ]

        # Build tests
        tests = []
        for t in suite_def.tests_json:
            task_data = t["task"]
            tests.append(
                TestDefinition(
                    id=t["id"],
                    name=t["name"],
                    description=t.get("description"),
                    tags=t.get("tags", []),
                    task=TaskDefinition(
                        description=task_data["description"],
                        input_data=task_data.get("input_data"),
                        expected_artifacts=task_data.get("expected_artifacts"),
                    ),
                    constraints=Constraints(**t.get("constraints", {})),
                    assertions=[
                        Assertion(type=a["type"], config=a.get("config", {}))
                        for a in t.get("assertions", [])
                    ],
                    scoring=(
                        ScoringWeights(**t["scoring"]) if t.get("scoring") else None
                    ),
                )
            )

        # Create suite data
        suite_data = TestSuiteData(
            name=suite_def.name,
            version=suite_def.version,
            description=suite_def.description,
            defaults=defaults,
            agents=agents,
            tests=tests,
        )

        # Generate YAML
        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite_data)

        return YAMLExportResponse(
            yaml_content=yaml_content,
            suite_name=suite_def.name,
            test_count=len(suite_def.tests_json),
        )

    async def export_results_to_csv(
        self,
        suite_name: str | None = None,
        agent_id: int | None = None,
        limit: int = 1000,
    ) -> str:
        """Export suite execution results to CSV format.

        Args:
            suite_name: Optional filter by suite name.
            agent_id: Optional filter by agent ID.
            limit: Maximum number of records to export.

        Returns:
            CSV string with execution results.
        """
        stmt = (
            select(SuiteExecution)
            .options(
                selectinload(SuiteExecution.agent),
                selectinload(SuiteExecution.test_executions),
            )
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit)
        )

        if suite_name:
            stmt = stmt.where(SuiteExecution.suite_name == suite_name)
        if agent_id:
            stmt = stmt.where(SuiteExecution.agent_id == agent_id)

        result = await self._session.execute(stmt)
        executions = list(result.scalars().all())

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "suite_name",
                "agent_name",
                "started_at",
                "completed_at",
                "duration_seconds",
                "total_tests",
                "passed_tests",
                "failed_tests",
                "success_rate",
                "status",
            ]
        )

        # Write data
        for exec in executions:
            agent_name = exec.agent.name if exec.agent else ""
            writer.writerow(
                [
                    exec.suite_name,
                    agent_name,
                    exec.started_at.isoformat() if exec.started_at else "",
                    exec.completed_at.isoformat() if exec.completed_at else "",
                    exec.duration_seconds or "",
                    exec.total_tests,
                    exec.passed_tests,
                    exec.failed_tests,
                    exec.success_rate,
                    exec.status,
                ]
            )

        return output.getvalue()

    async def export_test_results_to_csv(
        self,
        suite_execution_id: int | None = None,
        test_id: str | None = None,
        limit: int = 1000,
    ) -> str:
        """Export test execution results to CSV format.

        Args:
            suite_execution_id: Optional filter by suite execution.
            test_id: Optional filter by test ID.
            limit: Maximum number of records to export.

        Returns:
            CSV string with test execution results.
        """
        stmt = (
            select(TestExecution)
            .options(selectinload(TestExecution.suite_execution))
            .order_by(TestExecution.started_at.desc())
            .limit(limit)
        )

        if suite_execution_id:
            stmt = stmt.where(TestExecution.suite_execution_id == suite_execution_id)
        if test_id:
            stmt = stmt.where(TestExecution.test_id == test_id)

        result = await self._session.execute(stmt)
        executions = list(result.scalars().all())

        # Build CSV
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(
            [
                "suite_execution_id",
                "test_id",
                "test_name",
                "tags",
                "started_at",
                "completed_at",
                "duration_seconds",
                "total_runs",
                "successful_runs",
                "success",
                "score",
                "status",
            ]
        )

        # Write data
        for exec in executions:
            writer.writerow(
                [
                    exec.suite_execution_id,
                    exec.test_id,
                    exec.test_name,
                    ",".join(exec.tags) if exec.tags else "",
                    exec.started_at.isoformat() if exec.started_at else "",
                    exec.completed_at.isoformat() if exec.completed_at else "",
                    exec.duration_seconds or "",
                    exec.total_runs,
                    exec.successful_runs,
                    exec.success,
                    exec.score or "",
                    exec.status,
                ]
            )

        return output.getvalue()

    async def export_results_to_json(
        self,
        suite_name: str | None = None,
        agent_id: int | None = None,
        limit: int = 1000,
        include_tests: bool = True,
    ) -> str:
        """Export suite execution results to JSON format.

        Args:
            suite_name: Optional filter by suite name.
            agent_id: Optional filter by agent ID.
            limit: Maximum number of records to export.
            include_tests: Whether to include test execution details.

        Returns:
            JSON string with execution results.
        """
        stmt = (
            select(SuiteExecution)
            .options(selectinload(SuiteExecution.agent))
            .order_by(SuiteExecution.started_at.desc())
            .limit(limit)
        )

        if include_tests:
            stmt = stmt.options(selectinload(SuiteExecution.test_executions))

        if suite_name:
            stmt = stmt.where(SuiteExecution.suite_name == suite_name)
        if agent_id:
            stmt = stmt.where(SuiteExecution.agent_id == agent_id)

        result = await self._session.execute(stmt)
        executions = list(result.scalars().all())

        # Build JSON structure
        data: list[dict[str, Any]] = []
        for exec in executions:
            exec_data: dict[str, Any] = {
                "id": exec.id,
                "suite_name": exec.suite_name,
                "agent": {
                    "id": exec.agent.id,
                    "name": exec.agent.name,
                    "type": exec.agent.agent_type,
                }
                if exec.agent
                else None,
                "started_at": exec.started_at.isoformat() if exec.started_at else None,
                "completed_at": (
                    exec.completed_at.isoformat() if exec.completed_at else None
                ),
                "duration_seconds": exec.duration_seconds,
                "total_tests": exec.total_tests,
                "passed_tests": exec.passed_tests,
                "failed_tests": exec.failed_tests,
                "success_rate": exec.success_rate,
                "status": exec.status,
            }

            if include_tests and exec.test_executions:
                exec_data["tests"] = [
                    {
                        "id": t.id,
                        "test_id": t.test_id,
                        "test_name": t.test_name,
                        "tags": t.tags,
                        "success": t.success,
                        "score": t.score,
                        "duration_seconds": t.duration_seconds,
                        "status": t.status,
                    }
                    for t in exec.test_executions
                ]

            data.append(exec_data)

        return json.dumps({"executions": data}, indent=2)

    async def get_suite_definition(
        self, suite_id: int
    ) -> SuiteDefinitionResponse | None:
        """Get a suite definition by ID.

        Args:
            suite_id: Suite definition ID.

        Returns:
            Suite definition response or None if not found.
        """
        suite_def = await self._session.get(SuiteDefinition, suite_id)
        if suite_def is None:
            return None

        return self._build_suite_definition_response(suite_def)

    async def get_suite_definition_by_name(
        self, name: str
    ) -> SuiteDefinitionResponse | None:
        """Get a suite definition by name.

        Args:
            name: Suite name.

        Returns:
            Suite definition response or None if not found.
        """
        stmt = select(SuiteDefinition).where(SuiteDefinition.name == name)
        result = await self._session.execute(stmt)
        suite_def = result.scalar_one_or_none()
        if suite_def is None:
            return None

        return self._build_suite_definition_response(suite_def)

    def _build_suite_definition_response(
        self, suite_def: SuiteDefinition
    ) -> SuiteDefinitionResponse:
        """Build a SuiteDefinitionResponse from a SuiteDefinition model.

        Args:
            suite_def: The database model.

        Returns:
            The response schema.
        """
        # Convert JSON fields back to Pydantic models
        defaults = TestDefaultsCreate(**suite_def.defaults_json)
        agents = [AgentConfigCreate(**a) for a in suite_def.agents_json]
        tests = [
            TestResponse(
                id=t["id"],
                name=t["name"],
                description=t.get("description"),
                tags=t.get("tags", []),
                task=TaskCreate(**t["task"]),
                constraints=ConstraintsCreate(**t.get("constraints", {})),
                assertions=[AssertionCreate(**a) for a in t.get("assertions", [])],
                scoring=(
                    ScoringWeightsCreate(**t["scoring"]) if t.get("scoring") else None
                ),
            )
            for t in suite_def.tests_json
        ]

        return SuiteDefinitionResponse(
            id=suite_def.id,
            name=suite_def.name,
            version=suite_def.version,
            description=suite_def.description,
            defaults=defaults,
            agents=agents,
            tests=tests,
            created_at=suite_def.created_at,
            updated_at=suite_def.updated_at,
        )
