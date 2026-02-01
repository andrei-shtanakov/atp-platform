"""Integration tests for the analytics dashboard routes.

These tests verify the advanced analytics API endpoints including
trends, anomalies, correlations, and scheduled reports.
"""

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from atp.analytics.database import AnalyticsDatabase
from atp.analytics.models import AnalyticsBase, ScheduledReport
from atp.dashboard.database import Database
from atp.dashboard.models import (
    Agent,
    RunResult,
    SuiteExecution,
    TestExecution,
)
from atp.dashboard.v2.dependencies import get_db_session
from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
async def analytics_engine():
    """Create an async SQLite in-memory engine for analytics models."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(AnalyticsBase.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def analytics_session(
    analytics_engine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create an async session for analytics testing."""
    async_session_factory = sessionmaker(
        analytics_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def v2_app(test_database: Database, analytics_engine, monkeypatch):
    """Create a test app with v2 routes and analytics DB override."""
    app = create_test_app(use_v2_routes=True)

    # Override the dependency to use our test session
    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_database.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_session

    # Monkey-patch the AnalyticsDatabase to use our test engine
    original_init = AnalyticsDatabase.__init__

    def mock_init(self, database_url: str | None = None):
        original_init(self, database_url)
        self._engine = analytics_engine

    monkeypatch.setattr(AnalyticsDatabase, "__init__", mock_init)

    return app


@pytest.fixture
async def test_data(test_database: Database) -> dict:
    """Create test data for analytics testing."""
    async with test_database.session_factory() as session:
        now = datetime.now()

        # Create agent
        agent = Agent(
            name="test-agent",
            agent_type="cli",
            config={},
        )
        session.add(agent)
        await session.flush()

        # Create suite executions with test executions
        suite_executions = []
        for i in range(10):
            suite = SuiteExecution(
                suite_name="test-suite",
                agent_id=agent.id,
                started_at=now - timedelta(days=10 - i),
                completed_at=now - timedelta(days=10 - i) + timedelta(hours=1),
                status="completed",
                runs_per_test=1,
                total_tests=2,
                passed_tests=1 if i % 2 == 0 else 2,
                failed_tests=1 if i % 2 == 0 else 0,
            )
            session.add(suite)
            await session.flush()
            suite_executions.append(suite)

            # Create test executions with varying scores
            for j in range(2):
                # Create trend: scores improving over time
                base_score = 0.5 + (i * 0.03) + (j * 0.1)
                score = min(1.0, base_score)

                test_exec = TestExecution(
                    suite_execution_id=suite.id,
                    test_id=f"test-{j + 1}",
                    test_name=f"Test {j + 1}",
                    tags=["unit"],
                    started_at=suite.started_at + timedelta(minutes=j * 10),
                    completed_at=suite.started_at + timedelta(minutes=(j + 1) * 10),
                    status="completed",
                    total_runs=1,
                    successful_runs=1 if score > 0.6 else 0,
                    success=score > 0.6,
                    score=score,
                )
                session.add(test_exec)
                await session.flush()

                # Create run result
                run = RunResult(
                    test_execution_id=test_exec.id,
                    run_number=1,
                    started_at=test_exec.started_at,
                    completed_at=test_exec.completed_at,
                    response_status="completed",
                    success=test_exec.success,
                    input_tokens=1000 + i * 100,
                    output_tokens=500 + i * 50,
                    total_tokens=1500 + i * 150,
                    total_steps=5,
                    tool_calls=3 + (i % 3),
                    llm_calls=2 + (i % 2),
                    cost_usd=Decimal("0.01") + Decimal(str(i * 0.001)),
                )
                session.add(run)

        await session.commit()

        return {
            "agent": agent,
            "suite_executions": suite_executions,
        }


@pytest.fixture
async def scheduled_reports_data(analytics_engine) -> dict:
    """Create test data for scheduled reports."""
    async_session_factory = sessionmaker(
        analytics_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_factory() as session:
        reports = [
            ScheduledReport(
                name="Daily Report",
                frequency="daily",
                recipients_json=["user@example.com"],
                include_trends=True,
                include_anomalies=True,
                include_correlations=False,
                is_active=True,
                next_run=datetime.now() + timedelta(days=1),
            ),
            ScheduledReport(
                name="Weekly Report",
                frequency="weekly",
                recipients_json=["admin@example.com"],
                include_trends=True,
                include_anomalies=True,
                include_correlations=True,
                is_active=True,
                next_run=datetime.now() + timedelta(days=7),
            ),
            ScheduledReport(
                name="Inactive Report",
                frequency="monthly",
                is_active=False,
            ),
        ]
        session.add_all(reports)
        await session.commit()

        return {"reports": reports}


class TestTrendsRoutes:
    """Test trend analysis routes."""

    @pytest.mark.anyio
    async def test_get_score_trends(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/trends returns trend analysis."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/analytics/trends", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert "trends" in data
            assert "period_start" in data
            assert "period_end" in data

    @pytest.mark.anyio
    async def test_get_score_trends_with_filters(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/trends with filters."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/trends",
                params={
                    "suite_name": "test-suite",
                    "metrics": "score,success_rate",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()

            assert "trends" in data
            assert data["suite_name"] == "test-suite"

    @pytest.mark.anyio
    async def test_get_score_trends_empty(self, v2_app, auth_headers):
        """Test GET /api/analytics/trends with no data."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/trends",
                params={"suite_name": "nonexistent-suite"},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()

            assert data["trends"] == [] or all(
                t["direction"] == "insufficient_data" for t in data["trends"]
            )


class TestAnomaliesRoutes:
    """Test anomaly detection routes."""

    @pytest.mark.anyio
    async def test_detect_anomalies(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/anomalies returns anomaly detection results."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/anomalies", headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert "anomalies" in data
            assert "total_records_analyzed" in data
            assert "anomaly_rate" in data
            assert "period_start" in data
            assert "period_end" in data

    @pytest.mark.anyio
    async def test_detect_anomalies_with_sensitivity(
        self, v2_app, test_data, auth_headers
    ):
        """Test GET /api/analytics/anomalies with different sensitivities."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Test with high sensitivity (should find more anomalies)
            response_high = await client.get(
                "/api/analytics/anomalies",
                params={"sensitivity": "high"},
                headers=auth_headers,
            )
            assert response_high.status_code == 200

            # Test with low sensitivity (should find fewer anomalies)
            response_low = await client.get(
                "/api/analytics/anomalies",
                params={"sensitivity": "low"},
                headers=auth_headers,
            )
            assert response_low.status_code == 200

    @pytest.mark.anyio
    async def test_detect_anomalies_invalid_sensitivity(
        self, v2_app, test_data, auth_headers
    ):
        """Test GET /api/analytics/anomalies with invalid sensitivity."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/anomalies",
                params={"sensitivity": "invalid"},
                headers=auth_headers,
            )
            assert response.status_code == 422  # Validation error


class TestCorrelationsRoutes:
    """Test correlation analysis routes."""

    @pytest.mark.anyio
    async def test_analyze_correlations(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/correlations returns correlation analysis."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/correlations", headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert "correlations" in data
            assert "sample_size" in data
            assert "factors_analyzed" in data

    @pytest.mark.anyio
    async def test_analyze_correlations_with_factors(
        self, v2_app, test_data, auth_headers
    ):
        """Test GET /api/analytics/correlations with specific factors."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/correlations",
                params={"factors": "duration,total_tokens"},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()

            assert "factors_analyzed" in data


class TestExportRoutes:
    """Test data export routes."""

    @pytest.mark.anyio
    async def test_export_csv(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/export/csv returns CSV file."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/export/csv", headers=auth_headers
            )
            assert response.status_code == 200
            assert "text/csv" in response.headers["content-type"]
            assert "attachment" in response.headers["content-disposition"]

    @pytest.mark.anyio
    async def test_export_csv_with_filters(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/export/csv with filters."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/export/csv",
                params={
                    "suite_name": "test-suite",
                    "include_runs": True,
                },
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert "text/csv" in response.headers["content-type"]


class TestScheduledReportsRoutes:
    """Test scheduled reports routes."""

    @pytest.mark.anyio
    async def test_list_scheduled_reports(
        self, v2_app, scheduled_reports_data, auth_headers
    ):
        """Test GET /api/analytics/reports returns list of reports."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get("/api/analytics/reports", headers=auth_headers)
            assert response.status_code == 200
            data = response.json()

            assert "items" in data
            assert "total" in data
            assert data["total"] == 3

    @pytest.mark.anyio
    async def test_list_scheduled_reports_filter_active(
        self, v2_app, scheduled_reports_data, auth_headers
    ):
        """Test GET /api/analytics/reports with active filter."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/reports",
                params={"is_active": True},
                headers=auth_headers,
            )
            assert response.status_code == 200
            data = response.json()

            assert data["total"] == 2
            for report in data["items"]:
                assert report["is_active"] is True

    @pytest.mark.anyio
    async def test_get_scheduled_report(
        self, v2_app, scheduled_reports_data, auth_headers
    ):
        """Test GET /api/analytics/reports/{id} returns report details."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # First get the list to find an ID
            list_response = await client.get(
                "/api/analytics/reports", headers=auth_headers
            )
            report_id = list_response.json()["items"][0]["id"]

            response = await client.get(
                f"/api/analytics/reports/{report_id}", headers=auth_headers
            )
            assert response.status_code == 200
            data = response.json()

            assert data["id"] == report_id
            assert "name" in data
            assert "frequency" in data

    @pytest.mark.anyio
    async def test_get_scheduled_report_not_found(
        self, v2_app, scheduled_reports_data, auth_headers
    ):
        """Test GET /api/analytics/reports/{id} returns 404 for missing report."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/reports/99999", headers=auth_headers
            )
            assert response.status_code == 404

    @pytest.mark.anyio
    async def test_create_scheduled_report_unauthorized(
        self, v2_app, scheduled_reports_data
    ):
        """Test POST /api/analytics/reports requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/analytics/reports",
                json={
                    "name": "New Report",
                    "frequency": "daily",
                    "recipients": ["new@example.com"],
                },
            )
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_update_scheduled_report_unauthorized(
        self, v2_app, scheduled_reports_data
    ):
        """Test PUT /api/analytics/reports/{id} requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.put(
                "/api/analytics/reports/1",
                json={"name": "Updated Report"},
            )
            assert response.status_code == 401

    @pytest.mark.anyio
    async def test_delete_scheduled_report_unauthorized(
        self, v2_app, scheduled_reports_data
    ):
        """Test DELETE /api/analytics/reports/{id} requires authentication."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.delete("/api/analytics/reports/1")
            assert response.status_code == 401


class TestExcelExportRoutes:
    """Test Excel export routes."""

    @pytest.mark.anyio
    async def test_export_excel(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/export/excel returns Excel file."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/export/excel", headers=auth_headers
            )
            assert response.status_code == 200
            assert (
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                in response.headers["content-type"]
            )
            assert "attachment" in response.headers["content-disposition"]

    @pytest.mark.anyio
    async def test_export_excel_with_options(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/export/excel with include options."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/export/excel",
                params={
                    "include_trends": True,
                    "include_anomalies": True,
                },
                headers=auth_headers,
            )
            assert response.status_code == 200


class TestAnalyticsRouteConsistency:
    """Test that analytics routes follow expected patterns."""

    @pytest.mark.anyio
    async def test_analytics_routes_registered(self, v2_app):
        """Verify analytics routes are registered."""
        routes = [route.path for route in v2_app.routes]

        expected_routes = [
            "/api/analytics/trends",
            "/api/analytics/anomalies",
            "/api/analytics/correlations",
            "/api/analytics/export/csv",
            "/api/analytics/export/excel",
            "/api/analytics/reports",
            "/api/analytics/reports/{report_id}",
        ]

        for expected in expected_routes:
            matching = [r for r in routes if expected in r]
            assert len(matching) > 0, f"Route {expected} not found"


class TestTrendsWithData:
    """Test trend analysis with various data scenarios."""

    @pytest.mark.anyio
    async def test_trends_with_date_range(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/trends with date range filter."""
        from datetime import datetime, timedelta

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            # Use a date range that covers the test data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            response = await client.get(
                "/api/analytics/trends",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
                headers=auth_headers,
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_trends_with_agent_filter(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/trends filtered by agent."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/trends",
                params={"agent_name": "test-agent"},
                headers=auth_headers,
            )
            assert response.status_code == 200


class TestAnomaliesWithData:
    """Test anomaly detection with various scenarios."""

    @pytest.mark.anyio
    async def test_anomalies_with_date_range(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/anomalies with date range."""
        from datetime import datetime, timedelta

        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            response = await client.get(
                "/api/analytics/anomalies",
                params={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "sensitivity": "medium",
                },
                headers=auth_headers,
            )
            assert response.status_code == 200

    @pytest.mark.anyio
    async def test_anomalies_with_specific_metrics(
        self, v2_app, test_data, auth_headers
    ):
        """Test GET /api/analytics/anomalies with specific metrics."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/anomalies",
                params={"metrics": "score"},
                headers=auth_headers,
            )
            assert response.status_code == 200


class TestCorrelationsWithData:
    """Test correlation analysis with various scenarios."""

    @pytest.mark.anyio
    async def test_correlations_empty_factors(self, v2_app, test_data, auth_headers):
        """Test GET /api/analytics/correlations with empty factors."""
        async with AsyncClient(
            transport=ASGITransport(app=v2_app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/api/analytics/correlations",
                params={"factors": "duration"},
                headers=auth_headers,
            )
            assert response.status_code == 200
