"""Tests for dashboard UI routes."""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from atp.dashboard.v2.factory import create_test_app


@pytest.fixture
def app():
    """Create test app with UI routes."""
    return create_test_app()


@pytest.fixture
async def client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestUIRoutes:
    """Tests for UI page routes."""

    @pytest.mark.anyio
    async def test_home_returns_html(self, client) -> None:
        """GET /ui/ returns HTML page."""
        resp = await client.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "ATP Platform" in resp.text

    @pytest.mark.anyio
    async def test_benchmarks_returns_html(self, client) -> None:
        """GET /ui/benchmarks returns HTML page."""
        resp = await client.get("/ui/benchmarks")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Benchmarks" in resp.text

    @pytest.mark.anyio
    async def test_games_page_returns_html(self, client) -> None:
        """GET /ui/games returns 200 with Games heading."""
        resp = await client.get("/ui/games")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Games" in resp.text

    @pytest.mark.anyio
    async def test_runs_page_returns_html(self, client) -> None:
        """GET /ui/runs returns 200 with Runs heading."""
        resp = await client.get("/ui/runs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Runs" in resp.text

    @pytest.mark.anyio
    async def test_leaderboard_page_returns_html(self, client) -> None:
        """GET /ui/leaderboard returns 200 with leaderboard heading."""
        resp = await client.get("/ui/leaderboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Leaderboard" in resp.text

    @pytest.mark.anyio
    async def test_leaderboard_benchmark_filter(self, client) -> None:
        """GET /ui/leaderboard?benchmark_id=1 returns 200."""
        resp = await client.get("/ui/leaderboard?benchmark_id=1")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.anyio
    async def test_leaderboard_partial(self, client) -> None:
        """GET /ui/leaderboard?partial=1 returns table fragment."""
        resp = await client.get("/ui/leaderboard?partial=1")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    @pytest.mark.anyio
    async def test_suites_page_returns_html(self, client) -> None:
        """GET /ui/suites returns suites page with table."""
        resp = await client.get("/ui/suites")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Suites" in resp.text
        assert "Upload YAML Suite" in resp.text

    @pytest.mark.anyio
    async def test_analytics_page_returns_html(self, client) -> None:
        """GET /ui/analytics returns 200 with analytics content."""
        resp = await client.get("/ui/analytics")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Analytics" in resp.text

    @pytest.mark.anyio
    async def test_login_page(self, client) -> None:
        """GET /ui/login returns login page."""
        resp = await client.get("/ui/login")
        assert resp.status_code == 200
        assert "Login" in resp.text or "login" in resp.text

    @pytest.mark.anyio
    async def test_sidebar_has_all_nav_links(self, client) -> None:
        """Home page sidebar contains all navigation links."""
        resp = await client.get("/ui/")
        html = resp.text
        assert "/ui/benchmarks" in html
        assert "/ui/games" in html
        assert "/ui/runs" in html
        assert "/ui/leaderboard" in html
        assert "/ui/suites" in html
        assert "/ui/analytics" in html

    @pytest.mark.anyio
    async def test_htmx_loaded(self, client) -> None:
        """Pages include HTMX CDN script."""
        resp = await client.get("/ui/")
        assert "htmx.org" in resp.text

    @pytest.mark.anyio
    async def test_pico_css_loaded(self, client) -> None:
        """Pages include Pico CSS CDN."""
        resp = await client.get("/ui/")
        assert "pico" in resp.text.lower()


class TestSuitesPage:
    """Tests for suites page routes."""

    @pytest.mark.anyio
    async def test_suites_create_benchmark_missing(self, client) -> None:
        """POST /ui/suites/999/create-benchmark returns 404 for unknown suite."""
        resp = await client.post("/ui/suites/999/create-benchmark")
        assert resp.status_code == 404
        assert "not found" in resp.text.lower()

    @pytest.mark.anyio
    async def test_suites_upload_invalid_yaml(self, client) -> None:
        """POST /ui/suites/upload with invalid YAML returns error fragment."""
        import io

        bad_yaml = b": invalid: yaml: ["
        resp = await client.post(
            "/ui/suites/upload",
            files={"file": ("bad.yaml", io.BytesIO(bad_yaml), "application/yaml")},
        )
        assert resp.status_code == 200
        assert "failed" in resp.text.lower() or "error" in resp.text.lower()

    @pytest.mark.anyio
    async def test_suites_upload_valid_yaml(self, client) -> None:
        """POST /ui/suites/upload with valid YAML creates a suite."""
        import io

        yaml_content = b"""test_suite: ui-test-suite
version: "1.0"
tests:
  - id: t1
    name: Say Hello
    task:
      prompt: "Say hello"
      description: "Simple hello test"
    assertions:
      - type: contains
        config:
          value: hello
"""
        resp = await client.post(
            "/ui/suites/upload",
            files={
                "file": ("suite.yaml", io.BytesIO(yaml_content), "application/yaml")
            },
        )
        assert resp.status_code == 200
        assert "ui-test-suite" in resp.text

    @pytest.mark.anyio
    async def test_suites_create_benchmark_from_uploaded(self, client) -> None:
        """Upload a suite then create a benchmark from it."""
        import io

        yaml_content = b"""test_suite: benchmark-from-suite
version: "1.0"
tests:
  - id: t1
    name: Say Hello
    task:
      prompt: "Say hello"
      description: "Simple hello test"
    assertions:
      - type: contains
        config:
          value: hello
"""
        upload_resp = await client.post(
            "/ui/suites/upload",
            files={
                "file": ("suite.yaml", io.BytesIO(yaml_content), "application/yaml")
            },
        )
        assert upload_resp.status_code == 200
        # Extract suite id from response text (id=N)
        import re

        match = re.search(r"id=(\d+)", upload_resp.text)
        assert match, f"No suite id in upload response: {upload_resp.text}"
        suite_id = int(match.group(1))

        resp = await client.post(f"/ui/suites/{suite_id}/create-benchmark")
        assert resp.status_code == 200
        assert "benchmark-from-suite" in resp.text
        assert "created" in resp.text.lower()


class TestBenchmarkDetail:
    """Tests for benchmark detail page."""

    @pytest.mark.anyio
    async def test_benchmark_not_found(self, client) -> None:
        """GET /ui/benchmarks/999 returns 404 error page."""
        resp = await client.get("/ui/benchmarks/999")
        assert resp.status_code == 404
        assert "Not Found" in resp.text


class TestRunDetailPage:
    """Tests for run detail page /ui/runs/{run_id}."""

    @pytest.mark.anyio
    async def test_run_detail_returns_html(self, client) -> None:
        """GET /ui/runs/{id} returns HTML for existing run."""
        import uuid

        from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus, TaskResult
        from atp.dashboard.database import get_database

        bm_name = f"test-bm-{uuid.uuid4().hex[:8]}"
        db = get_database()
        async with db.session() as session:
            bm = Benchmark(name=bm_name, tasks_count=2, suite={}, tags=[])
            session.add(bm)
            await session.flush()

            run = Run(
                benchmark_id=bm.id,
                agent_name="test-agent",
                status=RunStatus.COMPLETED,
                current_task_index=2,
                total_score=0.85,
            )
            session.add(run)
            await session.flush()

            tr = TaskResult(
                run_id=run.id,
                task_index=0,
                request={"task": {"description": "Solve 1+1"}},
                response={"answer": "2"},
                score=1.0,
            )
            session.add(tr)
            await session.commit()

            run_id = run.id

        resp = await client.get(f"/ui/runs/{run_id}")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert f"Run #{run_id}" in resp.text
        assert bm_name in resp.text
        assert "test-agent" in resp.text
        assert "Solve 1+1" in resp.text

    @pytest.mark.anyio
    async def test_run_detail_404(self, client) -> None:
        """GET /ui/runs/99999 returns 404."""
        resp = await client.get("/ui/runs/99999")
        assert resp.status_code == 404
        assert "Not Found" in resp.text

    @pytest.mark.anyio
    async def test_run_detail_htmx_header_partial(self, client) -> None:
        """HTMX request with HX-Target: run-header returns partial."""
        import uuid

        from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
        from atp.dashboard.database import get_database

        bm_name = f"htmx-bm-{uuid.uuid4().hex[:8]}"
        db = get_database()
        async with db.session() as session:
            bm = Benchmark(name=bm_name, tasks_count=1, suite={}, tags=[])
            session.add(bm)
            await session.flush()

            run = Run(
                benchmark_id=bm.id,
                agent_name="htmx-agent",
                status=RunStatus.IN_PROGRESS,
                current_task_index=0,
            )
            session.add(run)
            await session.commit()
            run_id = run.id

        resp = await client.get(
            f"/ui/runs/{run_id}",
            headers={"HX-Request": "true", "HX-Target": "run-header"},
        )
        assert resp.status_code == 200
        assert "run-header" in resp.text
        assert "hx-trigger" in resp.text  # Should have polling for in-progress

    @pytest.mark.anyio
    async def test_run_detail_htmx_tasks_partial(self, client) -> None:
        """HTMX request with HX-Target: run-tasks returns partial."""
        import uuid

        from atp.dashboard.benchmark.models import Benchmark, Run, RunStatus
        from atp.dashboard.database import get_database

        bm_name = f"tasks-bm-{uuid.uuid4().hex[:8]}"
        db = get_database()
        async with db.session() as session:
            bm = Benchmark(name=bm_name, tasks_count=1, suite={}, tags=[])
            session.add(bm)
            await session.flush()

            run = Run(
                benchmark_id=bm.id,
                agent_name="tasks-agent",
                status=RunStatus.COMPLETED,
                current_task_index=1,
            )
            session.add(run)
            await session.commit()
            run_id = run.id

        resp = await client.get(
            f"/ui/runs/{run_id}",
            headers={"HX-Request": "true", "HX-Target": "run-tasks"},
        )
        assert resp.status_code == 200
        assert "run-tasks" in resp.text
        assert "hx-trigger" not in resp.text  # No polling for completed run
