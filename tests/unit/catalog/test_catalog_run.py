"""Unit tests for _execute_catalog_run and helpers in catalog CLI."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atp.cli.commands.catalog import _execute_catalog_run, _parse_adapter_config

# ---------------------------------------------------------------------------
# _parse_adapter_config tests
# ---------------------------------------------------------------------------


def test_parse_adapter_config_empty() -> None:
    """Empty string returns empty dict."""
    assert _parse_adapter_config("") == {}


def test_parse_adapter_config_blank() -> None:
    """Blank string returns empty dict."""
    assert _parse_adapter_config("   ") == {}


def test_parse_adapter_config_single_pair() -> None:
    """Single key=value pair is parsed correctly."""
    result = _parse_adapter_config("url=http://localhost:8080")
    assert result == {"url": "http://localhost:8080"}


def test_parse_adapter_config_multiple_pairs() -> None:
    """Multiple comma-separated pairs are all parsed."""
    result = _parse_adapter_config("url=http://host,timeout=30")
    assert result == {"url": "http://host", "timeout": "30"}


def test_parse_adapter_config_flag_without_value() -> None:
    """Key without value is set to True."""
    result = _parse_adapter_config("verbose")
    assert result == {"verbose": True}


def test_parse_adapter_config_mixed() -> None:
    """Mix of key=value and bare key is parsed correctly."""
    result = _parse_adapter_config("url=http://host,debug")
    assert result == {"url": "http://host", "debug": True}


# ---------------------------------------------------------------------------
# Helpers for _execute_catalog_run tests
# ---------------------------------------------------------------------------


def _make_mock_session() -> AsyncMock:
    """Build a mock async SQLAlchemy session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


def _make_catalog_suite(tests: list[MagicMock] | None = None) -> MagicMock:
    """Build a mock CatalogSuite."""
    suite = MagicMock()
    suite.name = "Test Suite"
    suite.suite_yaml = "test_suite: test\ntests: []"
    suite.tests = tests or []
    return suite


def _make_suite_result(
    test_results: list[Any] | None = None,
) -> MagicMock:
    """Build a mock SuiteResult."""
    from atp.runner.models import SuiteResult

    result = MagicMock(spec=SuiteResult)
    result.tests = test_results or []
    result.passed_tests = 0
    result.total_tests = 0
    return result


# ---------------------------------------------------------------------------
# _execute_catalog_run tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_execute_catalog_run_suite_not_found() -> None:
    """Raises SystemExit(1) when suite is not found in DB."""
    session = _make_mock_session()

    with (
        patch(
            "atp.catalog.repository.CatalogRepository.get_suite_by_path",
            new=AsyncMock(return_value=None),
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        await _execute_catalog_run(
            session=session,
            category_slug="missing",
            suite_slug="suite",
            adapter_type="http",
            adapter_config={},
            agent_name="agent",
            runs_per_test=1,
        )

    assert exc_info.value.code == 1


@pytest.mark.anyio
async def test_execute_catalog_run_calls_orchestrator() -> None:
    """Verifies that TestOrchestrator.run_suite is invoked with correct args."""
    session = _make_mock_session()
    catalog_suite = _make_catalog_suite()
    suite_result = _make_suite_result()

    mock_loader = MagicMock()
    mock_loaded_suite = MagicMock()
    mock_loaded_suite.tests = []
    mock_loader.load_string.return_value = mock_loaded_suite

    mock_adapter = MagicMock()

    mock_orchestrator = AsyncMock()
    mock_orchestrator.__aenter__ = AsyncMock(return_value=mock_orchestrator)
    mock_orchestrator.__aexit__ = AsyncMock(return_value=False)
    mock_orchestrator.run_suite = AsyncMock(return_value=suite_result)

    with (
        patch(
            "atp.catalog.repository.CatalogRepository.get_suite_by_path",
            new=AsyncMock(return_value=catalog_suite),
        ),
        patch(
            "atp.cli.commands.catalog.TestLoader",
            return_value=mock_loader,
        ),
        patch(
            "atp.cli.commands.catalog.create_adapter",
            return_value=mock_adapter,
        ),
        patch(
            "atp.cli.commands.catalog.TestOrchestrator",
            return_value=mock_orchestrator,
        ),
    ):
        await _execute_catalog_run(
            session=session,
            category_slug="coding",
            suite_slug="file-ops",
            adapter_type="http",
            adapter_config={"url": "http://localhost"},
            agent_name="my-agent",
            runs_per_test=2,
        )

    mock_loader.load_string.assert_called_once_with(catalog_suite.suite_yaml)
    mock_orchestrator.run_suite.assert_called_once_with(
        suite=mock_loaded_suite,
        agent_name="my-agent",
        runs_per_test=2,
    )


@pytest.mark.anyio
async def test_execute_catalog_run_creates_submissions() -> None:
    """Verifies that create_submission is called for each test result."""
    session = _make_mock_session()

    catalog_test = MagicMock()
    catalog_test.id = 42
    catalog_test.slug = "test-one"
    catalog_test.name = "Test One"
    catalog_test.avg_score = None
    catalog_test.best_score = None

    catalog_suite = _make_catalog_suite(tests=[catalog_test])

    # Build a test result matching the catalog_test slug
    test_result = MagicMock()
    test_result.test = MagicMock()
    test_result.test.id = "test-one"
    test_result.successful_runs = 1
    test_result.total_runs = 1
    test_result.duration_seconds = 5.0
    test_result.runs = []

    suite_result = _make_suite_result(test_results=[test_result])
    suite_result.passed_tests = 1
    suite_result.total_tests = 1

    mock_loader = MagicMock()
    mock_loaded_suite = MagicMock()
    mock_loaded_suite.tests = []
    mock_loader.load_string.return_value = mock_loaded_suite

    mock_orchestrator = AsyncMock()
    mock_orchestrator.__aenter__ = AsyncMock(return_value=mock_orchestrator)
    mock_orchestrator.__aexit__ = AsyncMock(return_value=False)
    mock_orchestrator.run_suite = AsyncMock(return_value=suite_result)

    create_submission_mock = AsyncMock(return_value=MagicMock())
    update_stats_mock = AsyncMock()
    get_top_submissions_mock = AsyncMock(return_value=[])

    with (
        patch(
            "atp.catalog.repository.CatalogRepository.get_suite_by_path",
            new=AsyncMock(return_value=catalog_suite),
        ),
        patch(
            "atp.cli.commands.catalog.TestLoader",
            return_value=mock_loader,
        ),
        patch(
            "atp.cli.commands.catalog.create_adapter",
            return_value=MagicMock(),
        ),
        patch(
            "atp.cli.commands.catalog.TestOrchestrator",
            return_value=mock_orchestrator,
        ),
        patch(
            "atp.catalog.repository.CatalogRepository.create_submission",
            new=create_submission_mock,
        ),
        patch(
            "atp.catalog.repository.CatalogRepository.update_test_stats",
            new=update_stats_mock,
        ),
        patch(
            "atp.catalog.repository.CatalogRepository.get_top_submissions",
            new=get_top_submissions_mock,
        ),
    ):
        await _execute_catalog_run(
            session=session,
            category_slug="coding",
            suite_slug="file-ops",
            adapter_type="cli",
            adapter_config={},
            agent_name="my-agent",
            runs_per_test=1,
        )

    create_submission_mock.assert_called_once()
    call_kwargs = create_submission_mock.call_args.kwargs
    assert call_kwargs["test_id"] == 42
    assert call_kwargs["agent_name"] == "my-agent"
    assert call_kwargs["agent_type"] == "cli"
    assert call_kwargs["score"] == 100.0

    update_stats_mock.assert_called_once_with(42)
    session.commit.assert_called_once()


@pytest.mark.anyio
async def test_execute_catalog_run_extracts_metrics() -> None:
    """Verifies token/cost metrics are extracted from first run response."""
    session = _make_mock_session()

    catalog_test = MagicMock()
    catalog_test.id = 10
    catalog_test.slug = "t1"
    catalog_test.name = "T1"
    catalog_test.avg_score = None
    catalog_test.best_score = None

    catalog_suite = _make_catalog_suite(tests=[catalog_test])

    # Build run with metrics
    mock_metrics = MagicMock()
    mock_metrics.total_tokens = 500
    mock_metrics.input_tokens = None
    mock_metrics.output_tokens = None
    mock_metrics.cost_usd = 0.005

    mock_response = MagicMock()
    mock_response.metrics = mock_metrics

    mock_run = MagicMock()
    mock_run.response = mock_response
    mock_run.success = True

    test_result = MagicMock()
    test_result.test = MagicMock()
    test_result.test.id = "t1"
    test_result.successful_runs = 1
    test_result.total_runs = 1
    test_result.duration_seconds = 3.0
    test_result.runs = [mock_run]

    suite_result = _make_suite_result(test_results=[test_result])
    suite_result.passed_tests = 1
    suite_result.total_tests = 1

    mock_loader = MagicMock()
    mock_loaded_suite = MagicMock()
    mock_loaded_suite.tests = []
    mock_loader.load_string.return_value = mock_loaded_suite

    mock_orchestrator = AsyncMock()
    mock_orchestrator.__aenter__ = AsyncMock(return_value=mock_orchestrator)
    mock_orchestrator.__aexit__ = AsyncMock(return_value=False)
    mock_orchestrator.run_suite = AsyncMock(return_value=suite_result)

    create_submission_mock = AsyncMock(return_value=MagicMock())

    with (
        patch(
            "atp.catalog.repository.CatalogRepository.get_suite_by_path",
            new=AsyncMock(return_value=catalog_suite),
        ),
        patch(
            "atp.cli.commands.catalog.TestLoader",
            return_value=mock_loader,
        ),
        patch(
            "atp.cli.commands.catalog.create_adapter",
            return_value=MagicMock(),
        ),
        patch(
            "atp.cli.commands.catalog.TestOrchestrator",
            return_value=mock_orchestrator,
        ),
        patch(
            "atp.catalog.repository.CatalogRepository.create_submission",
            new=create_submission_mock,
        ),
        patch(
            "atp.catalog.repository.CatalogRepository.update_test_stats",
            new=AsyncMock(),
        ),
        patch(
            "atp.catalog.repository.CatalogRepository.get_top_submissions",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await _execute_catalog_run(
            session=session,
            category_slug="coding",
            suite_slug="file-ops",
            adapter_type="http",
            adapter_config={},
            agent_name="bot",
            runs_per_test=1,
        )

    call_kwargs = create_submission_mock.call_args.kwargs
    assert call_kwargs["total_tokens"] == 500
    assert call_kwargs["cost_usd"] == pytest.approx(0.005)
