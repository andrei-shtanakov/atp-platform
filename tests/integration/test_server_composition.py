"""Contract tests for how a running server composes evaluation.

The six checks the design calls for, in order:

1. the dashboard factory imports without atp-platform present;
2. `deterministic_allowlist` with no resolver fails at startup;
3. `completion_only` with no resolver runs and says so;
4. the top-level composition root injects a filtered resolver;
5. a forbidden evaluator cannot be obtained even though the full platform
   registry has it registered;
6. the production command uses that same composition root.

The point of the set is that behaviour must not depend on how the process
happened to be launched. A server that quietly scores completions while
believing it scores quality publishes numbers that look real.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from atp.dashboard.v2.evaluation_composition import EvaluationCapability
from atp.dashboard.v2.factory import create_dashboard_app
from atp.evaluation import (
    COMPLETION_ONLY,
    DETERMINISTIC_ALLOWLIST,
    UNTRUSTED_SUBMISSION,
    EvaluatorNotPermitted,
    FilteredResolver,
    IncompleteComposition,
)
from atp.server import build_server_resolver, create_server_app

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app refuses to build without a signing key."""
    monkeypatch.setenv("ATP_SECRET_KEY", "test-secret-for-composition-tests")


class TestFactoryStandsAlone:
    """1 — the dashboard must not need atp-platform to be importable."""

    def test_factory_module_imports_without_platform_modules(self) -> None:
        """Run in a subprocess with the platform packages blocked entirely."""
        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name.split('.')[0:2] == ['atp', 'evaluators']:\n"
            "            raise ImportError('atp.evaluators is not available here')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Blocker())\n"
            "from atp.dashboard.v2.factory import create_dashboard_app\n"
            "app = create_dashboard_app("
            "evaluator_resolver=None, evaluation_mode='completion_only')\n"
            "print(app.state.evaluation.mode)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "ATP_SECRET_KEY": "x", "HOME": "/tmp"},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "completion_only" in result.stdout


class TestStartupRefusesAnIncompleteComposition:
    """2 and 3 — a declared mode must match what was injected."""

    def test_deterministic_without_a_resolver_fails_at_startup(self) -> None:
        with pytest.raises(IncompleteComposition) as exc:
            create_dashboard_app(
                evaluator_resolver=None, evaluation_mode=DETERMINISTIC_ALLOWLIST
            )
        assert "requires an evaluator resolver" in str(exc.value)

    def test_completion_only_without_a_resolver_runs_and_says_so(self) -> None:
        app = create_dashboard_app(
            evaluator_resolver=None, evaluation_mode=COMPLETION_ONLY
        )
        described = app.state.evaluation.describe()
        assert described["evaluation_mode"] == "completion_only"
        assert described["resolver_connected"] is False
        assert described["allowed_assertion_types"] == []

    def test_a_resolver_in_completion_only_is_also_refused(self) -> None:
        """Ignoring it would leave someone believing evaluation was on."""
        resolver = build_server_resolver()
        with pytest.raises(IncompleteComposition):
            create_dashboard_app(
                evaluator_resolver=resolver, evaluation_mode=COMPLETION_ONLY
            )

    def test_validation_happens_before_the_app_is_built(self) -> None:
        """Fail fast: nothing should be half-constructed behind the error."""
        with pytest.raises(IncompleteComposition):
            create_dashboard_app(
                evaluator_resolver=None,
                evaluation_mode=DETERMINISTIC_ALLOWLIST,
                title="should-never-be-created",
            )


class TestCompositionRoot:
    """4 and 5 — what the root injects, and what it withholds."""

    def test_root_wires_a_filtered_resolver(self) -> None:
        app = create_server_app()
        described = app.state.evaluation.describe()
        assert described["evaluation_mode"] == "deterministic_allowlist"
        assert described["resolver_connected"] is True
        assert described["policy"] == "untrusted_submission"
        assert len(described["allowed_assertion_types"]) == 25

    def test_forbidden_evaluator_is_unreachable_though_registered(self) -> None:
        """The platform registry has it; what crosses the boundary does not."""
        from atp.evaluators.registry import get_registry

        assert get_registry().get_evaluator_for_assertion("pytest") is not None

        resolver = build_server_resolver()
        with pytest.raises(EvaluatorNotPermitted):
            resolver.create_for_assertion("pytest")

    @pytest.mark.parametrize("assertion", ["pytest", "code_exec", "llm_eval", "npm"])
    def test_no_executing_or_network_evaluator_crosses_the_boundary(
        self, assertion: str
    ) -> None:
        resolver = build_server_resolver()
        with pytest.raises(EvaluatorNotPermitted):
            resolver.create_for_assertion(assertion)

    def test_permitted_evaluator_is_still_reachable(self) -> None:
        """A boundary that blocks everything would be simpler and useless."""
        assert build_server_resolver().create_for_assertion("contains") is not None

    def test_the_resolver_does_not_expose_the_inner_registry(self) -> None:
        """A public attribute would make the restriction opt-in again."""
        resolver = build_server_resolver()
        assert not hasattr(resolver, "registry")
        assert not hasattr(resolver, "inner")


class TestProductionLaunchPath:
    """6 — the deployed command must use the composition root."""

    def test_docker_compose_starts_the_dashboard_through_the_cli(self) -> None:
        """Asserted by what the command does, not by the service's name."""
        compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
        commands = [
            service["command"]
            for service in compose["services"].values()
            if isinstance(service.get("command"), list)
        ]
        serving = [c for c in commands if "atp" in c and "dashboard" in c]
        assert serving, f"no service runs `atp dashboard`; commands were {commands}"

    def test_no_service_bypasses_the_root_with_a_raw_uvicorn_command(self) -> None:
        """`uvicorn atp.dashboard...:app` would be a completion-only server."""
        raw = (REPO_ROOT / "docker-compose.yml").read_text()
        assert "atp.dashboard.v2.factory:app" not in raw

    def test_the_cli_launches_the_composition_root(self) -> None:
        """Not `atp.dashboard...:app`, which would be completion-only."""
        source = (REPO_ROOT / "atp" / "cli" / "main.py").read_text()
        assert "from atp.server import run_server" in source

    def test_the_root_serves_itself_as_a_uvicorn_factory(self) -> None:
        """Reload and multi-worker launches must compose identically."""
        source = (REPO_ROOT / "atp" / "server.py").read_text()
        assert '"atp.server:create_server_app"' in source
        assert "factory=True" in source


class TestCapabilityObject:
    """The capability is constructor state, not a mutable global."""

    def test_capability_is_frozen(self) -> None:
        capability = EvaluationCapability.build(COMPLETION_ONLY, None)
        with pytest.raises(Exception):
            capability.mode = DETERMINISTIC_ALLOWLIST  # type: ignore[misc]

    def test_two_apps_can_hold_different_capabilities(self) -> None:
        """A module-global setter would make this impossible to express."""
        plain = create_dashboard_app(
            evaluator_resolver=None, evaluation_mode=COMPLETION_ONLY
        )
        evaluating = create_server_app()
        assert plain.state.evaluation.mode == "completion_only"
        assert evaluating.state.evaluation.mode == "deterministic_allowlist"

    def test_there_is_no_module_level_setter(self) -> None:
        """Order dependence and test leakage are what this design removes."""
        import atp.dashboard.v2.factory as factory

        assert not hasattr(factory, "set_evaluator_registry")
        assert not hasattr(factory, "set_evaluator_resolver")

    def test_filtered_resolver_reports_what_it_permits(self) -> None:
        resolver = FilteredResolver(build_server_resolver(), UNTRUSTED_SUBMISSION)
        assert resolver.permitted_assertion_types() is not None
        assert "contains" in (resolver.permitted_assertion_types() or set())
