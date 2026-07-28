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
        # `find_spec`, not the legacy `find_module`: the latter was removed
        # from the import system in 3.12, so a blocker using it is inert and
        # this test would pass without blocking anything.
        script = (
            "import sys\n"
            "class Blocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
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
        # By membership, not by count: a count moves whenever the vocabulary
        # grows and says nothing about *which* types crossed the boundary,
        # so it can stay satisfied while the wrong ones do.
        permitted = set(described["allowed_assertion_types"])
        # `composite` is permitted again as of ADR-008 track A: it resolves its
        # leaves through this very resolver, so nesting an excluded assertion
        # under it no longer reaches one. `file_exists` as of track B: its root
        # is granted by the composition, so it addresses the artifact sandbox
        # rather than a directory the suite named.
        assert {
            "contains",
            "behavior",
            "findings_match",
            "composite",
            "file_exists",
        } <= permitted
        assert permitted.isdisjoint({"pytest", "code_exec", "llm_eval", "factuality"})

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

    def _compose_files(self) -> list[Path]:
        """Every compose file in the repo, not just the one at the root.

        Checking only the root file is how `deploy/docker-compose.yml` sat
        there running `uvicorn atp.dashboard.v2.factory:app` -- the production
        path bypassing the composition root while a green test said otherwise.
        """
        found = sorted(REPO_ROOT.glob("**/docker-compose*.yml"))
        assert found, "no compose files found; the glob is wrong"
        return [p for p in found if ".venv" not in p.parts]

    def test_some_service_serves_the_dashboard(self) -> None:
        """Asserted by what the command does, not by the service's name."""
        commands = []
        for path in self._compose_files():
            compose = yaml.safe_load(path.read_text()) or {}
            for service in (compose.get("services") or {}).values():
                command = service.get("command")
                if command:
                    commands.append(str(command))
        assert any("dashboard" in c or "atp.server" in c for c in commands), commands

    @pytest.mark.parametrize(
        "path", sorted(REPO_ROOT.glob("**/docker-compose*.yml")), ids=str
    )
    def test_no_compose_file_bypasses_the_composition_root(self, path: Path) -> None:
        """A completion-only server deployed by accident is the whole risk.

        Reads the parsed commands rather than the file text: a comment
        explaining why the direct path is wrong is documentation, and a test
        that punishes documentation gets the documentation deleted.
        """
        compose = yaml.safe_load(path.read_text()) or {}
        for name, service in (compose.get("services") or {}).items():
            command = str(service.get("command") or "")
            assert "atp.dashboard.v2.factory:app" not in command, (
                f"{path.relative_to(REPO_ROOT)} service '{name}' launches the "
                "dashboard app directly, which skips the evaluator resolver. "
                "Use `uvicorn atp.server:create_server_app --factory` or "
                "`atp dashboard`."
            )

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
