"""Unit tests for plugin interfaces and validation."""

from collections.abc import Generator
from typing import Any, ClassVar
from unittest.mock import MagicMock

from atp.evaluators.base import EvalResult
from atp.loader.models import Assertion, TestDefinition
from atp.plugins.discovery import ADAPTER_GROUP, EVALUATOR_GROUP, REPORTER_GROUP
from atp.plugins.interfaces import (
    MIN_ATP_VERSION,
    AdapterPlugin,
    EvaluatorPlugin,
    PluginValidationError,
    PluginVersionError,
    ReporterPlugin,
    check_version_compatibility,
    get_plugin_protocol_for_group,
    get_required_attributes,
    get_required_methods,
    validate_plugin,
    validate_plugin_full,
)
from atp.protocol import ATPEvent, ATPRequest, ATPResponse
from atp.reporters.base import SuiteReport


class TestAdapterPluginProtocol:
    """Tests for AdapterPlugin protocol."""

    def test_valid_adapter_protocol_compliance(self) -> None:
        """Test that a valid adapter class passes protocol check."""

        class ValidAdapter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def adapter_type(self) -> str:
                return "valid"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        # Should pass validation
        errors = validate_plugin(ValidAdapter, ADAPTER_GROUP, "valid")
        assert errors == []

    def test_adapter_missing_adapter_type(self) -> None:
        """Test that adapter without adapter_type fails validation."""

        class MissingAdapterType:
            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        errors = validate_plugin(MissingAdapterType, ADAPTER_GROUP, "test")
        assert len(errors) == 1
        assert "adapter_type" in errors[0]

    def test_adapter_missing_execute(self) -> None:
        """Test that adapter without execute method fails validation."""

        class MissingExecute:
            @property
            def adapter_type(self) -> str:
                return "test"

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        errors = validate_plugin(MissingExecute, ADAPTER_GROUP, "test")
        assert len(errors) == 1
        assert "execute" in errors[0]

    def test_adapter_missing_stream_events(self) -> None:
        """Test that adapter without stream_events method fails validation."""

        class MissingStreamEvents:
            @property
            def adapter_type(self) -> str:
                return "test"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

        errors = validate_plugin(MissingStreamEvents, ADAPTER_GROUP, "test")
        assert len(errors) == 1
        assert "stream_events" in errors[0]

    def test_adapter_with_non_callable_method(self) -> None:
        """Test that adapter with non-callable 'method' fails validation."""

        class NonCallableMethod:
            adapter_type = "test"
            execute = "not a method"  # Not callable
            stream_events = 123  # Not callable

        errors = validate_plugin(NonCallableMethod, ADAPTER_GROUP, "test")
        assert len(errors) == 2
        assert any("execute" in e and "callable" in e for e in errors)
        assert any("stream_events" in e and "callable" in e for e in errors)


class TestEvaluatorPluginProtocol:
    """Tests for EvaluatorPlugin protocol."""

    def test_valid_evaluator_protocol_compliance(self) -> None:
        """Test that a valid evaluator class passes protocol check."""

        class ValidEvaluator:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def name(self) -> str:
                return "valid"

            async def evaluate(
                self,
                task: TestDefinition,
                response: ATPResponse,
                trace: list[ATPEvent],
                assertion: Assertion,
            ) -> EvalResult:
                return MagicMock()

        errors = validate_plugin(ValidEvaluator, EVALUATOR_GROUP, "valid")
        assert errors == []

    def test_evaluator_missing_name(self) -> None:
        """Test that evaluator without name fails validation."""

        class MissingName:
            async def evaluate(
                self,
                task: TestDefinition,
                response: ATPResponse,
                trace: list[ATPEvent],
                assertion: Assertion,
            ) -> EvalResult:
                return MagicMock()

        errors = validate_plugin(MissingName, EVALUATOR_GROUP, "test")
        assert len(errors) == 1
        assert "name" in errors[0]

    def test_evaluator_missing_evaluate(self) -> None:
        """Test that evaluator without evaluate method fails validation."""

        class MissingEvaluate:
            @property
            def name(self) -> str:
                return "test"

        errors = validate_plugin(MissingEvaluate, EVALUATOR_GROUP, "test")
        assert len(errors) == 1
        assert "evaluate" in errors[0]


class TestReporterPluginProtocol:
    """Tests for ReporterPlugin protocol."""

    def test_valid_reporter_protocol_compliance(self) -> None:
        """Test that a valid reporter class passes protocol check."""

        class ValidReporter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def name(self) -> str:
                return "valid"

            def report(self, report: SuiteReport) -> None:
                pass

        errors = validate_plugin(ValidReporter, REPORTER_GROUP, "valid")
        assert errors == []

    def test_reporter_missing_name(self) -> None:
        """Test that reporter without name fails validation."""

        class MissingName:
            def report(self, report: SuiteReport) -> None:
                pass

        errors = validate_plugin(MissingName, REPORTER_GROUP, "test")
        assert len(errors) == 1
        assert "name" in errors[0]

    def test_reporter_missing_report(self) -> None:
        """Test that reporter without report method fails validation."""

        class MissingReport:
            @property
            def name(self) -> str:
                return "test"

        errors = validate_plugin(MissingReport, REPORTER_GROUP, "test")
        assert len(errors) == 1
        assert "report" in errors[0]


class TestVersionCompatibility:
    """Tests for version compatibility checking."""

    def test_compatible_version(self) -> None:
        """Test that compatible versions pass."""

        class CompatiblePlugin:
            atp_version: ClassVar[str] = "0.1.0"

        error = check_version_compatibility(CompatiblePlugin, "0.1.0", "test")
        assert error is None

    def test_compatible_higher_version(self) -> None:
        """Test that higher current version is compatible."""

        class CompatiblePlugin:
            atp_version: ClassVar[str] = "0.1.0"

        error = check_version_compatibility(CompatiblePlugin, "1.0.0", "test")
        assert error is None

    def test_incompatible_version(self) -> None:
        """Test that incompatible versions are rejected."""

        class IncompatiblePlugin:
            atp_version: ClassVar[str] = "2.0.0"

        error = check_version_compatibility(IncompatiblePlugin, "0.1.0", "test")
        assert error is not None
        assert "2.0.0" in error
        assert "0.1.0" in error

    def test_missing_atp_version_uses_default(self) -> None:
        """Test that missing atp_version uses MIN_ATP_VERSION."""

        class NoVersionPlugin:
            pass

        error = check_version_compatibility(NoVersionPlugin, "0.1.0", "test")
        assert error is None

    def test_invalid_version_format(self) -> None:
        """Test that invalid version format returns error."""

        class InvalidVersionPlugin:
            atp_version: ClassVar[str] = "invalid"

        error = check_version_compatibility(InvalidVersionPlugin, "0.1.0", "test")
        assert error is not None
        assert "parsing error" in error.lower() or "invalid" in error.lower()

    def test_invalid_current_version_format(self) -> None:
        """Test that invalid current version format returns error."""

        class CompatiblePlugin:
            atp_version: ClassVar[str] = "0.1.0"

        error = check_version_compatibility(CompatiblePlugin, "invalid", "test")
        assert error is not None

    def test_non_string_atp_version(self) -> None:
        """Test that non-string atp_version returns error."""

        class NonStringVersion:
            atp_version: ClassVar[int] = 1  # type: ignore

        error = check_version_compatibility(NonStringVersion, "0.1.0", "test")
        assert error is not None
        assert "must be a string" in error


class TestPluginValidationError:
    """Tests for PluginValidationError."""

    def test_error_message_formatting(self) -> None:
        """Test that error message is properly formatted."""
        error = PluginValidationError(
            plugin_name="my_plugin",
            plugin_type="atp.adapters",
            errors=["Missing execute method", "Missing adapter_type"],
        )

        assert "my_plugin" in str(error)
        assert "atp.adapters" in str(error)
        assert "Missing execute method" in str(error)
        assert "Missing adapter_type" in str(error)

    def test_error_attributes(self) -> None:
        """Test that error has correct attributes."""
        error = PluginValidationError(
            plugin_name="test",
            plugin_type="atp.evaluators",
            errors=["Error 1", "Error 2"],
        )

        assert error.plugin_name == "test"
        assert error.plugin_type == "atp.evaluators"
        assert error.errors == ["Error 1", "Error 2"]


class TestPluginVersionError:
    """Tests for PluginVersionError."""

    def test_error_message_formatting(self) -> None:
        """Test that error message is properly formatted."""
        error = PluginVersionError(
            plugin_name="my_plugin",
            required_version="2.0.0",
            current_version="0.1.0",
        )

        assert "my_plugin" in str(error)
        assert "2.0.0" in str(error)
        assert "0.1.0" in str(error)

    def test_error_attributes(self) -> None:
        """Test that error has correct attributes."""
        error = PluginVersionError(
            plugin_name="test",
            required_version="1.0.0",
            current_version="0.5.0",
        )

        assert error.plugin_name == "test"
        assert error.required_version == "1.0.0"
        assert error.current_version == "0.5.0"


class TestGetPluginProtocolForGroup:
    """Tests for get_plugin_protocol_for_group function."""

    def test_adapter_group_returns_adapter_protocol(self) -> None:
        """Test that adapter group returns AdapterPlugin."""
        protocol = get_plugin_protocol_for_group(ADAPTER_GROUP)
        assert protocol is AdapterPlugin

    def test_evaluator_group_returns_evaluator_protocol(self) -> None:
        """Test that evaluator group returns EvaluatorPlugin."""
        protocol = get_plugin_protocol_for_group(EVALUATOR_GROUP)
        assert protocol is EvaluatorPlugin

    def test_reporter_group_returns_reporter_protocol(self) -> None:
        """Test that reporter group returns ReporterPlugin."""
        protocol = get_plugin_protocol_for_group(REPORTER_GROUP)
        assert protocol is ReporterPlugin

    def test_unknown_group_returns_none(self) -> None:
        """Test that unknown group returns None."""
        protocol = get_plugin_protocol_for_group("unknown.group")
        assert protocol is None


class TestGetRequiredAttributes:
    """Tests for get_required_attributes function."""

    def test_adapter_required_attributes(self) -> None:
        """Test adapter required attributes."""
        attrs = get_required_attributes(ADAPTER_GROUP)
        assert "adapter_type" in attrs
        assert "execute" in attrs
        assert "stream_events" in attrs

    def test_evaluator_required_attributes(self) -> None:
        """Test evaluator required attributes."""
        attrs = get_required_attributes(EVALUATOR_GROUP)
        assert "name" in attrs
        assert "evaluate" in attrs

    def test_reporter_required_attributes(self) -> None:
        """Test reporter required attributes."""
        attrs = get_required_attributes(REPORTER_GROUP)
        assert "name" in attrs
        assert "report" in attrs

    def test_unknown_group_returns_empty(self) -> None:
        """Test that unknown group returns empty list."""
        attrs = get_required_attributes("unknown.group")
        assert attrs == []


class TestGetRequiredMethods:
    """Tests for get_required_methods function."""

    def test_adapter_required_methods(self) -> None:
        """Test adapter required methods."""
        methods = get_required_methods(ADAPTER_GROUP)
        assert "execute" in methods
        assert "stream_events" in methods

    def test_evaluator_required_methods(self) -> None:
        """Test evaluator required methods."""
        methods = get_required_methods(EVALUATOR_GROUP)
        assert "evaluate" in methods

    def test_reporter_required_methods(self) -> None:
        """Test reporter required methods."""
        methods = get_required_methods(REPORTER_GROUP)
        assert "report" in methods

    def test_unknown_group_returns_empty(self) -> None:
        """Test that unknown group returns empty list."""
        methods = get_required_methods("unknown.group")
        assert methods == []


class TestValidatePluginFull:
    """Tests for validate_plugin_full function."""

    def test_valid_plugin_returns_no_errors(self) -> None:
        """Test that valid plugin has no errors."""

        class ValidAdapter:
            atp_version: ClassVar[str] = "0.1.0"

            @property
            def adapter_type(self) -> str:
                return "valid"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        errors, version_error = validate_plugin_full(
            ValidAdapter, ADAPTER_GROUP, "0.1.0", "test"
        )
        assert errors == []
        assert version_error is None

    def test_invalid_plugin_returns_interface_errors(self) -> None:
        """Test that invalid plugin returns interface errors."""

        class InvalidAdapter:
            pass

        errors, version_error = validate_plugin_full(
            InvalidAdapter, ADAPTER_GROUP, "0.1.0", "test"
        )
        assert len(errors) > 0
        assert version_error is None

    def test_version_incompatible_returns_version_error(self) -> None:
        """Test that version incompatible plugin returns version error."""

        class VersionIncompatible:
            atp_version: ClassVar[str] = "99.0.0"

            @property
            def adapter_type(self) -> str:
                return "test"

            async def execute(self, request: ATPRequest) -> ATPResponse:
                return MagicMock()

            def stream_events(self, request: ATPRequest) -> Generator[Any, None, None]:
                yield MagicMock()

        errors, version_error = validate_plugin_full(
            VersionIncompatible, ADAPTER_GROUP, "0.1.0", "test"
        )
        assert errors == []
        assert version_error is not None


class TestUnknownGroup:
    """Tests for validation with unknown groups."""

    def test_validate_unknown_group(self) -> None:
        """Test that unknown group returns error."""

        class SomePlugin:
            pass

        errors = validate_plugin(SomePlugin, "unknown.group", "test")
        assert len(errors) == 1
        assert "Unknown plugin group" in errors[0]


class TestMinATPVersion:
    """Tests for MIN_ATP_VERSION constant."""

    def test_min_atp_version_is_valid_semver(self) -> None:
        """Test that MIN_ATP_VERSION is valid semver."""
        parts = MIN_ATP_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()
