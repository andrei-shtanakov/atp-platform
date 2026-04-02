"""Regression tests for public API surface of each package.

These tests ensure that public exports don't change unexpectedly.
If a test fails, it means the public API changed — update the test
intentionally after confirming the change is desired.
"""


def test_protocol_exports() -> None:
    """atp.protocol exports are stable."""
    from atp import protocol

    expected = {
        "PROTOCOL_VERSION",
        "SUPPORTED_VERSIONS",
        "ATPRequest",
        "ATPResponse",
        "ATPEvent",
        "Task",
        "Context",
        "Metrics",
        "ArtifactFile",
        "ArtifactStructured",
        "ArtifactReference",
        "ResponseStatus",
        "EventType",
        "ToolCallPayload",
        "LLMRequestPayload",
        "ReasoningPayload",
        "ErrorPayload",
        "ProgressPayload",
    }
    actual = set(protocol.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"


def test_adapters_exports() -> None:
    """atp.adapters exports are stable."""
    from atp import adapters

    expected = {
        "AgentAdapter",
        "AdapterConfig",
        "track_response_cost",
        "HTTPAdapter",
        "HTTPAdapterConfig",
        "ContainerAdapter",
        "ContainerAdapterConfig",
        "ContainerResources",
        "CLIAdapter",
        "CLIAdapterConfig",
        "LangGraphAdapter",
        "LangGraphAdapterConfig",
        "CrewAIAdapter",
        "CrewAIAdapterConfig",
        "AutoGenAdapter",
        "AutoGenAdapterConfig",
        "AzureOpenAIAdapter",
        "AzureOpenAIAdapterConfig",
        "MCPAdapter",
        "MCPAdapterConfig",
        "MCPTool",
        "MCPResource",
        "MCPPrompt",
        "MCPServerInfo",
        "BedrockAdapter",
        "BedrockAdapterConfig",
        "VertexAdapter",
        "VertexAdapterConfig",
        "AdapterRegistry",
        "get_registry",
        "create_adapter",
        "AdapterError",
        "AdapterTimeoutError",
        "AdapterConnectionError",
        "AdapterResponseError",
        "AdapterNotFoundError",
        "SDKAdapter",
        "SDKAdapterConfig",
    }
    actual = set(adapters.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"


def test_dashboard_exports() -> None:
    """atp.dashboard exports are stable."""
    from atp import dashboard

    expected = {
        "app",
        "create_app",
        "run_server",
        "Base",
        "Database",
        "get_database",
        "init_database",
        "set_database",
        "Agent",
        "Artifact",
        "EvaluationResult",
        "RunResult",
        "ScoreComponent",
        "SuiteExecution",
        "TestExecution",
        "User",
        "ResultStorage",
    }
    actual = set(dashboard.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"


def test_core_exports() -> None:
    """atp.core exports are stable."""
    from atp import core

    expected = {
        # Settings
        "ATPSettings",
        "DashboardSettings",
        "LLMSettings",
        "LoggingSettings",
        "RunnerSettings",
        "generate_example_config",
        "generate_json_schema",
        "get_cached_settings",
        "get_settings",
        # Logging
        "bind_context",
        "configure_logging",
        "configure_logging_from_settings",
        "correlation_context",
        "generate_correlation_id",
        "get_correlation_id",
        "get_logger",
        "get_module_log_level",
        "reset_logging",
        "set_correlation_id",
        "set_module_log_level",
        # Telemetry
        "TelemetrySettings",
        "add_exporter_to_provider",
        "add_span_event",
        "configure_telemetry",
        "create_adapter_span",
        "create_evaluator_span",
        "create_test_span",
        "ensure_debug_exporter",
        "extract_trace_context",
        "get_current_span",
        "get_debug_exporter",
        "get_telemetry_settings",
        "get_tracer",
        "inject_trace_context",
        "record_exception",
        "reset_telemetry",
        "set_adapter_response_attributes",
        "set_evaluator_result_attributes",
        "set_span_attribute",
        "set_span_attributes",
        "set_test_result_attributes",
        "span",
        # Metrics
        "ATPMetrics",
        "MetricsSettings",
        "configure_metrics",
        "generate_metrics",
        "get_metrics",
        "get_metrics_settings",
        "get_registry",
        "record_adapter_duration",
        "record_adapter_error",
        "record_evaluator_duration",
        "record_llm_call",
        "record_test_execution",
        "reset_metrics",
    }
    actual = set(core.__all__)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing exports: {missing}"
    assert not extra, f"Unexpected new exports: {extra}"
