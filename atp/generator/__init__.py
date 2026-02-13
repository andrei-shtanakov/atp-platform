"""Test generator module for ATP test suites."""

from atp.generator.core import TestGenerator, TestSuiteData
from atp.generator.nl_generator import (
    NLTestGenerator,
    OpenAICompatibleClient,
)
from atp.generator.regression import (
    AnonymizationLevel,
    DataAnonymizer,
    Recording,
    RecordingSession,
    RecordingStatus,
    RegressionTestGenerator,
    TestDeduplicator,
    TestParameterizer,
    create_recording_session,
    load_recordings_from_file,
    save_recordings_to_file,
)
from atp.generator.templates import (
    TemplateRegistry,
    TestTemplate,
    extract_variables,
    get_template_variables,
    substitute_in_assertion,
    substitute_variables,
)
from atp.generator.trace_import import TraceImporter, TraceRecord
from atp.generator.writer import YAMLWriter

__all__ = [
    # Core generator
    "TestGenerator",
    "TestSuiteData",
    # NL generator
    "NLTestGenerator",
    "OpenAICompatibleClient",
    # Templates
    "TestTemplate",
    "TemplateRegistry",
    "substitute_variables",
    "substitute_in_assertion",
    "extract_variables",
    "get_template_variables",
    # Writer
    "YAMLWriter",
    # Trace import
    "TraceImporter",
    "TraceRecord",
    # Regression test generator
    "RegressionTestGenerator",
    "Recording",
    "RecordingSession",
    "RecordingStatus",
    "AnonymizationLevel",
    "DataAnonymizer",
    "TestDeduplicator",
    "TestParameterizer",
    "create_recording_session",
    "load_recordings_from_file",
    "save_recordings_to_file",
]
