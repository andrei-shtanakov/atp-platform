"""Test generator module for ATP test suites."""

from atp.generator.core import TestGenerator, TestSuiteData
from atp.generator.templates import (
    TemplateRegistry,
    TestTemplate,
    extract_variables,
    get_template_variables,
    substitute_in_assertion,
    substitute_variables,
)

__all__ = [
    "TestGenerator",
    "TestSuiteData",
    "TestTemplate",
    "TemplateRegistry",
    "substitute_variables",
    "substitute_in_assertion",
    "extract_variables",
    "get_template_variables",
]
