"""JSON Schema for test suite validation."""

from typing import Any

# Reusable schema definitions for multi-agent configurations
COLLABORATION_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "max_turns": {"type": "integer", "minimum": 1, "default": 10},
        "turn_timeout_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 60,
        },
        "require_consensus": {"type": "boolean", "default": False},
        "allow_parallel_turns": {"type": "boolean", "default": False},
        "coordinator_agent": {"type": ["string", "null"]},
        "termination_condition": {
            "type": "string",
            "enum": ["all_complete", "any_complete", "consensus", "max_turns"],
            "default": "all_complete",
        },
    },
    "additionalProperties": False,
}

HANDOFF_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "handoff_trigger": {
            "type": "string",
            "enum": ["always", "on_success", "on_failure", "on_partial", "explicit"],
            "default": "always",
        },
        "context_accumulation": {
            "type": "string",
            "enum": ["append", "replace", "merge", "summary"],
            "default": "append",
        },
        "max_context_size": {"type": ["integer", "null"], "minimum": 1},
        "allow_backtrack": {"type": "boolean", "default": False},
        "final_agent_decides": {"type": "boolean", "default": True},
        "agent_timeout_seconds": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 120,
        },
        "continue_on_failure": {"type": "boolean", "default": False},
    },
    "additionalProperties": False,
}

COMPARISON_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "metrics": {
            "type": "array",
            "items": {"type": "string"},
            "default": ["quality", "speed", "cost"],
        },
        "determine_winner": {"type": "boolean", "default": True},
        "parallel_execution": {"type": "boolean", "default": True},
    },
    "additionalProperties": False,
}

TEST_SUITE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["test_suite", "tests"],
    "properties": {
        "test_suite": {"type": "string", "minLength": 1},
        "version": {"type": "string", "default": "1.0"},
        "description": {"type": "string"},
        "defaults": {
            "type": "object",
            "properties": {
                "runs_per_test": {"type": "integer", "minimum": 1, "default": 1},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 300},
                "scoring": {
                    "type": "object",
                    "properties": {
                        "quality_weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.4,
                        },
                        "completeness_weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.3,
                        },
                        "efficiency_weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.2,
                        },
                        "cost_weight": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.1,
                        },
                    },
                    "additionalProperties": False,
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "max_steps": {"type": "integer", "minimum": 1},
                        "max_tokens": {"type": "integer", "minimum": 1},
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 300,
                        },
                        "allowed_tools": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                        },
                        "budget_usd": {"type": "number", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "type": {"type": "string"},
                    "config": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
        "tests": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "name", "task"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "task": {
                        "type": "object",
                        "required": ["description"],
                        "properties": {
                            "description": {"type": "string", "minLength": 1},
                            "input_data": {"type": "object"},
                            "expected_artifacts": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "additionalProperties": False,
                    },
                    "constraints": {
                        "type": "object",
                        "properties": {
                            "max_steps": {"type": "integer", "minimum": 1},
                            "max_tokens": {"type": "integer", "minimum": 1},
                            "timeout_seconds": {
                                "type": "integer",
                                "minimum": 1,
                                "default": 300,
                            },
                            "allowed_tools": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                            "budget_usd": {"type": "number", "minimum": 0},
                        },
                        "additionalProperties": False,
                    },
                    "assertions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type"],
                            "properties": {
                                "type": {"type": "string", "minLength": 1},
                                "config": {"type": "object"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "scoring": {
                        "type": "object",
                        "properties": {
                            "quality_weight": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "completeness_weight": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "efficiency_weight": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "cost_weight": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "additionalProperties": False,
                    },
                    # Multi-agent fields
                    "agents": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "description": "List of agent names to run this test against",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["comparison", "collaboration", "handoff"],
                        "description": "Multi-agent execution mode",
                    },
                    "comparison_config": COMPARISON_CONFIG_SCHEMA,
                    "collaboration_config": COLLABORATION_CONFIG_SCHEMA,
                    "handoff_config": HANDOFF_CONFIG_SCHEMA,
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def validate_schema(data: dict[str, Any]) -> list[str]:
    """Validate data against JSON Schema.

    Args:
        data: Data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    try:
        import jsonschema

        validator = jsonschema.Draft7Validator(TEST_SUITE_SCHEMA)
        errors = []

        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{path}: {error.message}")

        return errors

    except ImportError:
        # If jsonschema is not available, skip validation
        # We'll rely on pydantic validation instead
        return []
