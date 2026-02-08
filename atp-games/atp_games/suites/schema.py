"""JSON Schema for game suite YAML validation."""

from typing import Any

GAME_SUITE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["type", "name", "game", "agents"],
    "properties": {
        "type": {"type": "string", "const": "game_suite"},
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "default": "1.0"},
        "extends": {"type": "string"},
        "game": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string", "minLength": 1},
                "variant": {
                    "type": "string",
                    "enum": ["one_shot", "repeated"],
                    "default": "one_shot",
                },
                "config": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "agents": {
            "type": "array",
            "minItems": 2,
            "items": {
                "type": "object",
                "required": ["name", "adapter"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "adapter": {
                        "type": "string",
                        "enum": [
                            "http",
                            "docker",
                            "cli",
                            "builtin",
                        ],
                    },
                    "endpoint": {"type": "string"},
                    "strategy": {"type": "string"},
                    "config": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "evaluation": {
            "type": "object",
            "properties": {
                "episodes": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string", "minLength": 1},
                            "weight": {
                                "type": "number",
                                "minimum": 0.0,
                                "default": 1.0,
                            },
                            "config": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                },
                "thresholds": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "reporting": {
            "type": "object",
            "properties": {
                "include_strategy_profile": {
                    "type": "boolean",
                    "default": True,
                },
                "include_payoff_matrix": {
                    "type": "boolean",
                    "default": True,
                },
                "include_round_by_round": {
                    "type": "boolean",
                    "default": False,
                },
                "export_formats": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def validate_game_suite_schema(data: dict[str, Any]) -> list[str]:
    """Validate data against the game suite JSON Schema.

    Args:
        data: Data to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    try:
        import jsonschema

        validator = jsonschema.Draft7Validator(GAME_SUITE_SCHEMA)
        errors = []

        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{path}: {error.message}")

        return errors

    except ImportError:
        return []
