"""Unit tests for JSON Schema generation."""

import pytest

from atp.protocol.schema import (
    generate_all_schemas,
    generate_event_schema,
    generate_request_schema,
    generate_response_schema,
)


class TestSchemaGeneration:
    """Tests for JSON Schema generation."""

    def test_generate_request_schema(self) -> None:
        """Test generating request schema."""
        schema = generate_request_schema()
        assert schema is not None
        assert "properties" in schema
        assert "version" in schema["properties"]
        assert "task_id" in schema["properties"]
        assert "task" in schema["properties"]
        assert "constraints" in schema["properties"]

    def test_generate_response_schema(self) -> None:
        """Test generating response schema."""
        schema = generate_response_schema()
        assert schema is not None
        assert "properties" in schema
        assert "version" in schema["properties"]
        assert "task_id" in schema["properties"]
        assert "status" in schema["properties"]
        assert "artifacts" in schema["properties"]
        assert "metrics" in schema["properties"]

    def test_generate_event_schema(self) -> None:
        """Test generating event schema."""
        schema = generate_event_schema()
        assert schema is not None
        assert "properties" in schema
        assert "version" in schema["properties"]
        assert "task_id" in schema["properties"]
        assert "timestamp" in schema["properties"]
        assert "sequence" in schema["properties"]
        assert "event_type" in schema["properties"]
        assert "payload" in schema["properties"]

    def test_generate_all_schemas(self) -> None:
        """Test generating all schemas."""
        schemas = generate_all_schemas()
        assert schemas is not None
        assert "ATPRequest" in schemas
        assert "ATPResponse" in schemas
        assert "ATPEvent" in schemas
        assert isinstance(schemas["ATPRequest"], dict)
        assert isinstance(schemas["ATPResponse"], dict)
        assert isinstance(schemas["ATPEvent"], dict)

    def test_request_schema_required_fields(self) -> None:
        """Test request schema has required fields."""
        schema = generate_request_schema()
        assert "required" in schema
        assert "task_id" in schema["required"]
        assert "task" in schema["required"]

    def test_response_schema_required_fields(self) -> None:
        """Test response schema has required fields."""
        schema = generate_response_schema()
        assert "required" in schema
        assert "task_id" in schema["required"]
        assert "status" in schema["required"]

    def test_event_schema_required_fields(self) -> None:
        """Test event schema has required fields."""
        schema = generate_event_schema()
        assert "required" in schema
        assert "task_id" in schema["required"]
        assert "sequence" in schema["required"]
        assert "event_type" in schema["required"]
        assert "payload" in schema["required"]

    def test_schemas_are_json_serializable(self) -> None:
        """Test that generated schemas are JSON serializable."""
        import json

        schemas = generate_all_schemas()
        for schema_name, schema in schemas.items():
            try:
                json.dumps(schema)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Schema {schema_name} is not JSON serializable: {e}")

    def test_schema_has_definitions(self) -> None:
        """Test that schemas include model definitions."""
        schema = generate_request_schema()
        # Pydantic v2 uses $defs instead of definitions
        assert "$defs" in schema or "definitions" in schema
