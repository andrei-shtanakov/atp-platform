"""Artifact evaluator for checking agent outputs."""

import re
from typing import Any

from atp.loader.models import Assertion, TestDefinition
from atp.protocol import ATPEvent, ATPResponse

from .base import EvalCheck, EvalResult, Evaluator


class ArtifactEvaluator(Evaluator):
    """
    Evaluator for artifact-related assertions.

    Supports the following assertion types:
    - artifact_exists: Check if artifact with given path exists
    - contains: Check if artifact contains a pattern (plain text or regex)
    - schema: Validate artifact data against JSON schema
    - sections: Check if artifact contains required sections
    """

    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "artifact"

    async def evaluate(
        self,
        task: TestDefinition,
        response: ATPResponse,
        trace: list[ATPEvent],
        assertion: Assertion,
    ) -> EvalResult:
        """
        Evaluate artifact assertions.

        Args:
            task: Test definition (unused for artifact checks).
            response: ATP Response containing artifacts.
            trace: Event trace (unused for artifact checks).
            assertion: Assertion to evaluate.

        Returns:
            EvalResult with check outcomes.
        """
        assertion_type = assertion.type
        config = assertion.config

        if assertion_type == "artifact_exists":
            check = self._check_artifact_exists(response, config)
        elif assertion_type == "contains":
            check = self._check_contains(response, config)
        elif assertion_type == "schema":
            check = self._check_schema(response, config)
        elif assertion_type == "sections":
            check = self._check_sections(response, config)
        else:
            check = self._create_check(
                name=f"unknown_{assertion_type}",
                passed=False,
                message=f"Unknown assertion type: {assertion_type}",
            )

        return self._create_result([check])

    def _check_artifact_exists(
        self, response: ATPResponse, config: dict[str, Any]
    ) -> EvalCheck:
        """Check if an artifact with the given path exists."""
        path = config.get("path", "")
        if not path:
            return self._create_check(
                name="artifact_exists",
                passed=False,
                message="No path specified in assertion config",
            )

        for artifact in response.artifacts:
            artifact_path = getattr(artifact, "path", None) or getattr(
                artifact, "name", None
            )
            if artifact_path == path:
                return self._create_check(
                    name="artifact_exists",
                    passed=True,
                    message=f"Artifact found: {path}",
                    details={"path": path, "found": True},
                )

        return self._create_check(
            name="artifact_exists",
            passed=False,
            message=f"Artifact not found: {path}",
            details={
                "path": path,
                "found": False,
                "available_artifacts": [
                    getattr(a, "path", None) or getattr(a, "name", None)
                    for a in response.artifacts
                ],
            },
        )

    def _check_contains(
        self, response: ATPResponse, config: dict[str, Any]
    ) -> EvalCheck:
        """Check if an artifact contains a pattern."""
        pattern = config.get("pattern", "")
        if not pattern:
            return self._create_check(
                name="contains",
                passed=False,
                message="No pattern specified in assertion config",
            )

        use_regex = config.get("regex", False)
        artifact_path = config.get("path")

        target_artifacts = response.artifacts
        if artifact_path:
            target_artifacts = [
                a
                for a in response.artifacts
                if (getattr(a, "path", None) or getattr(a, "name", None))
                == artifact_path
            ]
            if not target_artifacts:
                return self._create_check(
                    name="contains",
                    passed=False,
                    message=f"Artifact not found: {artifact_path}",
                    details={"path": artifact_path, "pattern": pattern},
                )

        for artifact in target_artifacts:
            content = self._get_artifact_content(artifact)
            if content is None:
                continue

            if use_regex:
                try:
                    if re.search(pattern, content):
                        return self._create_check(
                            name="contains",
                            passed=True,
                            message=f"Pattern found (regex): {pattern}",
                            details={
                                "pattern": pattern,
                                "regex": True,
                                "found": True,
                            },
                        )
                except re.error as e:
                    return self._create_check(
                        name="contains",
                        passed=False,
                        message=f"Invalid regex pattern: {e}",
                        details={"pattern": pattern, "regex_error": str(e)},
                    )
            else:
                if pattern in content:
                    return self._create_check(
                        name="contains",
                        passed=True,
                        message=f"Pattern found: {pattern}",
                        details={"pattern": pattern, "regex": False, "found": True},
                    )

        return self._create_check(
            name="contains",
            passed=False,
            message=f"Pattern not found: {pattern}",
            details={
                "pattern": pattern,
                "regex": use_regex,
                "found": False,
                "path": artifact_path,
            },
        )

    def _check_schema(self, response: ATPResponse, config: dict[str, Any]) -> EvalCheck:
        """Validate artifact data against a JSON schema."""
        schema = config.get("schema")
        if not schema:
            return self._create_check(
                name="schema",
                passed=False,
                message="No schema specified in assertion config",
            )

        artifact_path = config.get("path")

        target_artifacts = response.artifacts
        if artifact_path:
            target_artifacts = [
                a
                for a in response.artifacts
                if (getattr(a, "path", None) or getattr(a, "name", None))
                == artifact_path
            ]
            if not target_artifacts:
                return self._create_check(
                    name="schema",
                    passed=False,
                    message=f"Artifact not found: {artifact_path}",
                    details={"path": artifact_path},
                )

        for artifact in target_artifacts:
            data = getattr(artifact, "data", None)
            if data is None:
                continue

            validation_result = self._validate_against_schema(data, schema)
            if validation_result["valid"]:
                return self._create_check(
                    name="schema",
                    passed=True,
                    message="Schema validation passed",
                    details={"path": artifact_path, "valid": True},
                )
            else:
                return self._create_check(
                    name="schema",
                    passed=False,
                    message=f"Schema validation failed: {validation_result['error']}",
                    details={
                        "path": artifact_path,
                        "valid": False,
                        "error": validation_result["error"],
                    },
                )

        return self._create_check(
            name="schema",
            passed=False,
            message="No structured artifact found for schema validation",
            details={"path": artifact_path},
        )

    def _check_sections(
        self, response: ATPResponse, config: dict[str, Any]
    ) -> EvalCheck:
        """Check if artifact contains required sections."""
        sections = config.get("sections", [])
        if not sections:
            return self._create_check(
                name="sections",
                passed=False,
                message="No sections specified in assertion config",
            )

        artifact_path = config.get("path")

        target_artifacts = response.artifacts
        if artifact_path:
            target_artifacts = [
                a
                for a in response.artifacts
                if (getattr(a, "path", None) or getattr(a, "name", None))
                == artifact_path
            ]
            if not target_artifacts:
                return self._create_check(
                    name="sections",
                    passed=False,
                    message=f"Artifact not found: {artifact_path}",
                    details={"path": artifact_path, "sections": sections},
                )

        for artifact in target_artifacts:
            content = self._get_artifact_content(artifact)
            if content is None:
                continue

            missing_sections = []
            found_sections = []

            for section in sections:
                section_pattern = rf"(?:^|\n)#+\s*{re.escape(section)}"
                if re.search(section_pattern, content, re.IGNORECASE):
                    found_sections.append(section)
                elif section.lower() in content.lower():
                    found_sections.append(section)
                else:
                    missing_sections.append(section)

            if not missing_sections:
                return self._create_check(
                    name="sections",
                    passed=True,
                    message=f"All sections found: {', '.join(sections)}",
                    details={
                        "sections": sections,
                        "found": found_sections,
                        "missing": [],
                    },
                )

            return self._create_check(
                name="sections",
                passed=False,
                message=f"Missing sections: {', '.join(missing_sections)}",
                details={
                    "sections": sections,
                    "found": found_sections,
                    "missing": missing_sections,
                },
            )

        return self._create_check(
            name="sections",
            passed=False,
            message="No artifact content found for section check",
            details={"path": artifact_path, "sections": sections},
        )

    def _get_artifact_content(self, artifact: Any) -> str | None:
        """Extract content from an artifact."""
        if hasattr(artifact, "content") and artifact.content:
            return artifact.content
        if hasattr(artifact, "data") and artifact.data:
            import json

            return json.dumps(artifact.data)
        return None

    def _validate_against_schema(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate data against a JSON schema.

        Returns a dict with 'valid' (bool) and optionally 'error' (str).
        Uses jsonschema library if available, otherwise falls back to simple validation.
        """
        try:
            import jsonschema

            jsonschema.validate(data, schema)
            return {"valid": True}
        except ImportError:
            return self._simple_schema_validation(data, schema)
        except Exception as e:
            error_name = type(e).__name__
            if error_name == "ValidationError":
                return {"valid": False, "error": str(getattr(e, "message", str(e)))}
            elif error_name == "SchemaError":
                return {
                    "valid": False,
                    "error": f"Invalid schema: {getattr(e, 'message', str(e))}",
                }
            raise

    def _simple_schema_validation(
        self, data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Basic schema validation when jsonschema is not available."""
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in data:
                return {"valid": False, "error": f"Missing required field: {field}"}

        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get("type")
                if expected_type:
                    actual_type = type(data[field]).__name__
                    type_mapping = {
                        "string": "str",
                        "integer": "int",
                        "number": ("int", "float"),
                        "boolean": "bool",
                        "array": "list",
                        "object": "dict",
                    }
                    expected_python = type_mapping.get(expected_type, expected_type)
                    if isinstance(expected_python, tuple):
                        if actual_type not in expected_python:
                            err_msg = (
                                f"Field '{field}' expected type "
                                f"{expected_type}, got {actual_type}"
                            )
                            return {"valid": False, "error": err_msg}
                    elif actual_type != expected_python:
                        err_msg = (
                            f"Field '{field}' expected type "
                            f"{expected_type}, got {actual_type}"
                        )
                        return {"valid": False, "error": err_msg}

        return {"valid": True}
