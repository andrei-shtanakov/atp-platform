"""Unit tests for YAML writer module."""

import tempfile
from pathlib import Path

import pytest

from atp.generator.core import TestGenerator, TestSuiteData
from atp.generator.writer import (
    YAMLWriter,
    _clean_dict,
    _clean_list,
    _constraints_to_dict,
    _defaults_to_dict,
    _scoring_to_dict,
)
from atp.loader.models import (
    AgentConfig,
    Assertion,
    Constraints,
    ScoringWeights,
    TaskDefinition,
    TestDefaults,
    TestDefinition,
    TestSuite,
)


class TestCleanDict:
    """Tests for _clean_dict helper function."""

    def test_removes_none_values(self) -> None:
        """Test that None values are removed."""
        data = {"a": 1, "b": None, "c": "hello"}
        result = _clean_dict(data)
        assert result == {"a": 1, "c": "hello"}

    def test_removes_empty_dicts(self) -> None:
        """Test that empty nested dicts are removed."""
        data = {"a": 1, "b": {}, "c": {"d": None}}
        result = _clean_dict(data)
        assert result == {"a": 1}

    def test_removes_empty_lists(self) -> None:
        """Test that empty lists are removed."""
        data = {"a": 1, "b": [], "c": [1, 2]}
        result = _clean_dict(data)
        assert result == {"a": 1, "c": [1, 2]}

    def test_nested_cleaning(self) -> None:
        """Test recursive cleaning of nested structures."""
        data = {
            "a": {"b": None, "c": 1},
            "d": [{"e": None}, {"f": 2}],
        }
        result = _clean_dict(data)
        assert result == {"a": {"c": 1}, "d": [{"f": 2}]}

    def test_preserves_zero_and_false(self) -> None:
        """Test that zero and False values are preserved."""
        data = {"a": 0, "b": False, "c": None}
        result = _clean_dict(data)
        assert result == {"a": 0, "b": False}


class TestCleanList:
    """Tests for _clean_list helper function."""

    def test_removes_none_values(self) -> None:
        """Test that None values are removed from lists."""
        data = [1, None, 2, None, 3]
        result = _clean_list(data)
        assert result == [1, 2, 3]

    def test_cleans_nested_dicts(self) -> None:
        """Test cleaning of nested dicts in lists."""
        data = [{"a": None, "b": 1}, {"c": None}]
        result = _clean_list(data)
        assert result == [{"b": 1}]


class TestScoringToDict:
    """Tests for _scoring_to_dict helper function."""

    def test_default_scoring_returns_empty(self) -> None:
        """Test that default scoring weights return empty dict."""
        scoring = ScoringWeights()
        result = _scoring_to_dict(scoring)
        assert result == {}

    def test_non_default_scoring(self) -> None:
        """Test non-default scoring weights are included."""
        scoring = ScoringWeights(
            quality_weight=0.5,
            completeness_weight=0.3,
            efficiency_weight=0.2,
            cost_weight=0.0,
        )
        result = _scoring_to_dict(scoring)
        assert result == {"quality_weight": 0.5, "cost_weight": 0.0}

    def test_partial_non_default(self) -> None:
        """Test partially non-default scoring."""
        scoring = ScoringWeights(quality_weight=0.6)
        result = _scoring_to_dict(scoring)
        assert result == {"quality_weight": 0.6}


class TestConstraintsToDict:
    """Tests for _constraints_to_dict helper function."""

    def test_default_constraints_returns_minimal(self) -> None:
        """Test that default constraints return minimal dict."""
        constraints = Constraints()
        result = _constraints_to_dict(constraints)
        # Only timeout_seconds has a default of 300, which we skip
        assert result == {}

    def test_non_default_constraints(self) -> None:
        """Test non-default constraints are included."""
        constraints = Constraints(
            max_steps=10,
            max_tokens=1000,
            timeout_seconds=60,
            allowed_tools=["tool1", "tool2"],
            budget_usd=0.50,
        )
        result = _constraints_to_dict(constraints)
        assert result == {
            "max_steps": 10,
            "max_tokens": 1000,
            "timeout_seconds": 60,
            "allowed_tools": ["tool1", "tool2"],
            "budget_usd": 0.50,
        }

    def test_partial_constraints(self) -> None:
        """Test partially filled constraints."""
        constraints = Constraints(max_steps=5)
        result = _constraints_to_dict(constraints)
        assert result == {"max_steps": 5}


class TestDefaultsToDict:
    """Tests for _defaults_to_dict helper function."""

    def test_default_defaults_returns_empty(self) -> None:
        """Test that default TestDefaults return empty dict."""
        defaults = TestDefaults()
        result = _defaults_to_dict(defaults)
        assert result == {}

    def test_non_default_values(self) -> None:
        """Test non-default TestDefaults values."""
        defaults = TestDefaults(runs_per_test=3, timeout_seconds=120)
        result = _defaults_to_dict(defaults)
        assert result == {"runs_per_test": 3, "timeout_seconds": 120}

    def test_with_constraints(self) -> None:
        """Test defaults with constraints."""
        defaults = TestDefaults(
            constraints=Constraints(max_steps=10),
        )
        result = _defaults_to_dict(defaults)
        assert result == {"constraints": {"max_steps": 10}}


class TestYAMLWriterToYaml:
    """Tests for YAMLWriter.to_yaml() method."""

    def test_minimal_suite(self) -> None:
        """Test YAML output for minimal suite."""
        suite = TestSuiteData(name="minimal_suite")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Do something"),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        assert "test_suite: minimal_suite" in yaml_content
        assert "version:" in yaml_content
        assert "tests:" in yaml_content
        assert "id: test-001" in yaml_content
        assert "name: Test 1" in yaml_content
        assert "description: Do something" in yaml_content

    def test_suite_with_description(self) -> None:
        """Test YAML output includes suite description."""
        suite = TestSuiteData(
            name="my_suite",
            description="A comprehensive test suite",
        )
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        assert "description: A comprehensive test suite" in yaml_content

    def test_suite_with_agents(self) -> None:
        """Test YAML output includes agents."""
        suite = TestSuiteData(name="suite_with_agents")
        suite.agents.append(
            AgentConfig(
                name="test-agent",
                type="http",
                config={"endpoint": "http://localhost:8000"},
            )
        )
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        assert "agents:" in yaml_content
        assert "name: test-agent" in yaml_content
        assert "type: http" in yaml_content
        assert "endpoint: http://localhost:8000" in yaml_content

    def test_suite_with_tags(self) -> None:
        """Test YAML output includes test tags."""
        suite = TestSuiteData(name="suite_with_tags")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                tags=["smoke", "critical"],
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        assert "tags:" in yaml_content
        assert "smoke" in yaml_content
        assert "critical" in yaml_content

    def test_suite_with_assertions(self) -> None:
        """Test YAML output includes assertions."""
        suite = TestSuiteData(name="suite_with_assertions")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
                assertions=[
                    Assertion(type="artifact_exists", config={"path": "file.txt"}),
                    Assertion(type="behavior", config={"check": "no_errors"}),
                ],
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        assert "assertions:" in yaml_content
        assert "type: artifact_exists" in yaml_content
        assert "path: file.txt" in yaml_content
        assert "type: behavior" in yaml_content
        assert "check: no_errors" in yaml_content

    def test_suite_with_constraints(self) -> None:
        """Test YAML output includes non-default constraints."""
        suite = TestSuiteData(name="suite_with_constraints")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
                constraints=Constraints(max_steps=5, timeout_seconds=60),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        assert "constraints:" in yaml_content
        assert "max_steps: 5" in yaml_content
        assert "timeout_seconds: 60" in yaml_content

    def test_excludes_none_values(self) -> None:
        """Test that None values are not included in output."""
        suite = TestSuiteData(name="suite", description=None)
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                description=None,
                task=TaskDefinition(description="Task", input_data=None),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        # Count description occurrences - should be only the task description
        lines = yaml_content.split("\n")
        description_lines = [line for line in lines if "description:" in line]
        assert len(description_lines) == 1  # Only task description

    def test_excludes_default_constraints(self) -> None:
        """Test that default constraint values are excluded."""
        suite = TestSuiteData(name="suite")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
                constraints=Constraints(),  # All defaults
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        # Default constraints should not appear
        assert "timeout_seconds: 300" not in yaml_content

    def test_indentation_format(self) -> None:
        """Test that YAML has proper indentation."""
        suite = TestSuiteData(name="suite")
        suite.agents.append(
            AgentConfig(name="agent1", type="http", config={"key": "value"})
        )
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                tags=["tag1"],
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        # Check that sequences are properly indented
        lines = yaml_content.split("\n")

        # Find tests section and verify structure
        assert any(line.startswith("tests:") for line in lines)
        assert any(
            line.startswith("  - id:") or line.startswith("    - id:") for line in lines
        )

    def test_test_suite_model_input(self) -> None:
        """Test that TestSuite model can be serialized."""
        test_suite = TestSuite(
            test_suite="model_suite",
            version="2.0",
            description="From TestSuite model",
            tests=[
                TestDefinition(
                    id="test-001",
                    name="Test 1",
                    task=TaskDefinition(description="Task"),
                )
            ],
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(test_suite)

        assert "test_suite: model_suite" in yaml_content
        assert "version: '2.0'" in yaml_content or 'version: "2.0"' in yaml_content

    def test_invalid_input_type(self) -> None:
        """Test that invalid input type raises TypeError."""
        writer = YAMLWriter()

        with pytest.raises(TypeError, match="Expected TestSuiteData or TestSuite"):
            writer.to_yaml({"invalid": "dict"})  # type: ignore


class TestYAMLWriterSave:
    """Tests for YAMLWriter.save() method."""

    def test_save_creates_file(self) -> None:
        """Test that save creates a YAML file."""
        suite = TestSuiteData(name="save_test")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output.yaml"
            writer.save(suite, file_path)

            assert file_path.exists()
            content = file_path.read_text()
            assert "test_suite: save_test" in content

    def test_save_creates_parent_directories(self) -> None:
        """Test that save creates parent directories if needed."""
        suite = TestSuiteData(name="nested_test")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "nested" / "dir" / "output.yaml"
            writer.save(suite, file_path)

            assert file_path.exists()

    def test_save_accepts_string_path(self) -> None:
        """Test that save accepts string paths."""
        suite = TestSuiteData(name="string_path_test")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(description="Task"),
            )
        )

        writer = YAMLWriter()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = f"{tmpdir}/output.yaml"
            writer.save(suite, file_path)

            assert Path(file_path).exists()


class TestTestGeneratorYamlMethods:
    """Tests for TestGenerator.to_yaml() and save() methods."""

    def test_to_yaml_method(self) -> None:
        """Test TestGenerator.to_yaml() method."""
        generator = TestGenerator()
        suite = generator.create_suite("generator_suite", "Test suite")
        test = generator.create_custom_test("test-001", "Test 1", "Do something")
        suite = generator.add_test(suite, test)

        yaml_content = generator.to_yaml(suite)

        assert "test_suite: generator_suite" in yaml_content
        assert "id: test-001" in yaml_content

    def test_save_method(self) -> None:
        """Test TestGenerator.save() method."""
        generator = TestGenerator()
        suite = generator.create_suite("generator_suite")
        test = generator.create_custom_test("test-001", "Test 1", "Do something")
        suite = generator.add_test(suite, test)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "output.yaml"
            generator.save(suite, file_path)

            assert file_path.exists()
            content = file_path.read_text()
            assert "test_suite: generator_suite" in content


class TestRoundTrip:
    """Tests for YAML round-trip (generate -> save -> load)."""

    def test_basic_round_trip(self) -> None:
        """Test that a suite can be saved and loaded back."""
        from atp.loader.loader import TestLoader

        generator = TestGenerator()
        suite = generator.create_suite(
            "round_trip_suite",
            description="Round trip test",
            version="1.0",
        )
        suite = generator.add_agent(
            suite, "test-agent", "http", {"endpoint": "http://localhost:8000"}
        )
        test = generator.create_custom_test(
            "test-001",
            "Test 1",
            "Do something important",
            constraints=Constraints(max_steps=5, timeout_seconds=60),
            assertions=[Assertion(type="behavior", config={"check": "no_errors"})],
            tags=["smoke", "critical"],
        )
        suite = generator.add_test(suite, test)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "round_trip.yaml"
            generator.save(suite, file_path)

            # Load back
            loader = TestLoader()
            loaded_suite = loader.load_file(file_path)

            # Verify
            assert loaded_suite.test_suite == "round_trip_suite"
            assert loaded_suite.description == "Round trip test"
            assert len(loaded_suite.agents) == 1
            assert loaded_suite.agents[0].name == "test-agent"
            assert len(loaded_suite.tests) == 1
            assert loaded_suite.tests[0].id == "test-001"
            assert loaded_suite.tests[0].name == "Test 1"
            assert loaded_suite.tests[0].constraints.max_steps == 5
            assert "smoke" in loaded_suite.tests[0].tags

    def test_complex_round_trip(self) -> None:
        """Test round trip with complex suite."""
        from atp.loader.loader import TestLoader

        generator = TestGenerator()
        suite = generator.create_suite(
            "complex_suite",
            description="A complex test suite with multiple tests",
        )

        # Add multiple agents
        suite = generator.add_agent(
            suite, "http-agent", "http", {"endpoint": "http://localhost:8000"}
        )
        suite = generator.add_agent(
            suite, "cli-agent", "cli", {"command": "python agent.py"}
        )

        # Add multiple tests
        for i in range(3):
            test = generator.create_custom_test(
                f"test-{i:03d}",
                f"Test {i}",
                f"Task description for test {i}",
                tags=[f"tag{i}", "common"],
                assertions=[
                    Assertion(type="artifact_exists", config={"path": f"file{i}.txt"}),
                ],
            )
            suite = generator.add_test(suite, test)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "complex.yaml"
            generator.save(suite, file_path)

            # Load back
            loader = TestLoader()
            loaded_suite = loader.load_file(file_path)

            # Verify
            assert loaded_suite.test_suite == "complex_suite"
            assert len(loaded_suite.agents) == 2
            assert len(loaded_suite.tests) == 3
            for i, test in enumerate(loaded_suite.tests):
                assert test.id == f"test-{i:03d}"
                assert f"tag{i}" in test.tags


class TestMultilineStrings:
    """Tests for multiline string handling."""

    def test_multiline_description(self) -> None:
        """Test that multiline descriptions are handled properly."""
        suite = TestSuiteData(name="multiline_suite")
        suite.tests.append(
            TestDefinition(
                id="test-001",
                name="Test 1",
                task=TaskDefinition(
                    description="Line 1\nLine 2\nLine 3",
                ),
            )
        )

        writer = YAMLWriter()
        yaml_content = writer.to_yaml(suite)

        # Multiline strings should be formatted properly
        assert "Line 1" in yaml_content
        assert "Line 2" in yaml_content
        assert "Line 3" in yaml_content

    def test_multiline_preserves_content(self) -> None:
        """Test that multiline content is preserved in round trip."""
        from atp.loader.loader import TestLoader

        multiline_desc = "Step 1: Do this\nStep 2: Do that\nStep 3: Done"

        generator = TestGenerator()
        suite = generator.create_suite("multiline_test")
        test = generator.create_custom_test(
            "test-001", "Multiline Test", multiline_desc
        )
        suite = generator.add_test(suite, test)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "multiline.yaml"
            generator.save(suite, file_path)

            loader = TestLoader()
            loaded = loader.load_file(file_path)

            # Content should be preserved (whitespace handling may vary)
            assert "Step 1" in loaded.tests[0].task.description
            assert "Step 2" in loaded.tests[0].task.description
            assert "Step 3" in loaded.tests[0].task.description
