"""Tests for NL test generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from atp.generator.nl_generator import (
    NLTestGenerator,
    OpenAICompatibleClient,
    _clean_yaml_response,
    _create_default_client,
)
from atp.loader.models import TestSuite

# Sample valid YAML that the LLM might return
VALID_SUITE_YAML = """\
test_suite: "web-search-tests"
version: "1.0"
description: "Tests for web search and summarization"
tests:
  - id: "test-001"
    name: "Web search and summarize"
    tags:
      - search
      - summarization
    task:
      description: >
        Search the web for recent news about AI and
        create a summary document with key findings.
      expected_artifacts:
        - summary.txt
    constraints:
      max_steps: 10
      timeout_seconds: 120
    assertions:
      - type: artifact_exists
        config:
          path: summary.txt
      - type: behavior
        config:
          check: no_errors
"""

MULTI_TEST_YAML = """\
test_suite: "file-ops-tests"
version: "1.0"
description: "File operations test suite"
tests:
  - id: "test-001"
    name: "Create a text file"
    tags:
      - file
      - creation
    task:
      description: "Create a file named hello.txt with content Hello World"
      expected_artifacts:
        - hello.txt
    constraints:
      max_steps: 3
      timeout_seconds: 60
    assertions:
      - type: artifact_exists
        config:
          path: hello.txt

  - id: "test-002"
    name: "Delete a file"
    tags:
      - file
      - deletion
    task:
      description: "Delete the file named hello.txt"
    constraints:
      max_steps: 3
      timeout_seconds: 60
    assertions:
      - type: behavior
        config:
          check: no_errors
"""


class FakeLLMClient:
    """Fake LLM client that returns predetermined responses."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, prompt: str, system: str) -> str:
        self.calls.append((prompt, system))
        return self.response


class TestCleanYamlResponse:
    """Tests for _clean_yaml_response helper."""

    def test_plain_yaml(self) -> None:
        text = "test_suite: foo\ntests: []"
        assert _clean_yaml_response(text) == text

    def test_strips_markdown_yaml_fence(self) -> None:
        text = "```yaml\ntest_suite: foo\n```"
        assert _clean_yaml_response(text) == "test_suite: foo"

    def test_strips_plain_fence(self) -> None:
        text = "```\ntest_suite: foo\n```"
        assert _clean_yaml_response(text) == "test_suite: foo"

    def test_strips_whitespace(self) -> None:
        text = "  \n  test_suite: foo  \n  "
        assert _clean_yaml_response(text) == "test_suite: foo"

    def test_strips_fence_with_extra_whitespace(self) -> None:
        text = "  ```yaml\n  test_suite: foo\n  ```  "
        assert _clean_yaml_response(text) == "test_suite: foo"


class TestNLTestGenerator:
    """Tests for NLTestGenerator class."""

    def test_generate_single_test(self) -> None:
        """Test generating a suite from a single description."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        suite = gen.generate("test that the agent can search the web")

        assert isinstance(suite, TestSuite)
        assert len(suite.tests) >= 1
        assert suite.test_suite == "web-search-tests"
        assert len(client.calls) == 1

    def test_generate_multi_test(self) -> None:
        """Test generating a suite with multiple tests."""
        client = FakeLLMClient(MULTI_TEST_YAML)
        gen = NLTestGenerator(client=client)

        suite = gen.generate("test file creation and deletion")

        assert isinstance(suite, TestSuite)
        assert len(suite.tests) == 2
        assert suite.tests[0].id == "test-001"
        assert suite.tests[1].id == "test-002"

    def test_generate_passes_suite_name(self) -> None:
        """Test that suite_name is passed in the prompt."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        gen.generate(
            "test web search",
            suite_name="my-custom-suite",
        )

        prompt = client.calls[0][0]
        assert "my-custom-suite" in prompt

    def test_generate_passes_description_in_prompt(
        self,
    ) -> None:
        """Test that the description appears in the prompt."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        gen.generate("the agent should search and summarize")

        prompt = client.calls[0][0]
        assert "the agent should search and summarize" in prompt

    def test_generate_system_prompt_sent(self) -> None:
        """Test that the system prompt is sent to the LLM."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        gen.generate("test something")

        system = client.calls[0][1]
        assert "ATP" in system
        assert "YAML" in system

    def test_generate_strips_markdown_fences(self) -> None:
        """Test that markdown fences in LLM response are stripped."""
        fenced = f"```yaml\n{VALID_SUITE_YAML}\n```"
        client = FakeLLMClient(fenced)
        gen = NLTestGenerator(client=client)

        suite = gen.generate("test web search")

        assert isinstance(suite, TestSuite)
        assert len(suite.tests) >= 1

    def test_generate_invalid_yaml_raises(self) -> None:
        """Test that invalid YAML raises ValueError."""
        client = FakeLLMClient("this is not: valid: yaml: [")
        gen = NLTestGenerator(client=client)

        with pytest.raises(ValueError, match="not a valid"):
            gen.generate("test something")

    def test_generate_missing_required_fields_raises(
        self,
    ) -> None:
        """Test that YAML without required fields raises."""
        client = FakeLLMClient("foo: bar\nbaz: 123")
        gen = NLTestGenerator(client=client)

        with pytest.raises(ValueError, match="not a valid"):
            gen.generate("test something")

    def test_generate_from_file(self, tmp_path: Path) -> None:
        """Test generating from a description file."""
        desc_file = tmp_path / "desc.txt"
        desc_file.write_text("test web search capabilities")

        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        suite = gen.generate_from_file(desc_file)

        assert isinstance(suite, TestSuite)
        prompt = client.calls[0][0]
        assert "test web search capabilities" in prompt

    def test_generate_from_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        with pytest.raises(FileNotFoundError):
            gen.generate_from_file("/nonexistent/file.txt")

    def test_generate_from_file_empty(self, tmp_path: Path) -> None:
        """Test that empty file raises ValueError."""
        desc_file = tmp_path / "empty.txt"
        desc_file.write_text("")

        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        with pytest.raises(ValueError, match="empty"):
            gen.generate_from_file(desc_file)

    def test_to_yaml(self) -> None:
        """Test converting a suite to YAML string."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        suite = gen.generate("test web search")
        yaml_str = gen.to_yaml(suite)

        assert "test_suite" in yaml_str
        assert "tests" in yaml_str

    def test_save(self, tmp_path: Path) -> None:
        """Test saving a suite to file."""
        client = FakeLLMClient(VALID_SUITE_YAML)
        gen = NLTestGenerator(client=client)

        suite = gen.generate("test web search")
        output = tmp_path / "output.yaml"
        gen.save(suite, output)

        assert output.exists()
        content = output.read_text()
        assert "test_suite" in content

    def test_client_property_creates_default(self) -> None:
        """Test that client property creates default on access."""
        gen = NLTestGenerator()

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key"},
        ):
            client = gen.client
            assert isinstance(client, OpenAICompatibleClient)
            assert client.api_key == "test-key"

    def test_client_property_no_key_raises(self) -> None:
        """Test that missing API key raises ValueError."""
        gen = NLTestGenerator()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                _ = gen.client


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient."""

    def test_defaults(self) -> None:
        """Test default values."""
        client = OpenAICompatibleClient(api_key="sk-test")
        assert client.base_url == "https://api.openai.com/v1"
        assert client.model == "gpt-4o-mini"
        assert client.timeout == 60.0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        client = OpenAICompatibleClient(
            api_key="sk-test",
            base_url="http://localhost:11434/v1",
            model="llama3",
            timeout=120.0,
        )
        assert client.base_url == "http://localhost:11434/v1"
        assert client.model == "llama3"

    @patch("atp.generator.nl_generator.httpx.Client")
    def test_generate_sends_correct_request(self, mock_client_cls: MagicMock) -> None:
        """Test that generate sends the right HTTP request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "generated text"}}]
        }

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = OpenAICompatibleClient(api_key="sk-test")
        result = client.generate("hello", "system msg")

        assert result == "generated text"
        mock_client.post.assert_called_once()

        call_args = mock_client.post.call_args
        assert "/chat/completions" in call_args[0][0]

        payload = call_args[1]["json"]
        assert payload["model"] == "gpt-4o-mini"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

        headers = call_args[1]["headers"]
        assert "Bearer sk-test" in headers["Authorization"]

    @patch("atp.generator.nl_generator.httpx.Client")
    def test_generate_api_error_raises(self, mock_client_cls: MagicMock) -> None:
        """Test that non-200 response raises RuntimeError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "rate limited"

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        client = OpenAICompatibleClient(api_key="sk-test")

        with pytest.raises(RuntimeError, match="429"):
            client.generate("hello", "system")


class TestCreateDefaultClient:
    """Tests for _create_default_client."""

    def test_uses_atp_key(self) -> None:
        """Test that ATP_LLM_API_KEY takes precedence."""
        with patch.dict(
            "os.environ",
            {
                "ATP_LLM_API_KEY": "atp-key",
                "OPENAI_API_KEY": "openai-key",
            },
        ):
            client = _create_default_client()
            assert client.api_key == "atp-key"

    def test_falls_back_to_openai_key(self) -> None:
        """Test fallback to OPENAI_API_KEY."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "openai-key"},
            clear=True,
        ):
            client = _create_default_client()
            assert client.api_key == "openai-key"

    def test_no_key_raises(self) -> None:
        """Test that missing key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                _create_default_client()

    def test_custom_base_url(self) -> None:
        """Test custom base URL from env."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "key",
                "ATP_LLM_BASE_URL": "http://local:8080/v1",
            },
        ):
            client = _create_default_client()
            assert client.base_url == "http://local:8080/v1"

    def test_custom_model(self) -> None:
        """Test custom model from env."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "key",
                "ATP_LLM_MODEL": "gpt-4",
            },
        ):
            client = _create_default_client()
            assert client.model == "gpt-4"


class TestCLIFromDescription:
    """Tests for the from-description CLI command."""

    def test_cli_with_inline_description(self, tmp_path: Path) -> None:
        """Test CLI with inline description argument."""
        from click.testing import CliRunner

        from atp.cli.commands.generate import (
            generate_from_description,
        )

        client = FakeLLMClient(VALID_SUITE_YAML)
        output = tmp_path / "suite.yaml"

        runner = CliRunner()
        with patch(
            "atp.generator.nl_generator.NLTestGenerator",
            return_value=NLTestGenerator(client=client),
        ):
            result = runner.invoke(
                generate_from_description,
                [
                    "test web search",
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0
        assert output.exists()

    def test_cli_with_file_input(self, tmp_path: Path) -> None:
        """Test CLI with --file option."""
        from click.testing import CliRunner

        from atp.cli.commands.generate import (
            generate_from_description,
        )

        desc_file = tmp_path / "desc.txt"
        desc_file.write_text("test file operations")

        client = FakeLLMClient(MULTI_TEST_YAML)
        output = tmp_path / "suite.yaml"

        runner = CliRunner()
        with patch(
            "atp.generator.nl_generator.NLTestGenerator",
            return_value=NLTestGenerator(client=client),
        ):
            result = runner.invoke(
                generate_from_description,
                [
                    "--file",
                    str(desc_file),
                    "--output",
                    str(output),
                ],
            )

        assert result.exit_code == 0

    def test_cli_no_description_or_file(self) -> None:
        """Test CLI errors when no description provided."""
        from click.testing import CliRunner

        from atp.cli.commands.generate import (
            generate_from_description,
        )

        runner = CliRunner()
        result = runner.invoke(generate_from_description, [])

        assert result.exit_code != 0
        assert "description" in result.output.lower() or (result.exception is not None)

    def test_cli_stdout_output(self) -> None:
        """Test CLI outputs to stdout when no --output."""
        from click.testing import CliRunner

        from atp.cli.commands.generate import (
            generate_from_description,
        )

        client = FakeLLMClient(VALID_SUITE_YAML)

        runner = CliRunner()
        with patch(
            "atp.generator.nl_generator.NLTestGenerator",
            return_value=NLTestGenerator(client=client),
        ):
            result = runner.invoke(
                generate_from_description,
                ["test web search"],
            )

        assert result.exit_code == 0
        assert "test_suite" in result.output

    def test_cli_custom_suite_name(self, tmp_path: Path) -> None:
        """Test CLI with custom --name option."""
        from click.testing import CliRunner

        from atp.cli.commands.generate import (
            generate_from_description,
        )

        client = FakeLLMClient(VALID_SUITE_YAML)

        runner = CliRunner()
        with patch(
            "atp.generator.nl_generator.NLTestGenerator",
            return_value=NLTestGenerator(client=client),
        ):
            result = runner.invoke(
                generate_from_description,
                [
                    "test web search",
                    "--name",
                    "my-suite",
                ],
            )

        assert result.exit_code == 0
        assert "my-suite" in result.output
