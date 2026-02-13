"""Natural language test generation using LLM."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from atp.generator.core import TestGenerator
from atp.generator.writer import YAMLWriter
from atp.loader.models import TestSuite

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a test suite generator for the ATP (Agent Test Platform).
Your job is to convert natural language descriptions into valid \
ATP test suite YAML.

The YAML must follow this structure exactly:
- test_suite: string (name of the suite)
- version: "1.0"
- description: string (optional)
- tests: list of test objects, each with:
  - id: string (unique, e.g. "test-001")
  - name: string (human-readable)
  - tags: list of strings (optional)
  - task:
      description: string (what the agent should do)
      expected_artifacts: list of strings (optional)
  - constraints: (optional)
      max_steps: integer (optional)
      timeout_seconds: integer (optional)
  - assertions: list of assertion objects (optional), each with:
      - type: string (one of: artifact_exists, artifact_contains, \
behavior, llm_eval)
      - config: object with type-specific keys

Rules:
- Generate ONLY valid YAML. No markdown fences, no extra text.
- Each test must have a unique id in format "test-NNN".
- Task descriptions must be clear and specific.
- Add relevant tags for categorization.
- Add appropriate assertions based on the task.
- Use artifact_exists when files should be created.
- Use behavior with check: no_errors for basic checks.
- Use llm_eval for quality/correctness checks.
- Set reasonable constraints (max_steps, timeout_seconds).
"""

USER_PROMPT_TEMPLATE = """\
Generate an ATP test suite YAML from the following description:

{description}

Generate a suite named "{suite_name}" with appropriate tests.
Output ONLY the YAML content, nothing else.
"""


class LLMClient(Protocol):
    """Protocol for LLM API clients."""

    def generate(self, prompt: str, system: str) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            system: The system prompt.

        Returns:
            Generated text response.
        """
        ...


@dataclass
class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs.

    Uses httpx to call any OpenAI-compatible endpoint.

    Attributes:
        api_key: API key for authentication.
        base_url: Base URL for the API endpoint.
        model: Model name to use.
        timeout: Request timeout in seconds.
    """

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout: float = 60.0

    def generate(self, prompt: str, system: str) -> str:
        """Generate text using OpenAI-compatible chat API.

        Args:
            prompt: The user prompt.
            system: The system prompt.

        Returns:
            Generated text from the LLM.

        Raises:
            RuntimeError: If the API call fails.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM API error ({response.status_code}): {response.text}"
            )

        data = response.json()
        return data["choices"][0]["message"]["content"]


def _create_default_client() -> OpenAICompatibleClient:
    """Create a default LLM client from environment variables.

    Reads configuration from environment:
        - ATP_LLM_API_KEY or OPENAI_API_KEY: API key
        - ATP_LLM_BASE_URL: Base URL (default: OpenAI)
        - ATP_LLM_MODEL: Model name (default: gpt-4o-mini)

    Returns:
        Configured OpenAICompatibleClient.

    Raises:
        ValueError: If no API key is found.
    """
    api_key = os.environ.get("ATP_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "No LLM API key found. Set ATP_LLM_API_KEY or "
            "OPENAI_API_KEY environment variable."
        )

    base_url = os.environ.get("ATP_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("ATP_LLM_MODEL", "gpt-4o-mini")

    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def _clean_yaml_response(text: str) -> str:
    """Strip markdown fences and leading/trailing whitespace.

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned YAML string.
    """
    text = text.strip()
    if text.startswith("```yaml"):
        text = text[len("```yaml") :]
    elif text.startswith("```"):
        text = text[len("```") :]
    if text.endswith("```"):
        text = text[: -len("```")]
    return text.strip()


class NLTestGenerator:
    """Generate ATP test suites from natural language descriptions.

    Uses an LLM to convert human-readable descriptions into valid
    ATP test suite YAML, then validates the output.

    Args:
        client: LLM client to use. If None, creates a default client
            from environment variables.

    Example:
        >>> client = OpenAICompatibleClient(api_key="sk-...")
        >>> gen = NLTestGenerator(client=client)
        >>> suite = gen.generate("test that the agent can create files")
        >>> print(gen.to_yaml(suite))
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client
        self._generator = TestGenerator()
        self._writer = YAMLWriter()

    @property
    def client(self) -> LLMClient:
        """Get the LLM client, creating default if needed."""
        if self._client is None:
            self._client = _create_default_client()
        return self._client

    def generate(
        self,
        description: str,
        suite_name: str = "generated-suite",
    ) -> TestSuite:
        """Generate a test suite from a natural language description.

        Args:
            description: Natural language description of tests to
                generate. Can be a single sentence or a paragraph.
            suite_name: Name for the generated suite.

        Returns:
            Validated TestSuite object.

        Raises:
            ValueError: If the generated YAML is invalid.
            RuntimeError: If the LLM API call fails.
        """
        prompt = USER_PROMPT_TEMPLATE.format(
            description=description,
            suite_name=suite_name,
        )

        raw_response = self.client.generate(prompt, SYSTEM_PROMPT)
        yaml_text = _clean_yaml_response(raw_response)

        return self._parse_and_validate(yaml_text)

    def generate_from_file(
        self,
        file_path: str | Path,
        suite_name: str = "generated-suite",
    ) -> TestSuite:
        """Generate a test suite from a description file.

        Args:
            file_path: Path to a text file with the description.
            suite_name: Name for the generated suite.

        Returns:
            Validated TestSuite object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the generated YAML is invalid.
        """
        path = Path(file_path)
        description = path.read_text(encoding="utf-8").strip()
        if not description:
            raise ValueError(f"Description file is empty: {file_path}")
        return self.generate(description, suite_name=suite_name)

    def to_yaml(self, suite: TestSuite) -> str:
        """Convert a test suite to YAML string.

        Args:
            suite: TestSuite to serialize.

        Returns:
            YAML formatted string.
        """
        return self._writer.to_yaml(suite)

    def save(
        self,
        suite: TestSuite,
        output_path: str | Path,
    ) -> None:
        """Save a test suite to a YAML file.

        Args:
            suite: TestSuite to save.
            output_path: Path to output file.
        """
        self._writer.save(suite, output_path)

    def _parse_and_validate(self, yaml_text: str) -> TestSuite:
        """Parse YAML text and validate as a TestSuite.

        Args:
            yaml_text: YAML string to parse.

        Returns:
            Validated TestSuite.

        Raises:
            ValueError: If parsing or validation fails.
        """
        from atp.loader.loader import TestLoader

        try:
            loader = TestLoader()
            return loader.load_string(yaml_text)
        except Exception as e:
            raise ValueError(
                f"Generated YAML is not a valid ATP test suite: {e}"
            ) from e
