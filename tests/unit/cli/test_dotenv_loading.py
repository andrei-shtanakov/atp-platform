"""Tests for .env file loading in CLI entrypoint."""

import os
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv


class TestDotenvLoading:
    def test_loads_env_file(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_DOTENV_VAR=from_dotenv\n")

        # Clear if exists
        os.environ.pop("TEST_DOTENV_VAR", None)

        load_dotenv(env_file, override=False)
        assert os.environ.get("TEST_DOTENV_VAR") == "from_dotenv"

        # Cleanup
        os.environ.pop("TEST_DOTENV_VAR", None)

    def test_shell_takes_priority(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_DOTENV_VAR=from_dotenv\n")

        with patch.dict(os.environ, {"TEST_DOTENV_VAR": "from_shell"}):
            load_dotenv(env_file, override=False)
            assert os.environ["TEST_DOTENV_VAR"] == "from_shell"

    def test_missing_env_file_no_error(self, tmp_path: Path) -> None:
        # load_dotenv silently ignores missing files
        result = load_dotenv(tmp_path / ".env.nonexistent", override=False)
        assert result is False
