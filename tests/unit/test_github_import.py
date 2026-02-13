"""Tests for the GitHub import service."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from atp.dashboard.v2.services.github_import import (
    ParsedGitHubURL,
    build_raw_url,
    fetch_github_file,
    import_from_github,
    parse_github_url,
    validate_suite_yaml,
)

# ---------------------------------------------------------------------------
# Valid test suite YAML for validation tests
# ---------------------------------------------------------------------------

VALID_SUITE_YAML = """\
test_suite: imported-suite
version: "1.0"
tests:
  - id: t1
    name: Test One
    task:
      description: "Do something"
"""

INVALID_SUITE_YAML = """\
not_a_suite: true
random_key: value
"""

INVALID_YAML = """\
: : broken: [yaml
"""


# ---------------------------------------------------------------------------
# parse_github_url tests
# ---------------------------------------------------------------------------


class TestParseGitHubURL:
    """Tests for parse_github_url."""

    def test_simple_repo_url(self) -> None:
        """Parse simple repo URL with defaults."""
        result = parse_github_url("https://github.com/owner/repo")

        assert result.owner == "owner"
        assert result.repo == "repo"
        assert result.ref == "main"
        assert result.path == ""

    def test_blob_url(self) -> None:
        """Parse URL with blob path."""
        url = "https://github.com/owner/repo/blob/main/tests/suite.yaml"
        result = parse_github_url(url)

        assert result.owner == "owner"
        assert result.repo == "repo"
        assert result.ref == "main"
        assert result.path == "tests/suite.yaml"

    def test_tree_url(self) -> None:
        """Parse URL with tree path."""
        url = "https://github.com/owner/repo/tree/develop/src"
        result = parse_github_url(url)

        assert result.owner == "owner"
        assert result.repo == "repo"
        assert result.ref == "develop"
        assert result.path == "src"

    def test_custom_branch(self) -> None:
        """Parse URL with custom branch in URL."""
        url = "https://github.com/org/project/blob/v2.0/path/file.yaml"
        result = parse_github_url(url)

        assert result.ref == "v2.0"
        assert result.path == "path/file.yaml"

    def test_default_branch_override(self) -> None:
        """Default branch parameter used when not in URL."""
        result = parse_github_url(
            "https://github.com/owner/repo",
            branch="develop",
        )

        assert result.ref == "develop"

    def test_default_path_override(self) -> None:
        """Default path parameter used when not in URL."""
        result = parse_github_url(
            "https://github.com/owner/repo",
            path="suite.yaml",
        )

        assert result.path == "suite.yaml"

    def test_url_branch_takes_precedence(self) -> None:
        """Branch in URL overrides the default parameter."""
        url = "https://github.com/owner/repo/blob/feature/file.yaml"
        result = parse_github_url(url, branch="main")

        assert result.ref == "feature"

    def test_invalid_url(self) -> None:
        """Raises ValueError for non-GitHub URLs."""
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            parse_github_url("https://gitlab.com/owner/repo")

    def test_invalid_url_no_host(self) -> None:
        """Raises ValueError for random strings."""
        with pytest.raises(ValueError, match="Invalid GitHub URL"):
            parse_github_url("not-a-url")

    def test_http_scheme(self) -> None:
        """Accepts http:// in addition to https://."""
        result = parse_github_url("http://github.com/owner/repo")
        assert result.owner == "owner"
        assert result.repo == "repo"


# ---------------------------------------------------------------------------
# build_raw_url tests
# ---------------------------------------------------------------------------


class TestBuildRawURL:
    """Tests for build_raw_url."""

    def test_basic_url(self) -> None:
        """Builds correct raw.githubusercontent.com URL."""
        parsed = ParsedGitHubURL(
            owner="owner", repo="repo", ref="main", path="file.yaml"
        )
        url = build_raw_url(parsed)

        assert url == ("https://raw.githubusercontent.com/owner/repo/main/file.yaml")

    def test_nested_path(self) -> None:
        """Handles nested paths correctly."""
        parsed = ParsedGitHubURL(
            owner="org", repo="project", ref="v1.0", path="tests/suite.yaml"
        )
        url = build_raw_url(parsed)

        assert url == (
            "https://raw.githubusercontent.com/org/project/v1.0/tests/suite.yaml"
        )


# ---------------------------------------------------------------------------
# fetch_github_file tests
# ---------------------------------------------------------------------------


class TestFetchGitHubFile:
    """Tests for fetch_github_file (mocked HTTP)."""

    @pytest.mark.anyio()
    async def test_successful_fetch(self) -> None:
        """Successfully fetches file content."""
        parsed = ParsedGitHubURL(
            owner="owner", repo="repo", ref="main", path="suite.yaml"
        )
        mock_response = httpx.Response(200, text="test content")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await fetch_github_file(parsed)

        assert result.path == "suite.yaml"
        assert result.content == "test content"

    @pytest.mark.anyio()
    async def test_404_raises(self) -> None:
        """Raises ValueError on 404."""
        parsed = ParsedGitHubURL(
            owner="owner", repo="repo", ref="main", path="missing.yaml"
        )
        mock_response = httpx.Response(404, text="Not Found")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="File not found"):
                await fetch_github_file(parsed)

    @pytest.mark.anyio()
    async def test_401_raises(self) -> None:
        """Raises ValueError on 401."""
        parsed = ParsedGitHubURL(
            owner="owner", repo="private", ref="main", path="suite.yaml"
        )
        mock_response = httpx.Response(401, text="Unauthorized")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Authentication required"):
                await fetch_github_file(parsed)

    @pytest.mark.anyio()
    async def test_403_raises(self) -> None:
        """Raises ValueError on 403."""
        parsed = ParsedGitHubURL(
            owner="owner", repo="repo", ref="main", path="suite.yaml"
        )
        mock_response = httpx.Response(403, text="Forbidden")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(ValueError, match="Access denied"):
                await fetch_github_file(parsed)

    @pytest.mark.anyio()
    async def test_token_sent_in_headers(self) -> None:
        """Token is included in request headers."""
        parsed = ParsedGitHubURL(
            owner="owner", repo="repo", ref="main", path="suite.yaml"
        )
        mock_response = httpx.Response(200, text="content")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            await fetch_github_file(parsed, token="gh_test123")

            call_kwargs = mock_client.get.call_args
            headers = call_kwargs.kwargs.get(
                "headers", call_kwargs[1].get("headers", {})
            )
            assert headers["Authorization"] == "token gh_test123"


# ---------------------------------------------------------------------------
# validate_suite_yaml tests
# ---------------------------------------------------------------------------


class TestValidateSuiteYAML:
    """Tests for validate_suite_yaml."""

    def test_valid_suite(self) -> None:
        """Valid suite YAML passes validation."""
        result = validate_suite_yaml(VALID_SUITE_YAML)

        assert isinstance(result, dict)
        assert result["test_suite"] == "imported-suite"

    def test_invalid_yaml_syntax(self) -> None:
        """Invalid YAML syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid YAML"):
            validate_suite_yaml(INVALID_YAML)

    def test_invalid_suite_structure(self) -> None:
        """Valid YAML but invalid suite raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ATP test suite"):
            validate_suite_yaml(INVALID_SUITE_YAML)

    def test_non_dict_yaml(self) -> None:
        """Non-dict YAML content raises ValueError."""
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_suite_yaml("- just\n- a\n- list\n")


# ---------------------------------------------------------------------------
# import_from_github integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestImportFromGitHub:
    """Tests for the high-level import_from_github function."""

    @pytest.mark.anyio()
    async def test_successful_import(self) -> None:
        """Full successful import flow."""
        mock_response = httpx.Response(200, text=VALID_SUITE_YAML)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await import_from_github(
                url="https://github.com/owner/repo/blob/main/suite.yaml",
            )

        assert result.success is True
        assert result.suite_content is not None
        assert result.suite_content["test_suite"] == "imported-suite"
        assert result.files_imported == ["suite.yaml"]

    @pytest.mark.anyio()
    async def test_no_path_returns_error(self) -> None:
        """Missing path returns a descriptive error."""
        result = await import_from_github(
            url="https://github.com/owner/repo",
        )

        assert result.success is False
        assert result.error is not None
        assert "No file path" in result.error

    @pytest.mark.anyio()
    async def test_non_yaml_path_returns_error(self) -> None:
        """Non-YAML file path returns error."""
        result = await import_from_github(
            url="https://github.com/owner/repo",
            path="README.md",
        )

        assert result.success is False
        assert result.error is not None
        assert "YAML file" in result.error

    @pytest.mark.anyio()
    async def test_invalid_url_returns_error(self) -> None:
        """Invalid URL returns error."""
        result = await import_from_github(
            url="https://notgithub.com/owner/repo",
            path="suite.yaml",
        )

        assert result.success is False
        assert result.error is not None
        assert "Invalid GitHub URL" in result.error

    @pytest.mark.anyio()
    async def test_file_not_found_returns_error(self) -> None:
        """404 from GitHub returns descriptive error."""
        mock_response = httpx.Response(404, text="Not Found")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await import_from_github(
                url="https://github.com/owner/repo",
                path="missing.yaml",
            )

        assert result.success is False
        assert "File not found" in (result.error or "")

    @pytest.mark.anyio()
    async def test_invalid_yaml_returns_error(self) -> None:
        """Invalid YAML content returns validation error."""
        mock_response = httpx.Response(200, text=INVALID_YAML)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await import_from_github(
                url="https://github.com/owner/repo",
                path="bad.yaml",
            )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.anyio()
    async def test_timeout_returns_error(self) -> None:
        """Timeout returns descriptive error."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await import_from_github(
                url="https://github.com/owner/repo",
                path="suite.yaml",
            )

        assert result.success is False
        assert "timed out" in (result.error or "").lower()

    @pytest.mark.anyio()
    async def test_connect_error_returns_error(self) -> None:
        """Connection error returns descriptive error."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("connection refused")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await import_from_github(
                url="https://github.com/owner/repo",
                path="suite.yaml",
            )

        assert result.success is False
        assert "connect" in (result.error or "").lower()

    @pytest.mark.anyio()
    async def test_custom_branch_and_path(self) -> None:
        """Branch and path parameters are used correctly."""
        mock_response = httpx.Response(200, text=VALID_SUITE_YAML)

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            result = await import_from_github(
                url="https://github.com/owner/repo",
                branch="develop",
                path="tests/suite.yaml",
            )

        assert result.success is True
        # Verify the URL was constructed with correct branch/path
        call_args = mock_client.get.call_args
        url_called = call_args[0][0]
        assert "develop" in url_called
        assert "tests/suite.yaml" in url_called
