"""GitHub import service for marketplace.

Fetches test suite YAML files from GitHub repositories and validates
them for import into the ATP marketplace.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx
import yaml

from atp.loader.loader import TestLoader

logger = logging.getLogger(__name__)

# Regex to parse GitHub URLs:
#   https://github.com/owner/repo
#   https://github.com/owner/repo/blob/branch/path/to/file.yaml
#   https://github.com/owner/repo/tree/branch/path
_GITHUB_URL_RE = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[^/]+)/"
    r"(?P<repo>[^/]+)"
    r"(?:/(?:blob|tree)/(?P<ref>[^/]+)(?:/(?P<path>.+))?)?"
)

# GitHub raw content URL template
_RAW_URL_TEMPLATE = "https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"

# Maximum file size to download (1 MB)
_MAX_FILE_SIZE = 1_048_576

# HTTP timeout in seconds
_HTTP_TIMEOUT = 30.0


@dataclass
class ParsedGitHubURL:
    """Parsed components of a GitHub URL.

    Attributes:
        owner: Repository owner.
        repo: Repository name.
        ref: Branch, tag, or commit SHA.
        path: File or directory path within the repo.
    """

    owner: str
    repo: str
    ref: str
    path: str


@dataclass
class GitHubFile:
    """A file fetched from GitHub.

    Attributes:
        path: File path within the repo.
        content: Raw file content.
    """

    path: str
    content: str


@dataclass
class ImportResult:
    """Result of a GitHub import operation.

    Attributes:
        success: Whether the import succeeded.
        suite_content: Parsed suite content dict (if successful).
        files_imported: List of file paths imported.
        error: Error message (if failed).
    """

    success: bool
    suite_content: dict | None = None
    files_imported: list[str] | None = None
    error: str | None = None


def parse_github_url(
    url: str,
    branch: str = "main",
    path: str = "",
) -> ParsedGitHubURL:
    """Parse a GitHub URL into components.

    Supports formats:
        - https://github.com/owner/repo
        - https://github.com/owner/repo/blob/branch/path/file.yaml
        - https://github.com/owner/repo/tree/branch/path

    Args:
        url: GitHub URL to parse.
        branch: Default branch (used when not in URL).
        path: Default path (used when not in URL).

    Returns:
        Parsed URL components.

    Raises:
        ValueError: If the URL is not a valid GitHub URL.
    """
    match = _GITHUB_URL_RE.match(url)
    if not match:
        raise ValueError(
            f"Invalid GitHub URL: {url}. Expected format: https://github.com/owner/repo"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    ref = match.group("ref") or branch
    url_path = match.group("path") or path

    return ParsedGitHubURL(owner=owner, repo=repo, ref=ref, path=url_path)


def build_raw_url(parsed: ParsedGitHubURL) -> str:
    """Build a raw.githubusercontent.com URL.

    Args:
        parsed: Parsed GitHub URL components.

    Returns:
        Raw content URL.
    """
    return _RAW_URL_TEMPLATE.format(
        owner=parsed.owner,
        repo=parsed.repo,
        ref=parsed.ref,
        path=parsed.path,
    )


async def fetch_github_file(
    parsed: ParsedGitHubURL,
    token: str | None = None,
) -> GitHubFile:
    """Fetch a single file from GitHub.

    Args:
        parsed: Parsed GitHub URL with file path.
        token: Optional GitHub token for private repos.

    Returns:
        GitHubFile with content.

    Raises:
        ValueError: If the file cannot be fetched.
    """
    raw_url = build_raw_url(parsed)
    headers: dict[str, str] = {
        "Accept": "application/vnd.github.v3.raw",
    }
    if token:
        headers["Authorization"] = f"token {token}"

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        response = await client.get(raw_url, headers=headers)

        if response.status_code == 404:
            raise ValueError(
                f"File not found: {parsed.path} "
                f"in {parsed.owner}/{parsed.repo} "
                f"(ref: {parsed.ref})"
            )
        if response.status_code == 401:
            raise ValueError(
                "Authentication required. "
                "Provide a GitHub token for private repositories."
            )
        if response.status_code == 403:
            raise ValueError("Access denied. Check your GitHub token permissions.")
        if response.status_code != 200:
            raise ValueError(
                f"GitHub returned status {response.status_code} for {parsed.path}"
            )
        if len(response.content) > _MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {len(response.content)} bytes (max: {_MAX_FILE_SIZE})"
            )

    return GitHubFile(path=parsed.path, content=response.text)


def validate_suite_yaml(content: str) -> dict:
    """Parse and validate YAML content as an ATP test suite.

    Args:
        content: Raw YAML string.

    Returns:
        Parsed suite content dict.

    Raises:
        ValueError: If YAML is invalid or not a valid test suite.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("YAML content must be a mapping (dict)")

    # Validate as ATP test suite using the loader
    loader = TestLoader()
    try:
        loader.load_string(content)
    except Exception as e:
        raise ValueError(f"Invalid ATP test suite: {e}") from e

    return data


async def import_from_github(
    url: str,
    branch: str = "main",
    path: str = "",
    token: str | None = None,
) -> ImportResult:
    """Import a test suite from a GitHub repository.

    Parses the URL, fetches the file, validates it, and returns
    the parsed content.

    Args:
        url: GitHub repository URL.
        branch: Branch/tag/commit to fetch from.
        path: Path to the YAML file within the repo.
        token: Optional GitHub token for private repos.

    Returns:
        ImportResult with suite content or error.
    """
    try:
        parsed = parse_github_url(url, branch=branch, path=path)

        if not parsed.path:
            return ImportResult(
                success=False,
                error="No file path specified. Provide a path to "
                "a YAML file (e.g., tests/suite.yaml).",
            )

        # Ensure path ends with .yaml or .yml
        if not parsed.path.endswith((".yaml", ".yml")):
            return ImportResult(
                success=False,
                error=f"Path must point to a YAML file (.yaml or .yml): {parsed.path}",
            )

        file = await fetch_github_file(parsed, token=token)
        suite_content = validate_suite_yaml(file.content)

        return ImportResult(
            success=True,
            suite_content=suite_content,
            files_imported=[file.path],
        )

    except ValueError as e:
        return ImportResult(success=False, error=str(e))
    except httpx.TimeoutException:
        return ImportResult(
            success=False,
            error="Request to GitHub timed out. Try again later.",
        )
    except httpx.ConnectError:
        return ImportResult(
            success=False,
            error="Cannot connect to GitHub. Check your network.",
        )
