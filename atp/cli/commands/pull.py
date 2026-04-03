"""CLI command: atp pull — download test suites from remote server."""

from __future__ import annotations

import re
from pathlib import Path

import click
import httpx

from atp.cli.commands.remote import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    file_sha256,
    load_manifest,
    now_iso,
    resolve_auth_headers,
    resolve_server_url,
    save_manifest,
)


def _sanitize_filename(name: str) -> str:
    """Sanitize suite name to valid filename."""
    sanitized = re.sub(r"[^\w\-.]", "-", name).strip("-")
    return sanitized or "unnamed"


@click.command(name="pull")
@click.option("--server", help="ATP server URL")
@click.option("--api-key", help="API key for authentication")
@click.option(
    "--dir",
    "directory",
    type=click.Path(path_type=Path),
    default=".",
    help="Output directory",
)
@click.option("--id", "suite_id", type=int, help="Pull specific suite by ID")
@click.option(
    "--all",
    "pull_all",
    is_flag=True,
    default=True,
    help="Pull all suites (default)",
)
@click.option("--force", is_flag=True, help="Overwrite existing files")
def pull_command(
    server: str | None,
    api_key: str | None,
    directory: Path,
    suite_id: int | None,
    pull_all: bool,
    force: bool,
) -> None:
    """Download test suites from remote ATP server."""
    server_url = resolve_server_url(server, directory)
    if not server_url:
        raise click.ClickException(
            "No server URL. Use --server, ATP_SERVER env var, or .atp-sync.json"
        )

    directory.mkdir(parents=True, exist_ok=True)
    headers = resolve_auth_headers(api_key=api_key, server_url=server_url)

    click.echo(f"Pulling suites from {server_url}...")

    pulled = 0
    skipped = 0
    failed = 0

    with httpx.Client(timeout=30.0, headers=headers) as client:
        if suite_id:
            suites = [{"id": suite_id, "name": f"suite-{suite_id}"}]
        else:
            # List all suites
            resp = client.get(
                f"{server_url}/api/suite-definitions",
                params={"limit": 100, "offset": 0},
            )
            if resp.status_code != 200:
                raise click.ClickException(
                    f"Failed to list suites: HTTP {resp.status_code}"
                )
            suites = resp.json().get("items", [])

        used_filenames: set[str] = set()

        for suite_info in suites:
            sid = suite_info["id"]
            name = str(suite_info.get("name", f"suite-{sid}"))
            filename = _sanitize_filename(name) + ".yaml"

            # Handle filename collisions
            if filename in used_filenames:
                filename = f"{_sanitize_filename(name)}_{sid}.yaml"
            used_filenames.add(filename)

            target = directory / filename

            # Skip existing
            if target.exists() and not force:
                click.echo(f"  - skipped {filename} (exists, use --force)")
                skipped += 1
                continue

            # Export YAML
            resp = client.get(f"{server_url}/api/suite-definitions/{sid}/yaml")
            if resp.status_code != 200:
                click.echo(f"  \u2717 {filename} \u2192 HTTP {resp.status_code}")
                failed += 1
                continue

            data = resp.json()
            yaml_content = data.get("yaml_content", "")
            target.write_text(yaml_content)
            click.echo(f"  \u2713 {filename} (id={sid})")
            pulled += 1

            # Update manifest
            manifest = load_manifest(directory)
            manifest["server"] = server_url
            manifest["files"][filename] = {
                "sha256": file_sha256(target),
                "suite_id": sid,
                "synced_at": now_iso(),
            }
            save_manifest(directory, manifest)

    click.echo(f"\n{pulled} pulled, {skipped} skipped, {failed} failed")

    if failed > 0 and pulled == 0:
        raise SystemExit(EXIT_FAILURE)
    elif failed > 0:
        raise SystemExit(EXIT_PARTIAL)
    raise SystemExit(EXIT_SUCCESS)
