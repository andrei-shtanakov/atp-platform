"""CLI command: atp push — upload YAML test suites to remote server."""

from __future__ import annotations

from pathlib import Path

import click
import httpx

from atp.cli.commands.remote import (
    EXIT_FAILURE,
    EXIT_PARTIAL,
    EXIT_SUCCESS,
    file_sha256,
    find_yaml_files,
    load_manifest,
    now_iso,
    resolve_auth_headers,
    resolve_server_url,
    save_manifest,
)


@click.command(name="push")
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--server", help="ATP server URL")
@click.option("--api-key", help="API key for authentication")
@click.option("--force", is_flag=True, help="Re-upload even if suite exists")
@click.option("--dry-run", is_flag=True, help="Show what would be pushed")
def push_command(
    files: tuple[Path, ...],
    server: str | None,
    api_key: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    """Upload YAML test suite files to remote ATP server."""
    if not files:
        raise click.ClickException("No files specified")

    # Expand directories to YAML files
    all_files: list[Path] = []
    for f in files:
        if f.is_dir():
            all_files.extend(find_yaml_files(f))
        elif f.suffix in (".yaml", ".yml"):
            all_files.append(f)

    if not all_files:
        raise click.ClickException("No YAML files found")

    # Resolve server
    first_dir = all_files[0].parent
    server_url = resolve_server_url(server, first_dir)
    if not server_url:
        raise click.ClickException(
            "No server URL. Use --server, ATP_SERVER env var, or .atp-sync.json"
        )

    if dry_run:
        click.echo("Dry run — no changes will be made.")
        for f in all_files:
            click.echo(f"  push {f.name}")
        raise SystemExit(EXIT_SUCCESS)

    # Resolve auth
    headers = resolve_auth_headers(api_key=api_key, server_url=server_url)

    click.echo(f"Pushing {len(all_files)} file(s) to {server_url}...")

    succeeded = 0
    failed = 0

    with httpx.Client(timeout=30.0, headers=headers) as client:
        for filepath in all_files:
            result = _push_file(client, server_url, filepath, force)
            if result:
                succeeded += 1
                # Update manifest
                _update_manifest(filepath, server_url, result)
            else:
                failed += 1

    click.echo(f"\n{succeeded} succeeded, {failed} failed")

    if failed == len(all_files):
        raise SystemExit(EXIT_FAILURE)
    elif failed > 0:
        raise SystemExit(EXIT_PARTIAL)
    raise SystemExit(EXIT_SUCCESS)


def _push_file(
    client: httpx.Client,
    server_url: str,
    filepath: Path,
    force: bool,
) -> dict | None:
    """Push a single file. Returns suite info dict or None on failure."""
    name = filepath.name
    content = filepath.read_bytes()

    resp = client.post(
        f"{server_url}/api/suite-definitions/upload",
        files={"file": (name, content, "application/yaml")},
    )

    if resp.status_code == 201:
        data = resp.json()
        suite = data.get("suite", {})
        validation = data.get("validation", {})
        warnings = validation.get("warnings", [])
        suite_id = suite.get("id", "?")
        msg = f"created (id={suite_id})"
        if warnings:
            msg += f", {len(warnings)} warning(s)"
        click.echo(f"  \u2713 {name} \u2192 {msg}")
        return {"suite_id": suite_id}

    if resp.status_code == 409:
        if force:
            # TODO: delete and re-upload in sync_cmd, for push just warn
            click.echo(
                f"  \u26a0 {name} \u2192 already exists (use atp sync for updates)"
            )
        else:
            click.echo(f"  \u26a0 {name} \u2192 already exists (skip)")
        return None

    if resp.status_code in (400, 413, 422):
        data = resp.json()
        detail = data.get("detail", data)
        if isinstance(detail, dict):
            validation = detail.get("validation", {})
            errors = validation.get("errors", [])
        else:
            errors = [str(detail)]
        click.echo(f"  \u2717 {name} \u2192 {len(errors)} validation error(s)")
        for err in errors:
            click.echo(f"    - {err}")
        return None

    click.echo(f"  \u2717 {name} \u2192 HTTP {resp.status_code}")
    return None


def _update_manifest(filepath: Path, server_url: str, result: dict) -> None:
    """Update manifest file after successful push."""
    directory = filepath.parent
    manifest = load_manifest(directory)
    manifest["server"] = server_url
    manifest["files"][filepath.name] = {
        "sha256": file_sha256(filepath),
        "suite_id": result["suite_id"],
        "synced_at": now_iso(),
    }
    save_manifest(directory, manifest)
