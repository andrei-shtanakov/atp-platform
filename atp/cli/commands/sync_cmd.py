"""CLI command: atp sync — synchronize local YAML test suites with server."""

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


@click.command(name="sync")
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option("--server", help="ATP server URL")
@click.option("--api-key", help="API key for authentication")
@click.option("--dry-run", is_flag=True, help="Show what would happen")
def sync_command(
    directory: Path,
    server: str | None,
    api_key: str | None,
    dry_run: bool,
) -> None:
    """Synchronize a directory of YAML test suites with remote server."""
    server_url = resolve_server_url(server, directory)
    if not server_url:
        raise click.ClickException(
            "No server URL. Use --server, ATP_SERVER env var, or .atp-sync.json"
        )

    manifest = load_manifest(directory)
    manifest_files = manifest.get("files", {})
    local_files = find_yaml_files(directory)

    # Categorize files
    new_files: list[Path] = []
    changed_files: list[Path] = []
    unchanged_files: list[Path] = []
    deleted_names: list[str] = []

    local_names = {f.name for f in local_files}

    for f in local_files:
        entry = manifest_files.get(f.name)
        if entry is None:
            new_files.append(f)
        elif file_sha256(f) != entry.get("sha256"):
            changed_files.append(f)
        else:
            unchanged_files.append(f)

    for name in list(manifest_files.keys()):
        if name not in local_names:
            deleted_names.append(name)

    if dry_run:
        click.echo("Dry run — no changes will be made.")
        for f in new_files:
            click.echo(f"  push {f.name} (new)")
        for f in changed_files:
            click.echo(f"  push {f.name} (changed)")
        for f in unchanged_files:
            click.echo(f"  skip {f.name} (unchanged)")
        for name in deleted_names:
            click.echo(f"  warn {name} (removed locally)")
        raise SystemExit(EXIT_SUCCESS)

    click.echo(f"Syncing {directory} with {server_url}...")

    headers = resolve_auth_headers(api_key=api_key, server_url=server_url)
    created = 0
    updated = 0
    failed = 0

    with httpx.Client(timeout=30.0, headers=headers) as client:
        # Push new files
        for f in new_files:
            result = _upload_file(client, server_url, f)
            if result:
                manifest_files[f.name] = {
                    "sha256": file_sha256(f),
                    "suite_id": result["suite_id"],
                    "synced_at": now_iso(),
                }
                created += 1
                click.echo(
                    f"  \u2713 {f.name} \u2192 created (id={result['suite_id']})"
                )
            else:
                failed += 1

        # Push changed files (delete + re-upload)
        for f in changed_files:
            entry = manifest_files.get(f.name, {})
            old_id = entry.get("suite_id")

            # Delete old
            if old_id:
                del_resp = client.delete(f"{server_url}/api/suite-definitions/{old_id}")
                if del_resp.status_code not in (200, 204, 404):
                    click.echo(
                        f"  \u2717 {f.name} \u2192 failed to delete old "
                        f"(HTTP {del_resp.status_code})"
                    )
                    failed += 1
                    continue

            # Re-upload
            result = _upload_file(client, server_url, f)
            if result:
                manifest_files[f.name] = {
                    "sha256": file_sha256(f),
                    "suite_id": result["suite_id"],
                    "synced_at": now_iso(),
                }
                updated += 1
                click.echo(
                    f"  \u2713 {f.name} \u2192 updated (id={result['suite_id']})"
                )
            else:
                click.echo(
                    f"  \u2717 {f.name} \u2192 DELETE succeeded but upload failed!"
                    f"\n    Restore with: atp push {f.name} --force"
                )
                failed += 1

    # Handle deleted files
    for name in deleted_names:
        click.echo(f"  \u26a0 {name} \u2192 removed locally (cleared from manifest)")
        manifest_files.pop(name, None)

    # Unchanged
    for f in unchanged_files:
        click.echo(f"  \u2713 {f.name} \u2192 unchanged, skipped")

    # Save manifest
    manifest["server"] = server_url
    manifest["files"] = manifest_files
    save_manifest(directory, manifest)

    click.echo(
        f"\n{created} created, {updated} updated, "
        f"{len(unchanged_files)} unchanged, "
        f"{len(deleted_names)} removed locally"
    )

    if failed > 0 and created + updated == 0:
        raise SystemExit(EXIT_FAILURE)
    elif failed > 0:
        raise SystemExit(EXIT_PARTIAL)
    raise SystemExit(EXIT_SUCCESS)


def _upload_file(
    client: httpx.Client,
    server_url: str,
    filepath: Path,
) -> dict | None:
    """Upload a single file. Returns {"suite_id": N} or None."""
    resp = client.post(
        f"{server_url}/api/suite-definitions/upload",
        files={"file": (filepath.name, filepath.read_bytes(), "application/yaml")},
    )
    if resp.status_code == 201:
        data = resp.json()
        suite = data.get("suite", {})
        return {"suite_id": suite.get("id")}

    if resp.status_code in (400, 409, 413, 422):
        data = resp.json()
        detail = data.get("detail", data)
        if isinstance(detail, dict):
            errors = detail.get("validation", {}).get("errors", [])
        else:
            errors = [str(detail)]
        click.echo(f"  \u2717 {filepath.name} \u2192 {len(errors)} error(s)")
        for err in errors:
            click.echo(f"    - {err}")
    else:
        click.echo(f"  \u2717 {filepath.name} \u2192 HTTP {resp.status_code}")

    return None
