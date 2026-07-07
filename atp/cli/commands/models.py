"""CLI commands for the ATP model catalog (`atp models`)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from atp.model_catalog import (
    CatalogError,
    load_catalog,
    read_template,
    resolve_catalog_path,
)


@click.group(name="models")
def models_command() -> None:
    """Manage the model catalog (which models this instance uses)."""


@models_command.command(name="init")
@click.option(
    "--path",
    "path",
    type=click.Path(path_type=Path),
    default=None,
    help="Target file to create (overrides $ATP_CATALOG / XDG).",
)
@click.option("--force", is_flag=True, help="Overwrite an existing catalog.")
def init_cmd(path: Path | None, force: bool) -> None:
    """Write a starter catalog to the resolved user-config path."""
    try:
        target = path if path is not None else resolve_catalog_path(must_exist=False)
    except CatalogError as exc:
        raise click.ClickException(str(exc)) from exc
    if target.exists() and not force:
        raise click.ClickException(
            f"{target} already exists; pass --force to overwrite"
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(read_template(), encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"cannot write {target}: {exc}") from exc
    click.echo(f"Wrote model catalog: {target}")
    click.echo("Edit it to add your models (the starter ships empty).")


@models_command.command(name="list")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format.",
)
def list_cmd(fmt: str) -> None:
    """List the models in the resolved catalog."""
    try:
        path = resolve_catalog_path(must_exist=True)
        catalog = load_catalog(path)
    except CatalogError as exc:
        raise click.ClickException(str(exc)) from exc
    models = catalog.models
    if fmt == "json":
        click.echo(
            json.dumps({name: m.model_dump() for name, m in models.items()}, indent=2)
        )
        return
    if not models:
        click.echo(f"No models defined yet — edit {path}")
        return
    for name, m in models.items():
        aliases = ", ".join(m.aliases) if m.aliases else "-"
        click.echo(f"{name:30s}  {m.vendor:12s}  {m.status:11s}  {aliases}")
