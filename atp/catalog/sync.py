"""Sync builtin YAML catalog files into the database."""

from pathlib import Path

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from atp.catalog.repository import CatalogRepository

BUILTIN_DIR = Path(__file__).parent / "builtin"


def parse_catalog_yaml(content: str) -> tuple[dict, list[dict]]:
    """Parse a catalog YAML string into metadata and test list.

    Args:
        content: Raw YAML text of a catalog suite file.

    Returns:
        A tuple of (catalog_meta, tests) where catalog_meta is the dict
        under the ``catalog:`` key and tests is the list under ``tests:``.
    """
    data = yaml.safe_load(content)
    catalog_meta: dict = data.get("catalog", {})
    tests: list[dict] = data.get("tests", [])
    return catalog_meta, tests


async def sync_builtin_catalog(session: AsyncSession) -> None:
    """Scan the builtin directory and upsert all suites and tests into the DB.

    Iterates every subdirectory of the builtin catalog directory (each
    sub-directory corresponds to one category), then for each YAML file
    found inside it upserts the suite and its individual tests.

    Args:
        session: SQLAlchemy async session with an open transaction.
    """
    repo = CatalogRepository(session)

    for category_dir in sorted(BUILTIN_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        category_slug = category_dir.name
        # Derive a human-readable name from the slug
        category_name = category_slug.replace("-", " ").title()

        category = await repo.upsert_category(
            slug=category_slug,
            name=category_name,
        )

        for yaml_file in sorted(category_dir.glob("*.yaml")):
            raw_content = yaml_file.read_text(encoding="utf-8")
            catalog_meta, tests = parse_catalog_yaml(raw_content)

            suite_slug = catalog_meta.get("slug", yaml_file.stem)
            tags_list: list[str] = catalog_meta.get("tags", [])
            tags_payload: dict = {"tags": tags_list} if tags_list else {}

            suite_name: str = catalog_meta.get("name") or suite_slug
            suite = await repo.upsert_suite(
                category_id=category.id,
                slug=suite_slug,
                name=suite_name,
                author=catalog_meta.get("author", "curated"),
                source=catalog_meta.get("source", "builtin"),
                suite_yaml=raw_content,
                description=catalog_meta.get("description"),
                difficulty=catalog_meta.get("difficulty"),
                estimated_minutes=catalog_meta.get("estimated_minutes"),
                tags=tags_payload if tags_payload else None,
                version=str(catalog_meta.get("version", "1.0")),
            )

            for test_data in tests:
                test_slug: str = test_data.get("id", "")
                test_tags_list: list[str] = test_data.get("tags", [])
                test_tags: dict | None = (
                    {"tags": test_tags_list} if test_tags_list else None
                )
                task_block: dict = test_data.get("task", {})
                task_description: str = task_block.get("description", "")

                await repo.upsert_test(
                    suite_id=suite.id,
                    slug=test_slug,
                    name=test_data.get("name", test_slug),
                    task_description=task_description,
                    difficulty=catalog_meta.get("difficulty"),
                    tags=test_tags,
                )

    await session.commit()
