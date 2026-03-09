"""Alembic environment for ATP Analytics database."""

import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

from atp.analytics.models import AnalyticsBase

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = AnalyticsBase.metadata

# Allow override via environment variable
db_url = os.environ.get("ATP_ANALYTICS_DATABASE_URL")
if db_url:
    config.set_main_option("sqlalchemy.url", db_url)
elif not config.get_main_option("sqlalchemy.url"):
    db_path = Path.home() / ".atp" / "analytics.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
