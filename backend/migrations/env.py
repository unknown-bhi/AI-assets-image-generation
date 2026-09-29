"""
Alembic environment configuration.

Ye file:
1. .env se DATABASE_URL uthata hai (single source of truth)
2. Models import karta hai (autogenerate ke liye)
3. Async migrations chalata hai

Rule: Naya model banane pe yahan uska import ZAROOR add karo,
      warna alembic autogenerate us table ko detect nahi karega.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# ============ PATH SETUP ============
# backend/ folder ko sys.path mein daalo taake `app` import ho sake
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============ APP IMPORTS ============
from app.config import settings          # noqa: E402
from app.database import Base            # noqa: E402

# ⚠️ SAARE MODELS YAHAN IMPORT KARO
# Jab bhi naya model banao, yahan import add karo, warna autogenerate
# us table ko miss kar dega.
from app.modules.auth.model import User              # noqa: E402
from app.modules.generation.model import Generation  # noqa: E402


# ============ ALEMBIC CONFIG ============
config = context.config

# Logging setup
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Alembic ko batao models kahan hain (autogenerate ke liye)
target_metadata = Base.metadata


# ============ OFFLINE MODE ============
def run_migrations_offline() -> None:
    """
    Offline mode: sirf SQL script generate karo, DB se connect na karo.
    `.env` se URL uthate hain.
    """
    url = settings.DATABASE_URL

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ============ ONLINE MODE (async) ============
def do_run_migrations(connection: Connection) -> None:
    """Actual migration chalane wala function."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Async engine banao aur migrations chalao.

    ⭐ Yahan .env se URL uthate hain aur config section mein inject karte hain.
    Ye line `alembic.ini` mein hardcoded URL ki zaroorat khatam karti hai.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = settings.DATABASE_URL

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Async entry point."""
    asyncio.run(run_async_migrations())


# ============ MAIN ============
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()