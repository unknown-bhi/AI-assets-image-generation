"""
Pytest configuration and shared fixtures.

Key strategy:
- NullPool: no connection pooling (each test gets fresh connection)
- Single event loop per test via asyncio_mode=auto
- Override get_db to use test session
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import settings
from app.database import get_db
from app.main import app


# ============ TEST ENGINE (NullPool) ============
test_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    poolclass=NullPool,      # ⭐ No pooling
)

TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ============ OVERRIDE get_db DEPENDENCY ============
async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    """Override DB dependency to use test session."""
    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


app.dependency_overrides[get_db] = override_get_db


# ============ HTTP CLIENT ============
@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """HTTP test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as ac:
        yield ac


# ============ UNIQUE SUFFIX ============
@pytest.fixture
def unique_suffix() -> str:
    """Unique suffix for test data."""
    return uuid.uuid4().hex[:8]


# ============ AUTH TOKEN ============
@pytest_asyncio.fixture
async def auth_token(client: AsyncClient, unique_suffix: str) -> str:
    """Register + login, return access token."""
    username = f"testuser_{unique_suffix}"
    email = f"test_{unique_suffix}@test.com"
    password = "password123"

    # Register
    register_response = await client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": email,
            "password": password,
        },
    )
    assert register_response.status_code == 201, register_response.text

    # Login
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert login_response.status_code == 200, login_response.text

    return login_response.json()["access_token"]


# ============ AUTH HEADERS ============
@pytest_asyncio.fixture
async def auth_headers(auth_token: str) -> dict:
    """Authorization headers."""
    return {"Authorization": f"Bearer {auth_token}"}