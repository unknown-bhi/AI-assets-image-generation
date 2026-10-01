"""Authentication tests."""

import pytest


@pytest.mark.asyncio
async def test_register_success(client, unique_suffix):
    """Register a new user."""
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "username": f"user_{unique_suffix}",
            "email": f"user_{unique_suffix}@test.com",
            "password": "password123",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == f"user_{unique_suffix}"
    assert "id" in data
    assert "hashed_password" not in data      # Security check


@pytest.mark.asyncio
async def test_register_duplicate_username(client, unique_suffix):
    """Duplicate username should fail."""
    username = f"dup_{unique_suffix}"

    # First registration
    await client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": f"dup1_{unique_suffix}@test.com",
            "password": "password123",
        },
    )

    # Duplicate
    response = await client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": f"dup2_{unique_suffix}@test.com",
            "password": "password123",
        },
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_login_success(client, unique_suffix):
    """Login with valid credentials."""
    username = f"login_{unique_suffix}"

    await client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": f"{username}@test.com",
            "password": "password123",
        },
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "password123"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client, unique_suffix):
    """Login with wrong password fails."""
    username = f"wrongpw_{unique_suffix}"

    await client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": f"{username}@test.com",
            "password": "password123",
        },
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": "wrong"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_endpoint(client, auth_headers):
    """Get current user info."""
    response = await client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "username" in data
    assert "email" in data


@pytest.mark.asyncio
async def test_me_without_token(client):
    """Me endpoint without token returns 401."""
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_me_invalid_token(client):
    """Me endpoint with invalid token returns 401."""
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid_token"},
    )
    assert response.status_code == 401