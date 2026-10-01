"""Generation endpoint tests."""

import pytest


@pytest.mark.asyncio
async def test_txt2img_queue(client, auth_headers):
    """Queue a txt2img generation."""
    response = await client.post(
        "/api/v1/generate/txt2img",
        headers=auth_headers,
        json={
            "prompt": "a cute cat",
            "width": 256,
            "height": 256,
            "steps": 10,
            "cfg_scale": 7.0,
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "pending"
    assert data["type"] == "txt2img"
    assert data["prompt"] == "a cute cat"
    assert "id" in data


@pytest.mark.asyncio
async def test_txt2img_unauthorized(client):
    """Txt2img without auth returns 401."""
    response = await client.post(
        "/api/v1/generate/txt2img",
        json={"prompt": "a cat"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_txt2img_invalid_dimensions(client, auth_headers):
    """Dimensions too large rejected."""
    response = await client.post(
        "/api/v1/generate/txt2img",
        headers=auth_headers,
        json={
            "prompt": "a cat",
            "width": 5000,
            "height": 5000,
        },
    )
    assert response.status_code == 422      # Validation error


@pytest.mark.asyncio
async def test_get_generation(client, auth_headers):
    """Get a generation by ID."""
    # Queue
    create_response = await client.post(
        "/api/v1/generate/txt2img",
        headers=auth_headers,
        json={"prompt": "a cat", "width": 256, "height": 256, "steps": 10},
    )
    gen_id = create_response.json()["id"]

    # Get
    response = await client.get(
        f"/api/v1/generate/{gen_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["id"] == gen_id


@pytest.mark.asyncio
async def test_get_generation_not_found(client, auth_headers):
    """Non-existent generation returns 404."""
    response = await client.get(
        "/api/v1/generate/999999",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_history(client, auth_headers):
    """Generation history endpoint."""
    response = await client.get(
        "/api/v1/generate/history",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data


@pytest.mark.asyncio
async def test_stats(client, auth_headers):
    """Stats endpoint."""
    response = await client.get(
        "/api/v1/generate/stats",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "quota" in data