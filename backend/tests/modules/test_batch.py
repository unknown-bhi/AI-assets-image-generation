"""Batch endpoint tests."""

import pytest


@pytest.mark.asyncio
async def test_create_batch(client, auth_headers):
    """Create a batch with multiple items."""
    response = await client.post(
        "/api/v1/generate/batch",
        headers=auth_headers,
        json={
            "name": "Test batch",
            "items": [
                {"type": "txt2img", "prompt": "a cat", "width": 256, "height": 256, "steps": 10},
                {"type": "txt2img", "prompt": "a dog", "width": 256, "height": 256, "steps": 10},
            ],
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "batch_id" in data
    assert data["total_items"] == 2
    assert len(data["items"]) == 2


@pytest.mark.asyncio
async def test_get_batch(client, auth_headers):
    """Get batch status."""
    # Create
    create_response = await client.post(
        "/api/v1/generate/batch",
        headers=auth_headers,
        json={
            "name": "Get test",
            "items": [{"type": "txt2img", "prompt": "a cat", "width": 256, "height": 256, "steps": 10}],
        },
    )
    batch_id = create_response.json()["batch_id"]

    # Get
    response = await client.get(
        f"/api/v1/generate/batch/{batch_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["batch_id"] == batch_id


@pytest.mark.asyncio
async def test_list_batches(client, auth_headers):
    """List batches."""
    response = await client.get(
        "/api/v1/generate/batches",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data


@pytest.mark.asyncio
async def test_batch_unauthorized(client):
    """Batch without auth returns 401."""
    response = await client.post(
        "/api/v1/generate/batch",
        json={"name": "test", "items": [{"prompt": "a cat"}]},
    )
    assert response.status_code == 401