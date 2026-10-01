"""History service tests — via API client."""

import pytest


@pytest.mark.asyncio
async def test_history_endpoint(client, auth_headers):
    """History endpoint returns proper structure."""
    response = await client.get(
        "/api/v1/generate/history",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data


@pytest.mark.asyncio
async def test_stats_endpoint(client, auth_headers):
    """Stats endpoint has all required fields."""
    response = await client.get(
        "/api/v1/generate/stats",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert "total" in data["stats"]
    assert "by_status" in data["stats"]
    assert "by_type" in data["stats"]