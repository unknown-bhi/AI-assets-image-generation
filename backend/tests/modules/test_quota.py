"""Quota tests — via API client."""

import pytest


@pytest.mark.asyncio
async def test_quota_info(client, auth_headers):
    """Quota info in stats endpoint."""
    response = await client.get(
        "/api/v1/generate/stats",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "quota" in data
    quota = data["quota"]
    assert "used" in quota
    assert "limit" in quota
    assert "remaining" in quota
    assert "reset_at" in quota