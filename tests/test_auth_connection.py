import os

import pytest
import requests

from src.authentication import get_auth_headers

BASE_URL = "https://external-api.kalshi.com"
PATH = "/trade-api/v2/portfolio/balance"

@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_KALSHI_INTEGRATION_TESTS") != "1",
    reason="Set RUN_KALSHI_INTEGRATION_TESTS=1 to test live authentication.",
)
def test_auth_connection():
    headers = get_auth_headers("GET", PATH)
    response = requests.get(
        BASE_URL + PATH,
        headers=headers,
        timeout=15,
    )
    assert response.status_code == 200, (
        f"Kalshi authentication check returned HTTP {response.status_code}"
    )
