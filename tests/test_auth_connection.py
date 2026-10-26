import requests

from src.authentication import get_auth_headers

BASE_URL = "https://external-api.kalshi.com"
PATH = "/trade-api/v2/portfolio/balance"

headers = get_auth_headers("GET", PATH)

response = requests.get(
    BASE_URL + PATH,
    headers=headers,
    timeout=15
)

print("HTTP status:", response.status_code)

if response.status_code == 200:
    print("Kalshi authentication successful!")
elif response.status_code == 401:
    print("Authentication failed. Check API credentials.")
else:
    print("Request failed.")
    print("Response:", response.text[:300])
