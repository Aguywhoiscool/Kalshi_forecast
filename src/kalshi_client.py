import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv(
    "KALSHI_API_BASE_URL",
    "https://external-api.kalshi.com/trade-api/v2"
).rstrip("/")


class KalshiRestClient:

    def __init__(self):
        self.session = requests.Session()

    def get_markets(self, limit=100, status="open"):
        response = self.session.get(
            f"{BASE_URL}/markets",
            params={
                "limit": limit,
                "status": status
            },
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    def get_orderbook(self, ticker):
        response = self.session.get(
            f"{BASE_URL}/markets/{ticker}/orderbook",
            timeout=15
        )

        response.raise_for_status()

        return response.json()




if __name__ == "__main__":
    client = KalshiRestClient()
    result = client.get_markets(limit=100)

    active_markets = []

    for market in result.get("markets", []):
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
        volume = float(market.get("volume_24h_fp") or 0)

        if bid > 0 and ask > 0 and ask >= bid and volume > 0:
            active_markets.append(market)

    print(f"Found {len(active_markets)} quoted markets")

    for market in active_markets[:10]:
        print("-" * 60)
        print("Ticker:", market["ticker"])
        print("Title:", market["title"])
        print("YES bid:", market["yes_bid_dollars"])
        print("YES ask:", market["yes_ask_dollars"])
        print("24h volume:", market["volume_24h_fp"])
