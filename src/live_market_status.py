
"""Check the current trading status of Kalshi markets.

Run:
    python -m src.live_market_status
"""

import os

import requests
from dotenv import load_dotenv

from src.market_lifecycle import market_is_tradable
from src.market_ranker import rank_active_markets


load_dotenv()

BASE_URL = os.getenv(
    "KALSHI_REST_URL",
    "https://api.elections.kalshi.com/trade-api/v2",
).rstrip("/")


def get_live_market_status(ticker):
    """Fetch current market information from Kalshi."""

    response = requests.get(
        f"{BASE_URL}/markets/{ticker}",
        timeout=10,
    )

    response.raise_for_status()
    market = response.json()["market"]

    return {
        "ticker": ticker,
        "status": market.get("status"),
        "close_time": market.get("close_time"),
        "tradable": market_is_tradable(
            market,
            min_minutes_to_close=10,
        ),
    }


def main():
    selected = rank_active_markets(max_markets=10)

    print("\n=== LIVE MARKET STATUS ===")

    for market in selected:
        ticker = market["ticker"]

        try:
            result = get_live_market_status(ticker)

            print(
                f"{ticker} | "
                f"Status: {result['status']} | "
                f"Tradable: {result['tradable']} | "
                f"Closes: {result['close_time']}"
            )

        except (requests.RequestException, KeyError, ValueError) as error:
            print(f"{ticker} | STATUS CHECK FAILED: {error}")


if __name__ == "__main__":
    main()
