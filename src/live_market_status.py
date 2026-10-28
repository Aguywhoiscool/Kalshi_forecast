
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
    def quote_cents(field, dollar_field):
        value = market.get(dollar_field)

        if value is not None:
            return round(float(value) * 100, 4)

        value = market.get(field)

        if value is not None:
            return float(value)

        return None

    yes_bid = quote_cents("yes_bid", "yes_bid_dollars")
    yes_ask = quote_cents("yes_ask", "yes_ask_dollars")

    spread_cents = None

    if (
        yes_bid is not None
        and yes_ask is not None
        and 0 <= yes_bid <= yes_ask <= 100
    ):
        spread_cents = round(yes_ask - yes_bid, 4)

    return {
        "ticker": ticker,
        "status": market.get("status"),
        "close_time": market.get("close_time"),
        "tradable": market_is_tradable(
            market,
            min_minutes_to_close=10,
        ),
        "yes_bid_cents": yes_bid,
        "yes_ask_cents": yes_ask,
        "spread_cents": spread_cents,
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
