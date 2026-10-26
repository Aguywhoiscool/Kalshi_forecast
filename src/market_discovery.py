
from src.kalshi_client import KalshiRestClient


def discover_markets(max_pages=10):
    client = KalshiRestClient()

    all_markets = []
    cursor = None

    for page in range(max_pages):
        params = {
            "status": "open",
            "limit": 200,
            "mve_filter": "exclude"
        }

        if cursor:
            params["cursor"] = cursor

        response = client.session.get(
            f"{client_base_url()}/markets",
            params=params,
            timeout=15
        )

        response.raise_for_status()
        result = response.json()

        markets = result.get("markets", [])
        all_markets.extend(markets)

        print(f"Page {page + 1}: {len(markets)} markets")

        cursor = result.get("cursor")

        if not cursor:
            break

    return all_markets


def client_base_url():
    from src.kalshi_client import BASE_URL
    return BASE_URL

def discover_series(series_ticker="KXBTC"):
    from src.kalshi_client import BASE_URL
    import requests

    response = requests.get(
        f"{BASE_URL}/markets",
        params={
            "series_ticker": series_ticker,
            "status": "open",
            "mve_filter": "exclude",
            "limit": 200
        },
        timeout=15
    )

    response.raise_for_status()

    markets = response.json().get("markets", [])

    print(f"\nFound {len(markets)} markets in {series_ticker}")

    markets.sort(
        key=lambda m: float(m.get("volume_24h_fp") or 0),
        reverse=True
    )

    for market in markets[:10]:
        print("-" * 60)
        print("Ticker:", market.get("ticker"))
        print("Title:", market.get("title"))
        print("24h volume:", market.get("volume_24h_fp"))
        print("YES bid:", market.get("yes_bid_dollars"))
        print("YES ask:", market.get("yes_ask_dollars"))

    return markets

from datetime import datetime, timezone
from src.kalshi_client import KalshiRestClient


def select_live_bitcoin_market():
    client = KalshiRestClient()
    markets = discover_series("KXBTC")

    now = datetime.now(timezone.utc)
    candidates = []

    for market in markets:
        ticker = market["ticker"]

        # Reject expired contracts
        close_time = market.get("close_time")

        if not close_time:
            continue

        closes = datetime.fromisoformat(
            close_time.replace("Z", "+00:00")
        )

        if closes <= now:
            continue

        volume = float(
            market.get("volume_24h_fp") or 0
        )

        bid = float(
            market.get("yes_bid_dollars") or 0
        )

        ask = float(
            market.get("yes_ask_dollars") or 0
        )

        # Require trading activity and usable quotes
        if (
            volume > 0
            and bid > 0
            and ask > bid
            and ask - bid <= 0.05
        ):
            candidates.append(market)

    candidates.sort(
        key=lambda m: float(
            m.get("volume_24h_fp") or 0
        ),
        reverse=True
    )

    for market in candidates:
        ticker = market["ticker"]

        # Verify that the market still exists
        try:
            response = client.session.get(
                f"https://external-api.kalshi.com/"
                f"trade-api/v2/markets/{ticker}",
                timeout=15
            )

            if response.status_code == 200:
                print("Selected market:", ticker)
                print("Title:", market.get("title"))
                print("24h volume:", market.get("volume_24h_fp"))
                return ticker

        except Exception as error:
            print("Skipping:", ticker, error)

    print("No suitable Bitcoin market found")
    return None


if __name__ == "__main__":
    markets = discover_markets()

    # Examine market data
    quoted = []
    traded = []

    for market in markets:
        bid = float(market.get("yes_bid_dollars") or 0)
        ask = float(market.get("yes_ask_dollars") or 0)
        volume = float(market.get("volume_24h_fp") or 0)

        if bid > 0 or ask > 0:
            quoted.append(market)

        if volume > 0:
            traded.append(market)

    print()
    print("Total markets scanned:", len(markets))
    print("Markets with quotes:", len(quoted))
    print("Markets with 24h volume:", len(traded))

    # Rank markets by recent trading volume
    traded.sort(
        key=lambda m: float(
            m.get("volume_24h_fp") or 0
        ),
        reverse=True
    )

    print("\nTOP MARKETS BY 24H VOLUME")

    for market in traded[:10]:
        print("-" * 60)
        print("Ticker:", market.get("ticker"))
        print("Title:", market.get("title"))
        print("YES bid:", market.get("yes_bid_dollars"))
        print("YES ask:", market.get("yes_ask_dollars"))
        print("24h volume:", market.get("volume_24h_fp"))

    print("\nMARKET DATA DIAGNOSTICS")
    print("=" * 60)

    if markets:
        market = markets[0]

        print("Sample ticker:", market.get("ticker"))
        print("Available fields:", list(market.keys()))

        print("\nVolume fields:")
        print("24h volume:", repr(market.get("volume_24h_fp")))
        print("Total volume:", repr(market.get("volume_fp")))

        print("\nPrice fields:")
        print("YES bid:", market.get("yes_bid_dollars"))
        print("YES ask:", market.get("yes_ask_dollars"))

        print("\nMarket type:")
        print("MVE collection:", market.get("mve_collection_ticker"))

    total_traded = sum(
        float(m.get("volume_fp") or 0) > 0
        for m in markets
    )

    print("\nMarkets with lifetime trading volume:", total_traded)

