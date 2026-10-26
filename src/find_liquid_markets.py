
from src.market_discovery import discover_series
from src.kalshi_client import KalshiRestClient

def calculate_prices(book):
    yes_levels = book.get("yes_dollars", [])
    no_levels = book.get("no_dollars", [])

    if not yes_levels or not no_levels:
        return None

    # Use maximum prices, regardless of list order.
    best_yes_bid = max(
        Decimal(level[0]) for level in yes_levels
    )

    best_no_bid = max(
        Decimal(level[0]) for level in no_levels
    )

    best_yes_ask = Decimal("1") - best_no_bid

    spread = best_yes_ask - best_yes_bid
    midpoint = (best_yes_bid + best_yes_ask) / 2

    return {
        "bid": best_yes_bid,
        "ask": best_yes_ask,
        "spread": spread,
        "midpoint": midpoint
    }

def find_liquid_markets():
    client = KalshiRestClient()

    # Retrieve Bitcoin markets
    markets = discover_series("KXBTC")

    # Sort by recent trading volume
    markets.sort(
        key=lambda m: float(m.get("volume_24h_fp") or 0),
        reverse=True
    )

    print("\nSEARCHING FOR NONEMPTY ORDER BOOKS")
    print("=" * 60)

    found = 0

    # Check the 30 highest-volume markets
    for market in markets[:30]:
        ticker = market["ticker"]

        try:
            response = client.get_orderbook(ticker)
            book = response.get("orderbook_fp", {})

            yes_levels = book.get("yes_dollars", [])
            no_levels = book.get("no_dollars", [])

            
            if yes_levels or no_levels:
                print("-" * 60)
                print("Ticker:", ticker)
                print("YES levels:", len(yes_levels))
                print("NO levels:", len(no_levels))
                print("YES bids:", yes_levels[-3:])
                print("NO bids:", no_levels[-3:])

                # Calculate prices for EVERY qualifying market
                prices = calculate_prices(book)

                if prices is not None:
                    print("Best YES bid:", prices["bid"])
                    print("Best YES ask:", prices["ask"])
                    print("Spread:", prices["spread"])
                    print("Midpoint:", prices["midpoint"])
                else:
                    print("Could not calculate prices")

                found += 1

                if found >= 3:
                    break


        except Exception as error:
            print(f"Error checking {ticker}: {error}")

    print("\nMarkets found:", found)

from decimal import Decimal




if __name__ == "__main__":
    find_liquid_markets()
