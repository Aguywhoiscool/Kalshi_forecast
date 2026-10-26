
from decimal import Decimal


class OrderBookManager:
    def __init__(self):
        self.books = {}
        self.last_sequences = {}
        self.invalid_subscriptions = set()

    def process(self, message):
        kind = message.get("type")

        if kind not in ("orderbook_snapshot", "orderbook_delta"):
            return None

        sid = message["sid"]
        seq = message["seq"]
        payload = message["msg"]
        ticker = payload["market_ticker"]

        # Sequence continuity belongs to the subscription,
        # not to an individual ticker.
        if sid in self.invalid_subscriptions:
            return None

        previous = self.last_sequences.get(sid)

        if previous is not None and seq != previous + 1:
            print(
                f"SEQUENCE GAP: sid={sid}, "
                f"expected={previous + 1}, got={seq}"
            )
            self.invalid_subscriptions.add(sid)

            # We cannot safely continue using these books.
            for book in self.books.values():
                if book["sid"] == sid:
                    book["valid"] = False

            return None

        self.last_sequences[sid] = seq

        if kind == "orderbook_snapshot":
            return self._snapshot(sid, ticker, payload)

        return self._delta(sid, ticker, payload)

    def _snapshot(self, sid, ticker, payload):
        yes_levels = payload.get("yes_dollars_fp")
        no_levels = payload.get("no_dollars_fp")

        # Handle older or alternate snapshot formats.
        if yes_levels is None:
            yes_levels = payload.get("yes_dollars", [])

        if no_levels is None:
            no_levels = payload.get("no_dollars", [])

        yes = {
            Decimal(price): Decimal(quantity)
            for price, quantity in yes_levels
            if Decimal(quantity) > 0
        }

        no = {
            Decimal(price): Decimal(quantity)
            for price, quantity in no_levels
            if Decimal(quantity) > 0
        }

        self.books[ticker] = {
            "sid": sid,
            "yes": yes,
            "no": no,
            "valid": True
        }

        return self.get_prices(ticker)

    def _delta(self, sid, ticker, payload):
        book = self.books.get(ticker)

        # Never apply a delta before receiving a snapshot.
        if book is None or not book["valid"]:
            return None

        side = payload["side"]
        price = Decimal(payload["price_dollars"])

        delta_value = payload.get("delta_fp")

        if delta_value is None:
            delta_value = payload.get("delta")

        delta = Decimal(str(delta_value))

        if side not in ("yes", "no"):
            raise ValueError(f"Unknown order-book side: {side}")

        levels = book[side]

        current_quantity = levels.get(price, Decimal("0"))
        new_quantity = current_quantity + delta

        if new_quantity < 0:
            book["valid"] = False
            raise ValueError(
                f"Negative order-book quantity for {ticker}"
            )

        if new_quantity == 0:
            levels.pop(price, None)
        else:
            levels[price] = new_quantity

        return self.get_prices(ticker)

    def get_prices(self, ticker):
        book = self.books.get(ticker)

        if not book or not book["valid"]:
            return None

        yes = book["yes"]
        no = book["no"]

        best_bid = max(yes) if yes else None

        # Assumes default NO-leg price representation.
        best_ask = (
            Decimal("1") - max(no)
            if no else None
        )

        spread = None
        midpoint = None

        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            midpoint = (best_bid + best_ask) / 2

        return {
            "ticker": ticker,
            "yes_bid": best_bid,
            "yes_ask": best_ask,
            "spread": spread,
            "midpoint": midpoint,
            "yes_bid_depth": yes.get(best_bid) if best_bid is not None else None,
            "yes_ask_depth": (
                no.get(max(no)) if no else None
            ),
            "yes_levels": len(yes),
            "no_levels": len(no)
        }
