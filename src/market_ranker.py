"""Read-only diversified Kalshi market discovery.
Run: python -m src.market_ranker
"""
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import requests
from dotenv import load_dotenv

from src.market_lifecycle import market_is_tradable

load_dotenv()
BASE_URL = os.getenv('KALSHI_API_BASE_URL', 'https://external-api.kalshi.com/trade-api/v2').rstrip('/')
PAGE_LIMIT = 200
MAX_PAGES = int(os.getenv('DISCOVERY_MAX_PAGES', '50'))
MAX_MARKETS = int(os.getenv('MAX_MARKETS', '12'))
MAX_PER_SERIES = int(os.getenv('MAX_PER_SERIES', '2'))
MIN_VOLUME = Decimal(os.getenv('MIN_VOLUME_24H', '10'))
MAX_SPREAD = Decimal(os.getenv('MAX_SPREAD_DOLLARS', '0.10'))
MIN_DEPTH = Decimal(os.getenv('MIN_TOP_DEPTH', '1'))
MIN_HOURS = float(os.getenv('MIN_HOURS_TO_CLOSE', '1'))


def number(value):
    try:
        return Decimal(str(value or '0'))
    except (ValueError, InvalidOperation):
        return Decimal('0')


def series_of(market):
    return (market.get('event_ticker') or market['ticker']).split('-')[0]


def hours_until_close(market):
    stamp = market.get('close_time')
    if not stamp:
        return -1
    try:
        value = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        return (value - datetime.now(timezone.utc)).total_seconds() / 3600
    except ValueError:
        return -1


def fetch_candidates(session, max_pages=MAX_PAGES):
    """Broad paginated discovery; a page cap is not exchange-wide coverage."""
    cursor = None
    all_markets = []
    for page in range(max_pages):
        params = {'status': 'open', 'mve_filter': 'exclude', 'limit': PAGE_LIMIT}
        if cursor:
            params['cursor'] = cursor
        for attempt in range(3):
            try:
                response = session.get(f'{BASE_URL}/markets', params=params, timeout=20)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        result = response.json()
        rows = result.get('markets', [])
        all_markets.extend(rows)
        print(f'Page {page+1}: {len(rows)} markets; scanned {len(all_markets)}', flush=True)
        next_cursor = result.get('cursor')
        if not next_cursor or next_cursor == cursor or not rows:
            break
        cursor = next_cursor
    return all_markets


def rank_active_markets(
    max_markets=MAX_MARKETS,
    max_pages=MAX_PAGES,
    excluded_tickers=None ):
    with requests.Session() as session:
        markets = fetch_candidates(session, max_pages)
    excluded_tickers = set(excluded_tickers or [])
    candidates = []
    for m in markets:
        if m["ticker"] in excluded_tickers:
            continue
        if not market_is_tradable(
            m,
            min_minutes_to_close=60
        ):
            continue

        if hours_until_close(m) < MIN_HOURS:
            continue
        bid = number(m.get('yes_bid_dollars'))
        ask = number(m.get('yes_ask_dollars'))
        spread = ask - bid
        volume = number(m.get('volume_24h_fp'))
        bid_depth = number(m.get('yes_bid_size_fp'))
        ask_depth = number(m.get('yes_ask_size_fp'))
        if not (Decimal('0') < bid < ask < Decimal('1')):
            continue
        if spread > MAX_SPREAD or volume < MIN_VOLUME:
            continue
        if min(bid_depth, ask_depth) < MIN_DEPTH:
            continue
        # Relative scoring: favors live activity, price quality and displayed depth.
        score = (math.log1p(float(volume))
                 + .25 * math.log1p(float(min(bid_depth, ask_depth)))
                 - 30 * float(spread))
        candidates.append((score, m))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    per_series = Counter()
    events = set()
    for score, m in candidates:
        series = series_of(m)
        event = m.get('event_ticker') or m['ticker']
        if per_series[series] >= MAX_PER_SERIES or event in events:
            continue
        selected.append(m)
        per_series[series] += 1
        events.add(event)
        if len(selected) >= max_markets:
            break

    print(f'Eligible: {len(candidates)} / {len(markets)} scanned; selected: {len(selected)}')
    for m in selected:
        print(f"{m['ticker']} | series={series_of(m)} | volume24h={m.get('volume_24h_fp')} "
              f"| bid={m.get('yes_bid_dollars')} ask={m.get('yes_ask_dollars')} "
              f"| depth={m.get('yes_bid_size_fp')}/{m.get('yes_ask_size_fp')}")
    if len(selected) < max_markets:
        print('Selection under target: try increasing DISCOVERY_MAX_PAGES or relaxing thresholds.')
    return selected


if __name__ == '__main__':
    rank_active_markets()
