"""Read-only multi-market Kalshi streaming monitor.
Run: python -m src.websocket_client
Needs existing src.authentication.get_auth_headers and src.market_ranker.
"""
import asyncio
import json
import os
import time
from collections import Counter

import websockets
from dotenv import load_dotenv

from src.authentication import get_auth_headers
from src.market_ranker import rank_active_markets
from src.data_collector import save_raw_message
load_dotenv()
WS_URL = os.getenv('KALSHI_WS_URL', 'wss://external-api-ws.kalshi.com/trade-api/ws/v2')
REPORT_SECONDS = 30


async def connect_to_kalshi():
    # Discovery uses synchronous REST before starting websocket receive loop.
    selected = await asyncio.to_thread(rank_active_markets)
    tickers = [market['ticker'] for market in selected]
    if not tickers:
        print('No markets passed filters. Adjust discovery settings; no subscription made.')
        return

    auth = get_auth_headers('GET', '/trade-api/ws/v2')
    counts = Counter()
    snapshots = set()
    last_seq_by_sid = {}
    connected_since = time.monotonic()
    print(f'Connecting to {len(tickers)} markets...')
    async with websockets.connect(
        WS_URL,
        additional_headers=auth,
        ping_interval=20,
        ping_timeout=20,
        max_size=8 * 1024 * 1024,
    ) as ws:
        await ws.send(json.dumps({
            'id': 1,
            'cmd': 'subscribe',
            'params': {'channels': ['orderbook_delta'], 'market_tickers': tickers},
        }))
        print('Subscription sent. Waiting for exchange confirmation and snapshots.')
        last_report = time.monotonic()
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                
                event = json.loads(raw)
                save_raw_message(event, market_ticker=event.get('msg', {}).get('market_ticker'))
                kind = event.get('type')
                msg = event.get('msg') or {}
                ticker = msg.get('market_ticker')
                sid = event.get('sid')
                seq = event.get('seq')
                if sid is not None and isinstance(seq, int):
                    previous = last_seq_by_sid.get(sid)
                    if previous is not None and seq != previous + 1:
                        print(f'WARNING: sequence gap on sid {sid}: {previous} -> {seq}; resync needed')
                    last_seq_by_sid[sid] = seq
                if kind == 'subscribed':
                    print('Subscription confirmed:', msg)
                elif kind == 'orderbook_snapshot':
                    snapshots.add(ticker)
                    print('Snapshot:', ticker,
                          'YES levels:', len(msg.get('yes_dollars_fp', [])),
                          'NO levels:', len(msg.get('no_dollars_fp', [])))
                elif kind == 'orderbook_delta':
                    counts[ticker] += 1
                elif kind == 'error':
                    print('Kalshi subscription error:', msg)
                elif kind not in ('heartbeat',):
                    print('Message:', kind)
            if time.monotonic() - last_report >= REPORT_SECONDS:
                duration = max(time.monotonic() - connected_since, 1)
                print(f'\n=== Activity report: {duration:.0f} seconds ===')
                for ticker in tickers:
                    c = counts[ticker]
                    print(f'{ticker}: {c} deltas ({c * 60/duration:.1f}/min), '
                          f"snapshot={'yes' if ticker in snapshots else 'no'}")
                last_report = time.monotonic()


if __name__ == '__main__':
    try:
        asyncio.run(connect_to_kalshi())
    except KeyboardInterrupt:
        print('WebSocket disconnected')
