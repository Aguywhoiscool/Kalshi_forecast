
import asyncio
import json
import os

import websockets
from dotenv import load_dotenv

from src.authentication import get_auth_headers
from src.market_discovery import select_live_bitcoin_market

load_dotenv()

WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


# Previously verified Bitcoin contract.
# Replace with an actively traded ticker if it expires.



async def connect_to_kalshi():
    market_ticker = select_live_bitcoin_market()
    if market_ticker is None:
        print("No suitable Bitcoin market found.")
        return

    print("Selected market:", market_ticker)

    # Authenticate the WebSocket handshake
    headers = get_auth_headers(
        "GET",
        "/trade-api/ws/v2"
    )

    print("Connecting to Kalshi...")

    async with websockets.connect(
        WS_URL,
        additional_headers=headers,
        ping_interval=20,
        ping_timeout=20
    ) as websocket:

        print("WebSocket connected!")

        subscription = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["orderbook_delta"],
                "market_ticker": selected_tickers
            }
        }

        await websocket.send(
            json.dumps(subscription)
        )

        print("Subscribed to:", MARKET_TICKER)
        print("Waiting for updates...")

        async for message in websocket:

            data = json.loads(message)

            message_type = data.get("type")

            if message_type == "subscribed":
                print("Subscription confirmed")

            elif message_type == "orderbook_snapshot":
                print("ORDER BOOK SNAPSHOT")
                print(json.dumps(data, indent=2))

            elif message_type == "orderbook_delta":
                print("ORDER BOOK UPDATE")
                print(json.dumps(data, indent=2))

            elif message_type == "error":
                print("Kalshi error:", data)

            else:
                print("Message:", message_type)


if __name__ == "__main__":
    try:
        asyncio.run(connect_to_kalshi())
    except KeyboardInterrupt:
        print("\nWebSocket disconnected")
