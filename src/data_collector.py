
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Directory for raw market data
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Generate filename ONCE when the collector starts
SESSION_START = datetime.now().astimezone()

SESSION_FILENAME = (
    f"kalshi_{SESSION_START.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
)

SESSION_FILE = RAW_DIR / SESSION_FILENAME


def save_raw_message(message, market_ticker=None):
    # Record when each message is received
    received_at = datetime.now(timezone.utc)

    record = {
        "session_started_at": SESSION_START.isoformat(),
        "received_at": received_at.isoformat(),
        "received_timestamp_ns": time.time_ns(),
        "market_ticker": market_ticker,
        "message": message
    }

    with SESSION_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
