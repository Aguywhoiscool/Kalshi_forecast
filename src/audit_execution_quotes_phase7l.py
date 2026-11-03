
"""Phase 7L.1: Audit entry and exit quote availability.

Uses the non-overlapping validation timestamps from Phase 7K.1.
Does not access reserved test data or simulate profitability.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/processed/datasets/v2_phase7j")
FEATURE_DIR = Path("data/processed")
PREDICTIONS = Path(
    "models/v2_phase7k/two_stage/validation_predictions.npz"
)
OUTPUT = Path(
    "models/v2_phase7l/execution_quote_audit.csv"
)

MAX_ENTRY_AGE_SECONDS = 1.5
MAX_EXIT_DELAY_SECONDS = 1.5
MAX_SPREAD = 0.10

FIELDS = [
    "timestamp", "ticker", "bid", "ask",
    "spread", "bid_depth", "ask_depth"
]


def valid_quote(row):
    if row is None:
        return False

    values = np.array([
        row["bid"], row["ask"], row["spread"],
        row["bid_depth"], row["ask_depth"]
    ], dtype=float)

    return bool(
        np.isfinite(values).all()
        and 0 < row["bid"] < row["ask"] < 1
        and row["spread"] >= 0
        and row["spread"] <= MAX_SPREAD
        and abs(
            (row["ask"] - row["bid"]) - row["spread"]
        ) < 1e-5
        and row["bid_depth"] > 0
        and row["ask_depth"] > 0
    )


def find_quote(group, target, forward):
    """Find adjacent observation without crossing target time."""
    timestamps = group["timestamp"].to_numpy(
        dtype="datetime64[ns]"
    )
    target64 = np.datetime64(target.to_datetime64(), "ns")

    if forward:
        position = np.searchsorted(
            timestamps, target64, side="left"
        )
    else:
        position = (
            np.searchsorted(
                timestamps, target64, side="right"
            ) - 1
        )

    if position < 0 or position >= len(group):
        return None, float("nan")

    row = group.iloc[position]

    if forward:
        delta = (
            row["timestamp"] - target
        ).total_seconds()
    else:
        delta = (
            target - row["timestamp"]
        ).total_seconds()

    return row, delta


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Audit already exists: {OUTPUT}"
        )

    metadata = pd.read_csv(
        ROOT / "gru_dataset_combined.metadata.csv"
    )

    with np.load(PREDICTIONS) as data:
        val_indices = data[
            "validation_indices"
        ].astype(np.int64)
        spaced_positions = data[
            "spaced_positions"
        ].astype(np.int64)

    expected_val = np.load(
        ROOT / "splits/val_indices.npy"
    ).astype(np.int64)

    if not np.array_equal(val_indices, expected_val):
        raise ValueError(
            "Saved predictions do not align with validation."
        )

    if (
        spaced_positions.ndim != 1
        or len(spaced_positions) == 0
        or len(np.unique(spaced_positions)) != len(spaced_positions)
        or spaced_positions.min() < 0
        or spaced_positions.max() >= len(val_indices)
    ):
        raise ValueError("Invalid spaced positions.")

    selected = metadata.iloc[
        val_indices[spaced_positions]
    ].copy().reset_index(drop=True)

    for col in ("input_end", "target_end"):
        selected[col] = pd.to_datetime(
            selected[col], utc=True, errors="raise"
        )

    rows = []

    source_groups = list(
        selected.groupby("source_file", sort=False)
    )

    print("\n=== PHASE 7L QUOTE AVAILABILITY AUDIT ===")
    print("Spaced observations:", len(selected))
    print("Source files:", len(source_groups))

    for file_number, (filename, observations) in enumerate(
        source_groups, start=1
    ):
        path = FEATURE_DIR / Path(filename).name

        if not path.is_file():
            raise FileNotFoundError(path)

        features = pd.read_csv(
            path, usecols=FIELDS
        )

        features["timestamp"] = pd.to_datetime(
            features["timestamp"],
            utc=True,
            errors="raise",
        )

        for col in (
            "bid", "ask", "spread",
            "bid_depth", "ask_depth"
        ):
            features[col] = pd.to_numeric(
                features[col], errors="coerce"
            )

        ticker_groups = {
            ticker: group.sort_values(
                "timestamp"
            ).reset_index(drop=True)
            for ticker, group in features.groupby("ticker")
        }

        for _, obs in observations.iterrows():
            group = ticker_groups.get(obs["ticker"])

            entry = None
            exit_quote = None
            entry_age = float("nan")
            exit_delay = float("nan")

            if group is not None:
                entry, entry_age = find_quote(
                    group, obs["input_end"], forward=False
                )

                exit_quote, exit_delay = find_quote(
                    group, obs["target_end"], forward=True
                )

            entry_timely = (
                entry is not None
                and 0 <= entry_age <= MAX_ENTRY_AGE_SECONDS
            )

            exit_timely = (
                exit_quote is not None
                and 0 <= exit_delay <= MAX_EXIT_DELAY_SECONDS
            )

            entry_valid = (
                entry_timely and valid_quote(entry)
            )

            exit_valid = (
                exit_timely and valid_quote(exit_quote)
            )

            usable = (
                entry_valid
                and exit_valid
                and exit_quote["timestamp"] > entry["timestamp"]
            )

            rows.append({
                "ticker": obs["ticker"],
                "session_id": obs["session_id"],
                "source_file": filename,
                "input_end": obs["input_end"],
                "target_end": obs["target_end"],
                "entry_age_seconds": entry_age,
                "exit_delay_seconds": exit_delay,
                "entry_timely": entry_timely,
                "exit_timely": exit_timely,
                "entry_valid": entry_valid,
                "exit_valid": exit_valid,
                "usable": usable,
                "entry_bid": (
                    float(entry["bid"]) if entry_valid else np.nan
                ),
                "entry_ask": (
                    float(entry["ask"]) if entry_valid else np.nan
                ),
                "exit_bid": (
                    float(exit_quote["bid"])
                    if exit_valid else np.nan
                ),
                "exit_ask": (
                    float(exit_quote["ask"])
                    if exit_valid else np.nan
                ),
                "entry_bid_depth": (
                    float(entry["bid_depth"])
                    if entry_valid else np.nan
                ),
                "entry_ask_depth": (
                    float(entry["ask_depth"])
                    if entry_valid else np.nan
                ),
                "exit_bid_depth": (
                    float(exit_quote["bid_depth"])
                    if exit_valid else np.nan
                ),
                "exit_ask_depth": (
                    float(exit_quote["ask_depth"])
                    if exit_valid else np.nan
                ),
            })

        if (
            file_number % 10 == 0
            or file_number == len(source_groups)
        ):
            print(
                f"Processed {file_number}/{len(source_groups)} files"
            )

    audit = pd.DataFrame(rows)

    print("\n=== PHASE 7L EXECUTION QUOTE COVERAGE ===")

    print("Observations:", len(audit))

    for col in (
        "entry_timely", "exit_timely",
        "entry_valid", "exit_valid", "usable"
    ):
        count = int(audit[col].sum())
        print(
            f"{col}: {count}/{len(audit)} "
            f"({count / len(audit):.2%})"
        )

    usable = audit.loc[audit["usable"]]

    print("\n=== USABLE OBSERVATIONS ===")
    print("Count:", len(usable))
    print("Markets:", usable["ticker"].nunique())
    print("Sessions:", usable["session_id"].nunique())

    if len(usable):
        entry_spread = (
            usable["entry_ask"] - usable["entry_bid"]
        ) * 100

        exit_spread = (
            usable["exit_ask"] - usable["exit_bid"]
        ) * 100

        print(
            "Median entry spread (cents):",
            f"{entry_spread.median():.3f}"
        )
        print(
            "Median exit spread (cents):",
            f"{exit_spread.median():.3f}"
        )
        print(
            "Median entry quote age (seconds):",
            f"{usable['entry_age_seconds'].median():.3f}"
        )
        print(
            "Median exit delay (seconds):",
            f"{usable['exit_delay_seconds'].median():.3f}"
        )

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    audit.to_csv(OUTPUT, index=False)

    print("\nSaved audit:", OUTPUT)
    print("No profitability calculations performed.")
    print("Reserved test data not accessed.")


if __name__ == "__main__":
    main()
