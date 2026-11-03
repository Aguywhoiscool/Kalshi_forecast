
"""Phase 7M.1: Read-only 10/30/60-second horizon feasibility.

Uses the existing Phase 7L quote audit and validation source files.
No model predictions, training, or reserved test data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data/processed/datasets/v2_phase7j")
FEATURE_DIR = Path("data/processed")
AUDIT_PATH = Path("models/v2_phase7l/execution_quote_audit.csv")
PREDICTIONS_PATH = Path(
    "models/v2_phase7k/two_stage/validation_predictions.npz"
)
OUTPUT = Path("models/v2_phase7m/horizon_feasibility.csv")

HORIZONS = (10, 30, 60)
MAX_EXIT_DELAY = 1.5
MAX_SPREAD = 0.10

COLUMNS = [
    "timestamp", "ticker", "bid", "ask",
    "spread", "bid_depth", "ask_depth",
]


def quote_valid(bid, ask, spread, bid_depth, ask_depth):
    values = np.array(
        [bid, ask, spread, bid_depth, ask_depth],
        dtype=float,
    )
    return bool(
        np.isfinite(values).all()
        and 0 < bid < ask < 1
        and 0 <= spread <= MAX_SPREAD
        and abs((ask - bid) - spread) < 1e-5
        and bid_depth > 0
        and ask_depth > 0
    )


def summarize(frame, title):
    print(f"\n=== {title} ===")

    for horizon in HORIZONS:
        part = frame.loc[frame["horizon_seconds"] == horizon]
        good = part.loc[part["usable"]].copy()

        print(f"\nHORIZON: {horizon} SECONDS")
        print(
            "Usable:",
            f"{len(good)}/{len(part)}",
            f"({len(good) / len(part):.2%})",
        )

        if good.empty:
            continue

        delta = good["midpoint_change_cents"].abs()
        cost = good["breakeven_move_cents"]

        # This assumes perfect advance knowledge of direction.
        # It is an upper-bound diagnostic, NOT a trading strategy.
        oracle_gross = delta - cost

        print("Markets:", good["ticker"].nunique())
        print("Sessions:", good["session_id"].nunique())
        print(
            "Median absolute midpoint move:",
            f"{delta.median():.3f}c",
        )
        print(
            "Mean absolute midpoint move:",
            f"{delta.mean():.3f}c",
        )
        print(
            "90th percentile absolute move:",
            f"{delta.quantile(0.90):.3f}c",
        )
        print(
            "Median break-even movement:",
            f"{cost.median():.3f}c",
        )
        print(
            "Move >= 0.5c:",
            f"{(delta >= 0.5).mean():.2%}",
        )
        print(
            "Move > 1c:",
            f"{(delta > 1).mean():.2%}",
        )
        print(
            "Move > 3c:",
            f"{(delta > 3).mean():.2%}",
        )
        print(
            "Move > round-trip spread cost:",
            f"{(oracle_gross > 1e-8).mean():.2%}",
        )
        print(
            "Mean oracle gross P&L (before fees):",
            f"{oracle_gross.mean():+.3f}c",
        )
        print(
            "Median exit quote delay:",
            f"{good['exit_delay_seconds'].median():.3f}s",
        )


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    audit = pd.read_csv(AUDIT_PATH)
    metadata = pd.read_csv(
        ROOT / "gru_dataset_combined.metadata.csv"
    )

    with np.load(PREDICTIONS_PATH) as data:
        val_idx = data["validation_indices"].astype(np.int64)
        spaced = data["spaced_positions"].astype(np.int64)

    expected_val = np.load(
        ROOT / "splits/val_indices.npy"
    ).astype(np.int64)

    if not np.array_equal(val_idx, expected_val):
        raise ValueError("Validation index mismatch.")

    if (
        spaced.ndim != 1
        or len(spaced) == 0
        or len(np.unique(spaced)) != len(spaced)
        or spaced.min() < 0
        or spaced.max() >= len(val_idx)
    ):
        raise ValueError("Invalid spaced positions.")

    selected = metadata.iloc[
        val_idx[spaced]
    ].reset_index(drop=True)

    if len(audit) != len(selected):
        raise ValueError("Audit row count mismatch.")

    for column in ("ticker", "session_id", "source_file"):
        if not np.array_equal(
            selected[column].astype(str).to_numpy(),
            audit[column].astype(str).to_numpy(),
        ):
            raise ValueError(f"Audit mismatch: {column}")

    for column in ("input_end", "target_end"):
        selected[column] = pd.to_datetime(
            selected[column], utc=True, errors="raise"
        )
        audit_times = pd.to_datetime(
            audit[column], utc=True, errors="raise"
        )
        if not np.array_equal(
            selected[column].to_numpy(),
            audit_times.to_numpy(),
        ):
            raise ValueError(f"Timestamp mismatch: {column}")

    if not audit["usable"].eq(True).all():
        raise ValueError("Audit contains unusable entries.")

    results = []
    groups = selected.groupby("source_file", sort=False)

    print("\n=== PHASE 7M.1 HORIZON AUDIT ===")
    print("Starting observations:", len(selected))
    print("Source files:", groups.ngroups)

    for file_number, (filename, observations) in enumerate(
        groups, start=1
    ):
        source = FEATURE_DIR / Path(filename).name

        if not source.is_file():
            raise FileNotFoundError(source)

        features = pd.read_csv(source, usecols=COLUMNS)
        features["timestamp"] = pd.to_datetime(
            features["timestamp"], utc=True, errors="raise"
        )

        for column in (
            "bid", "ask", "spread", "bid_depth", "ask_depth"
        ):
            features[column] = pd.to_numeric(
                features[column], errors="coerce"
            )

        ticker_groups = {
            ticker: group.sort_values("timestamp").reset_index(drop=True)
            for ticker, group in features.groupby("ticker")
        }

        for original_position, obs in observations.iterrows():
            entry = audit.iloc[original_position]

            entry_bid = float(entry["entry_bid"])
            entry_ask = float(entry["entry_ask"])
            entry_mid = (entry_bid + entry_ask) / 2
            entry_spread = (entry_ask - entry_bid) * 100

            group = ticker_groups.get(obs["ticker"])

            for horizon in HORIZONS:
                # The actual 10-second target is preserved
                # exactly as recorded by the dataset builder.
                target = (
                    obs["target_end"]
                    if horizon == 10
                    else obs["input_end"]
                    + pd.Timedelta(seconds=horizon)
                )

                record = {
                    "ticker": obs["ticker"],
                    "session_id": obs["session_id"],
                    "source_file": filename,
                    "horizon_seconds": horizon,
                    "input_end": obs["input_end"],
                    "target_time": target,
                    "entry_bid": entry_bid,
                    "entry_ask": entry_ask,
                    "entry_spread_cents": entry_spread,
                    "usable": False,
                    "exit_delay_seconds": np.nan,
                    "exit_bid": np.nan,
                    "exit_ask": np.nan,
                    "exit_spread_cents": np.nan,
                    "midpoint_change_cents": np.nan,
                    "breakeven_move_cents": np.nan,
                    "oracle_gross_cents": np.nan,
                }

                if group is not None and not group.empty:
                    times = group["timestamp"].to_numpy(
                        dtype="datetime64[ns]"
                    )

                    position = np.searchsorted(
                        times,
                        np.datetime64(target.to_datetime64(), "ns"),
                        side="left",
                    )

                    if position < len(group):
                        exit_row = group.iloc[position]
                        delay = (
                            exit_row["timestamp"] - target
                        ).total_seconds()

                        if (
                            0 <= delay <= MAX_EXIT_DELAY
                            and quote_valid(
                                exit_row["bid"],
                                exit_row["ask"],
                                exit_row["spread"],
                                exit_row["bid_depth"],
                                exit_row["ask_depth"],
                            )
                        ):
                            exit_bid = float(exit_row["bid"])
                            exit_ask = float(exit_row["ask"])
                            exit_spread = (
                                exit_ask - exit_bid
                            ) * 100

                            exit_mid = (exit_bid + exit_ask) / 2
                            change = (
                                exit_mid - entry_mid
                            ) * 100

                            break_even = (
                                entry_spread + exit_spread
                            ) / 2

                            record.update({
                                "usable": True,
                                "exit_delay_seconds": delay,
                                "exit_bid": exit_bid,
                                "exit_ask": exit_ask,
                                "exit_spread_cents": exit_spread,
                                "midpoint_change_cents": change,
                                "breakeven_move_cents": break_even,
                                "oracle_gross_cents": (
                                    abs(change) - break_even
                                ),
                            })

                            if horizon == 10:
                                if not (
                                    np.isclose(
                                        exit_bid,
                                        float(entry["exit_bid"]),
                                        atol=1e-8,
                                    )
                                    and np.isclose(
                                        exit_ask,
                                        float(entry["exit_ask"]),
                                        atol=1e-8,
                                    )
                                ):
                                    raise ValueError(
                                        "10-second quote differs "
                                        "from Phase 7L audit."
                                    )

                results.append(record)

        if file_number % 10 == 0 or file_number == groups.ngroups:
            print(
                f"Processed {file_number}/{groups.ngroups} files"
            )

    frame = pd.DataFrame(results)

    summarize(frame, "HORIZON COVERAGE AND FEASIBILITY")

    # Fair comparison: retain only observations with valid
    # quotes at every horizon.
    usable_matrix = frame.pivot_table(
        index=["source_file", "ticker", "input_end"],
        columns="horizon_seconds",
        values="usable",
        aggfunc="first",
    )

    common = usable_matrix.index[
        usable_matrix.reindex(
            columns=list(HORIZONS), fill_value=False
        ).fillna(False).all(axis=1)
    ]

    common_frame = frame.set_index(
        ["source_file", "ticker", "input_end"]
    ).loc[common].reset_index()

    print("\nObservations valid at all horizons:", len(common))
    summarize(
        common_frame,
        "SAME-OBSERVATION HORIZON COMPARISON",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)

    print("\n=== PHASE 7M.1 COMPLETE ===")
    print("Saved:", OUTPUT)
    print("No models retrained.")
    print("No reserved test data accessed.")
    print("Oracle results assume perfect future direction.")
    print("Fees, slippage, and fill uncertainty excluded.")


if __name__ == "__main__":
    main()
