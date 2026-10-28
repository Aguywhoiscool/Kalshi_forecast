
"""Analyze coverage and activity in existing regression datasets.

Run:
    python -m src.analyze_training_coverage

Reads TRAIN and VALIDATION metadata only.
Does not inspect the reserved test set.
"""

from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/regression_v1")
OUTPUT_DIR = Path("models/training_coverage")

MOVEMENT_THRESHOLD_CENTS = 0.5
WINDOW_SECONDS = 60
HORIZON_SECONDS = 10

# Conservative spacing proxy for reducing window overlap.
SPACING_SECONDS = WINDOW_SECONDS + HORIZON_SECONDS


def load_split(name):
    path = DATA_DIR / f"{name}_metadata.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)

    required = {
        "ticker", "session_id", "source_file",
        "input_start", "input_end",
        "price_change_cents",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} missing columns: {sorted(missing)}"
        )

    for col in ("input_start", "input_end"):
        df[col] = pd.to_datetime(
            df[col], utc=True, errors="raise"
        )

    df["price_change_cents"] = pd.to_numeric(
        df["price_change_cents"], errors="raise"
    )

    if not np.isfinite(
        df["price_change_cents"].to_numpy()
    ).all():
        raise ValueError(f"Nonfinite targets in {name}")

    df["split"] = name

    df["moving"] = (
        df["price_change_cents"].abs()
        >= MOVEMENT_THRESHOLD_CENTS
    )

    # Broad series/category proxy, not a verified event ID.
    df["series_proxy"] = (
        df["ticker"].astype(str).str.split("-").str[0]
    )

    return df


def select_spaced_windows(df):
    """Greedy selection with 70s spacing per ticker.

    Spacing is applied across sessions to prevent treating
    reconnects as independent markets.
    """

    selected_indices = []
    next_allowed = {}

    ordered = df.sort_values(
        ["input_end", "ticker", "source_file"]
    )

    for idx, row in ordered.iterrows():
        ticker = row["ticker"]
        timestamp = row["input_end"]

        if (
            ticker in next_allowed
            and timestamp < next_allowed[ticker]
        ):
            continue

        selected_indices.append(idx)

        next_allowed[ticker] = (
            timestamp
            + pd.Timedelta(seconds=SPACING_SECONDS)
        )

    return df.loc[selected_indices].copy()


def describe_split(df, name):
    spaced = select_spaced_windows(df)

    print(f"\n=== {name.upper()} COVERAGE ===")
    print("Total overlapping windows:", f"{len(df):,}")
    print("Unique markets:", df["ticker"].nunique())
    print("Series proxies:", df["series_proxy"].nunique())
    print("Recording sessions:", df["session_id"].nunique())
    print("Feature files:", df["source_file"].nunique())

    print(
        "Windows with >= 0.5 cent movement:",
        f"{int(df['moving'].sum()):,}",
    )

    print("Movement rate:", f"{df['moving'].mean():.2%}")

    print(
        "Markets with >= 1 moving window:",
        df.loc[df["moving"], "ticker"].nunique(),
    )

    print(
        "70-second-spaced windows:",
        f"{len(spaced):,}",
    )

    print(
        "70-second-spaced moving windows:",
        f"{int(spaced['moving'].sum()):,}",
    )

    return spaced


def build_market_report(df, spaced):
    report = df.groupby("ticker").agg(
        windows=("moving", "size"),
        moving_windows=("moving", "sum"),
        movement_rate=("moving", "mean"),
        avg_abs_move_cents=(
            "price_change_cents",
            lambda s: s.abs().mean()
        ),
        max_abs_move_cents=(
            "price_change_cents",
            lambda s: s.abs().max()
        ),
        sessions=("session_id", "nunique"),
        source_files=("source_file", "nunique"),
        first_window=("input_end", "min"),
        last_window=("input_end", "max"),
    )

    spaced_counts = spaced.groupby("ticker").agg(
        spaced_windows=("moving", "size"),
        spaced_moving_windows=("moving", "sum"),
    )

    report = report.join(
        spaced_counts, how="left"
    ).fillna({
        "spaced_windows": 0,
        "spaced_moving_windows": 0,
    })

    report["spaced_windows"] = (
        report["spaced_windows"].astype(int)
    )

    report["spaced_moving_windows"] = (
        report["spaced_moving_windows"].astype(int)
    )

    report["series_proxy"] = (
        report.index.to_series().str.split("-").str[0]
    )

    return report.reset_index()


def print_market_summary(report, name):
    print(f"\n=== {name.upper()} MARKET ACTIVITY ===")

    print(
        "Markets with zero meaningful movements:",
        int((report["moving_windows"] == 0).sum())
    )

    print(
        "Markets with >= 10 moving windows:",
        int((report["moving_windows"] >= 10).sum())
    )

    print(
        "Markets with >= 30 spaced moving windows:",
        int(
            (report["spaced_moving_windows"] >= 30).sum()
        )
    )

    total_moves = report["moving_windows"].sum()

    if total_moves:
        top_two = (
            report.nlargest(2, "moving_windows")[
                "moving_windows"
            ].sum()
        )

        print(
            "Share of moving windows from top 2 markets:",
            f"{top_two / total_moves:.2%}"
        )

    columns = [
        "ticker", "windows", "moving_windows",
        "movement_rate", "spaced_moving_windows",
        "sessions",
    ]

    print("\nTop 15 markets by moving windows:")

    print(
        report.nlargest(
            15, "moving_windows"
        )[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    train = load_split("train")
    val = load_split("val")

    train_spaced = describe_split(train, "train")
    val_spaced = describe_split(val, "validation")

    train_report = build_market_report(
        train, train_spaced
    )

    val_report = build_market_report(
        val, val_spaced
    )

    print_market_summary(train_report, "train")
    print_market_summary(val_report, "validation")

    train_tickers = set(train_report["ticker"])
    val_tickers = set(val_report["ticker"])

    unseen = val_report[
        ~val_report["ticker"].isin(train_tickers)
    ]

    unseen_moving = unseen[
        unseen["moving_windows"] > 0
    ]

    print("\n=== GENERALIZATION COVERAGE ===")
    print("Markets seen in both:", len(
        train_tickers & val_tickers
    ))
    print("Validation-only markets:", len(
        val_tickers - train_tickers
    ))
    print(
        "Validation-only markets with movement:",
        len(unseen_moving)
    )
    print(
        "Spaced moving windows in validation-only markets:",
        int(unseen_moving["spaced_moving_windows"].sum())
    )

    # Event identity cannot be reliably recovered from
    # ticker strings alone, so report broad series only.
    series = train.groupby("series_proxy").agg(
        windows=("moving", "size"),
        moving_windows=("moving", "sum"),
        markets=("ticker", "nunique"),
    ).reset_index()

    series = series.sort_values(
        "moving_windows", ascending=False
    )

    print("\n=== TRAINING SERIES COVERAGE ===")
    print(series.head(15).to_string(index=False))

    train_report.to_csv(
        OUTPUT_DIR / "train_markets.csv", index=False
    )
    val_report.to_csv(
        OUTPUT_DIR / "validation_markets.csv", index=False
    )
    series.to_csv(
        OUTPUT_DIR / "training_series.csv", index=False
    )

    print("\nReports saved to:", OUTPUT_DIR)
    print("Reserved test data was not inspected.")


if __name__ == "__main__":
    main()
