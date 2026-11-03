
"""Phase 7J.2: Directional GRU robustness by market and session.

Read-only analysis using corrected validation predictions.
Does not access the reserved test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path("data/processed/datasets/v2_phase7j")
METADATA_PATH = ROOT / "gru_dataset_combined.metadata.csv"
VAL_INDICES_PATH = ROOT / "splits/val_indices.npy"

PREDICTIONS_PATH = Path(
    "models/v2_phase7j/direction_corrected/"
    "validation_predictions.npz"
)

THRESHOLD_CENTS = 0.5


def choose_nonoverlapping_indices(metadata):
    """Select timestamps without using outcomes or predictions."""
    selected = []

    ordered = metadata.sort_values(
        ["ticker", "input_start", "target_end"],
        kind="stable",
    )

    for _, group in ordered.groupby("ticker", sort=False):
        last_end = None

        for idx, row in group.iterrows():
            start = row["input_start"]
            end = row["target_end"]

            if last_end is None or start >= last_end:
                selected.append(int(idx))
                last_end = end

    return np.array(sorted(selected), dtype=np.int64)


def report_accuracy(name, frame):
    if frame.empty:
        print(f"{name}: no observations")
        return

    gru = frame["gru_correct"].mean()
    reverse = frame["reverse10_correct"].mean()

    print(
        f"{name}: "
        f"n={len(frame)} | "
        f"GRU={gru:.2%} | "
        f"reverse 10s={reverse:.2%} | "
        f"GRU advantage={(gru - reverse) * 100:+.2f} pp"
    )


def group_analysis(frame, column, name):
    grouped = (
        frame.groupby(column)
        .agg(
            n=("gru_correct", "size"),
            gru_accuracy=("gru_correct", "mean"),
            reverse_accuracy=("reverse10_correct", "mean"),
        )
    )

    grouped["gru_advantage_pp"] = (
        grouped["gru_accuracy"] - grouped["reverse_accuracy"]
    ) * 100

    print(f"\n=== {name} ===")
    print("Groups:", len(grouped))

    print(
        "Equal-group GRU accuracy:",
        f"{grouped['gru_accuracy'].mean():.2%}",
    )
    print(
        "Equal-group reverse 10s accuracy:",
        f"{grouped['reverse_accuracy'].mean():.2%}",
    )

    print(
        "Equal-group GRU advantage:",
        f"{grouped['gru_advantage_pp'].mean():+.2f} pp",
    )

    print(
        "Groups where GRU wins:",
        int((grouped["gru_advantage_pp"] > 0).sum()),
    )
    print(
        "Groups where reverse momentum wins:",
        int((grouped["gru_advantage_pp"] < 0).sum()),
    )
    print(
        "Groups tied:",
        int((grouped["gru_advantage_pp"] == 0).sum()),
    )

    print("\nTop 10 groups by observation count:")
    print(
        grouped.sort_values("n", ascending=False)
        .head(10)
        .to_string(
            formatters={
                "gru_accuracy": "{:.2%}".format,
                "reverse_accuracy": "{:.2%}".format,
                "gru_advantage_pp": "{:+.2f}".format,
            }
        )
    )

    return grouped


def main():
    metadata = pd.read_csv(METADATA_PATH)

    for col in ("input_start", "target_end"):
        metadata[col] = pd.to_datetime(
            metadata[col], utc=True, errors="raise"
        )

    val_indices = np.load(VAL_INDICES_PATH).astype(np.int64)

    if len(np.unique(val_indices)) != len(val_indices):
        raise ValueError("Duplicate validation indices.")

    # Choose spaced windows across ALL validation observations.
    val_metadata = metadata.iloc[val_indices].copy()
    spaced_indices = choose_nonoverlapping_indices(val_metadata)

    with np.load(PREDICTIONS_PATH) as data:
        saved_indices = data["validation_indices"].astype(np.int64)
        actual = data["actual"].astype(int)
        probability = data["probability_up"].reshape(-1)
        momentum10 = data["momentum_10s"].reshape(-1)
        saved_changes = data["price_change_cents"].reshape(-1)

    if not (
        len(saved_indices)
        == len(actual)
        == len(probability)
        == len(momentum10)
        == len(saved_changes)
    ):
        raise ValueError("Saved prediction array lengths differ.")

    if len(np.unique(saved_indices)) != len(saved_indices):
        raise ValueError("Duplicate saved prediction indices.")

    if not np.isin(saved_indices, val_indices).all():
        raise ValueError("Saved predictions are outside validation.")

    if not (
        np.isfinite(probability).all()
        and np.isfinite(momentum10).all()
        and np.isfinite(saved_changes).all()
    ):
        raise ValueError("Non-finite predictions.")

    # Verify the float64 label rule and saved row alignment.
    metadata_changes = pd.to_numeric(
        metadata.iloc[saved_indices]["price_change_cents"],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.all(np.abs(metadata_changes) >= THRESHOLD_CENTS):
        raise ValueError("Saved predictions include flat observations.")

    if not np.array_equal(actual, (metadata_changes > 0).astype(int)):
        raise ValueError("Direction labels do not match metadata.")

    if not np.allclose(
        saved_changes, metadata_changes,
        rtol=1e-5, atol=1e-5,
    ):
        raise ValueError("Saved changes do not match metadata.")

    # Keep only predictions at the independently spaced timestamps.
    positions = pd.Series(
        np.arange(len(saved_indices)),
        index=saved_indices,
    ).reindex(spaced_indices)

    mask = positions.notna().to_numpy()
    selected_indices = spaced_indices[mask]
    selected_positions = positions[mask].astype(int).to_numpy()

    y = actual[selected_positions]
    p = probability[selected_positions]
    m10 = momentum10[selected_positions]

    gru_pred = (p >= 0.5).astype(int)
    reverse10_pred = (m10 < 0).astype(int)

    frame = metadata.iloc[selected_indices].copy()
    frame["actual_up"] = y
    frame["probability_up"] = p
    frame["gru_correct"] = (gru_pred == y).astype(int)
    frame["reverse10_correct"] = (reverse10_pred == y).astype(int)

    # Contract family; useful as a category proxy, NOT an event ID.
    frame["market_family"] = (
        frame["ticker"].str.split("-").str[0]
    )

    # Preliminary event proxy: omit the final dash-delimited
    # contract suffix. This is not guaranteed to identify events.
    frame["event_proxy"] = frame["ticker"].str.rsplit(
        "-", n=1
    ).str[0]

    print("\n=== PHASE 7J.2 ROBUSTNESS CHECK ===")
    print("Total validation windows:", len(val_indices))
    print("All spaced validation windows:", len(spaced_indices))
    print("Spaced moving observations:", len(frame))
    print("Markets:", frame["ticker"].nunique())
    print("Sessions:", frame["session_id"].nunique())
    print("Market families:", frame["market_family"].nunique())

    print("\n=== OVERALL ACCURACY ===")
    report_accuracy("All spaced movements", frame)

    if len(np.unique(y)) == 2:
        print("GRU ROC-AUC:", f"{roc_auc_score(y, p):.4f}")

    market_stats = group_analysis(
        frame, "ticker", "EQUAL-MARKET ANALYSIS"
    )

    session_stats = group_analysis(
        frame, "session_id", "EQUAL-SESSION ANALYSIS"
    )

    print("\n=== EXCLUDING BUSIEST MARKETS ===")

    ranked_markets = (
        frame["ticker"].value_counts().index.tolist()
    )

    for k in (1, 3, 5, 10):
        excluded = set(ranked_markets[:k])
        remaining = frame.loc[
            ~frame["ticker"].isin(excluded)
        ]

        report_accuracy(
            f"Exclude top {k} markets",
            remaining,
        )

    print("\n=== MARKET FAMILY ANALYSIS ===")

    family_stats = (
        frame.groupby("market_family")
        .agg(
            observations=("gru_correct", "size"),
            markets=("ticker", "nunique"),
            gru_accuracy=("gru_correct", "mean"),
            reverse10_accuracy=("reverse10_correct", "mean"),
        )
        .sort_values("observations", ascending=False)
    )

    print(
        family_stats.head(20).to_string(
            formatters={
                "gru_accuracy": "{:.2%}".format,
                "reverse10_accuracy": "{:.2%}".format,
            }
        )
    )

    print("\n=== PRELIMINARY EVENT CONCENTRATION ===")
    event_counts = frame["event_proxy"].value_counts()

    print("Distinct event proxies:", len(event_counts))
    print(
        "Observations in five busiest event proxies:",
        int(event_counts.head(5).sum()),
    )
    print("Top 10 event proxies:")
    print(event_counts.head(10).to_string())

    print("\n=== FINAL CHECK ===")
    print("Results are conditional on actual movement.")
    print("Market-family and event groupings are heuristic.")
    print("No model retrained.")
    print("Reserved test set not accessed.")


if __name__ == "__main__":
    main()
