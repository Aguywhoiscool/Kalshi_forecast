
"""Phase 7L.2: Gross bid/ask execution simulation.

Read-only development-validation analysis.
One contract per selected signal.
No Kalshi fees, slippage, or guaranteed-fill assumptions.
Does not access reserved test data.

Run:
    python -m src.simulate_trades_phase7l
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("data/processed/datasets/v2_phase7j")

PREDICTIONS = Path(
    "models/v2_phase7k/two_stage/validation_predictions.npz"
)

AUDIT = Path(
    "models/v2_phase7l/execution_quote_audit.csv"
)

METADATA = ROOT / "gru_dataset_combined.metadata.csv"
VAL_INDICES = ROOT / "splits/val_indices.npy"

OUTPUT = Path(
    "models/v2_phase7l/gross_simulation_results.csv"
)

FRACTIONS = (0.01, 0.05, 0.10, 0.20)


def pnl_cents(entry_bid, entry_ask, exit_bid, exit_ask, up):
    """Gross P&L for buying one YES or one NO contract."""

    if up:
        # Buy YES at ask; sell YES at bid.
        return 100 * (exit_bid - entry_ask)

    # Buy NO at 1 - YES bid;
    # sell NO at 1 - YES ask.
    return 100 * (entry_bid - exit_ask)


def evaluate(name, frame, score_col, up_col):
    print(f"\n=== {name.upper()} GROSS EXECUTION ===")

    ranked = frame.sort_values(
        score_col,
        ascending=False,
        kind="stable",
    )

    results = []

    for fraction in FRACTIONS:
        count = max(1, int(len(ranked) * fraction))
        chosen = ranked.iloc[:count]

        pnl = np.where(
            chosen[up_col].to_numpy(dtype=bool),
            100 * (
                chosen["exit_bid"].to_numpy()
                - chosen["entry_ask"].to_numpy()
            ),
            100 * (
                chosen["entry_bid"].to_numpy()
                - chosen["exit_ask"].to_numpy()
            ),
        )

        gross = float(pnl.sum())
        mean = float(pnl.mean())
        profitable = int((pnl > 1e-8).sum())
        losses = int((pnl < -1e-8).sum())
        flat = count - profitable - losses

        results.append({
            "method": name,
            "fraction": fraction,
            "trades": count,
            "gross_total_cents": gross,
            "gross_average_cents": mean,
            "profitable_trades": profitable,
            "losing_trades": losses,
            "zero_pnl_trades": flat,
            "win_rate": profitable / count,
        })

        print(f"\nTop {fraction:.0%}: {count} trades")
        print(f"Gross total: {gross:+.2f} cents")
        print(f"Average per trade: {mean:+.4f} cents")
        print(f"Profitable: {profitable}/{count} ({profitable/count:.2%})")
        print(f"Losing: {losses}")
        print(f"Zero gross P&L: {flat}")
        print(f"Median P&L: {np.median(pnl):+.3f} cents")
        print(
            "Markets:",
            chosen["ticker"].nunique(),
            "| Sessions:",
            chosen["session_id"].nunique(),
        )

    return pd.DataFrame(results)


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    audit = pd.read_csv(AUDIT)

    with np.load(PREDICTIONS) as data:
        saved_indices = data[
            "validation_indices"
        ].astype(np.int64)

        spaced_positions = data[
            "spaced_positions"
        ].astype(np.int64)

        p_move_all = data[
            "probability_movement"
        ].reshape(-1)

        p_up_all = data[
            "probability_up_given_move"
        ].reshape(-1)

        volatility_all = data[
            "volatility"
        ].reshape(-1)

        momentum_all = data[
            "momentum10"
        ].reshape(-1)

    expected_indices = np.load(
        VAL_INDICES
    ).astype(np.int64)

    if not np.array_equal(saved_indices, expected_indices):
        raise ValueError(
            "Saved predictions do not match validation indices."
        )

    if (
        spaced_positions.ndim != 1
        or len(spaced_positions) == 0
        or len(np.unique(spaced_positions)) != len(spaced_positions)
        or spaced_positions.min() < 0
        or spaced_positions.max() >= len(saved_indices)
    ):
        raise ValueError("Invalid spaced positions.")

    metadata = pd.read_csv(METADATA)

    selected = metadata.iloc[
        saved_indices[spaced_positions]
    ].reset_index(drop=True)

    if len(audit) != len(selected):
        raise ValueError("Audit and prediction row counts differ.")

    # Confirm that both files refer to exactly the same
    # market, session, and forecast timestamps.
    for col in ("ticker", "session_id"):
        if not (
            audit[col].astype(str).to_numpy()
            == selected[col].astype(str).to_numpy()
        ).all():
            raise ValueError(f"Audit alignment mismatch: {col}")

    for col in ("input_end", "target_end"):
        audit_times = pd.to_datetime(
            audit[col], utc=True, errors="raise"
        )
        selected_times = pd.to_datetime(
            selected[col], utc=True, errors="raise"
        )

        if not np.array_equal(
            audit_times.to_numpy(),
            selected_times.to_numpy(),
        ):
            raise ValueError(
                f"Audit timestamp alignment mismatch: {col}"
            )

    if not audit["usable"].astype(bool).all():
        raise ValueError(
            "Some observations lack usable quotes. "
            "Review the quote audit before simulating."
        )

    frame = audit.copy()

    quote_cols = [
        "entry_bid", "entry_ask",
        "exit_bid", "exit_ask",
    ]

    if not np.isfinite(
        frame[quote_cols].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing or non-finite quote prices.")

    if not (
        (frame["entry_bid"] > 0).all()
        and (frame["entry_bid"] < frame["entry_ask"]).all()
        and (frame["entry_ask"] < 1).all()
        and (frame["exit_bid"] > 0).all()
        and (frame["exit_bid"] < frame["exit_ask"]).all()
        and (frame["exit_ask"] < 1).all()
    ):
        raise ValueError("Invalid bid/ask quotes.")

    p_move = p_move_all[spaced_positions]
    p_up = p_up_all[spaced_positions]
    volatility = volatility_all[spaced_positions]
    momentum = momentum_all[spaced_positions]

    if not all(
        np.isfinite(a).all()
        for a in (p_move, p_up, volatility, momentum)
    ):
        raise ValueError("Non-finite forecast scores.")

    if (
        np.any((p_move < 0) | (p_move > 1))
        or np.any((p_up < 0) | (p_up > 1))
    ):
        raise ValueError("Probability outside [0,1].")

    frame["gru_score"] = (
        p_move * np.abs(2 * p_up - 1)
    )

    frame["volatility_score"] = volatility

    frame["gru_up"] = p_up >= 0.5

    # Same reversed-momentum convention as Phase 7K.1.
    frame["baseline_up"] = momentum < 0

    entry_spread = (
        frame["entry_ask"] - frame["entry_bid"]
    ) * 100

    exit_spread = (
        frame["exit_ask"] - frame["exit_bid"]
    ) * 100

    # Required midpoint change in predicted direction
    # to break even before fees.
    frame["breakeven_move_cents"] = (
        entry_spread + exit_spread
    ) / 2

    # Per-signal gross outcomes for later inspection.
    frame["gru_pnl_cents"] = np.where(
        frame["gru_up"],
        100 * (frame["exit_bid"] - frame["entry_ask"]),
        100 * (frame["entry_bid"] - frame["exit_ask"]),
    )

    frame["baseline_pnl_cents"] = np.where(
        frame["baseline_up"],
        100 * (frame["exit_bid"] - frame["entry_ask"]),
        100 * (frame["entry_bid"] - frame["exit_ask"]),
    )

    print("\n=== PHASE 7L.2 EXECUTION SIMULATION ===")
    print("Observations:", len(frame))
    print("Markets:", frame["ticker"].nunique())
    print("Sessions:", frame["session_id"].nunique())
    print("Median entry spread:", f"{entry_spread.median():.2f} cents")
    print("Median exit spread:", f"{exit_spread.median():.2f} cents")
    print(
        "Median break-even midpoint move:",
        f"{frame['breakeven_move_cents'].median():.2f} cents",
    )

    print("\n=== ALL-OBSERVATION GROSS RESULTS ===")
    print(
        "GRU mean gross P&L:",
        f"{frame['gru_pnl_cents'].mean():+.4f} cents",
    )
    print(
        "Baseline mean gross P&L:",
        f"{frame['baseline_pnl_cents'].mean():+.4f} cents",
    )

    gru_results = evaluate(
        "Two-stage GRU",
        frame,
        "gru_score",
        "gru_up",
    )

    baseline_results = evaluate(
        "Volatility + reverse momentum",
        frame,
        "volatility_score",
        "baseline_up",
    )

    print("\n=== SIDE-BY-SIDE GROSS COMPARISON ===")

    for i, fraction in enumerate(FRACTIONS):
        g = gru_results.iloc[i]
        b = baseline_results.iloc[i]

        print(
            f"Top {fraction:.0%}: "
            f"GRU {g['gross_average_cents']:+.4f}c/trade | "
            f"Baseline {b['gross_average_cents']:+.4f}c/trade | "
            f"Difference "
            f"{g['gross_average_cents'] - b['gross_average_cents']:+.4f}c"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)

    print("\n=== PHASE 7L.2 COMPLETE ===")
    print("Saved detailed trade candidates:", OUTPUT)
    print("Gross P&L only; fees and slippage excluded.")
    print("Displayed quote execution is assumed, not verified.")
    print("No model training or reserved-test access.")


if __name__ == "__main__":
    main()
