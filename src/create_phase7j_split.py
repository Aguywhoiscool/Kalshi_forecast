
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed/datasets/v2_phase7j")
METADATA_PATH = DATA_DIR / "gru_dataset_combined.metadata.csv"
SPLIT_DIR = DATA_DIR / "splits"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
EMBARGO_SECONDS = 70


def main():
    if SPLIT_DIR.exists() and any(SPLIT_DIR.iterdir()):
        raise FileExistsError(
            f"{SPLIT_DIR} already contains files. "
            "Refusing to overwrite an existing split."
        )

    df = pd.read_csv(METADATA_PATH)

    df["input_start"] = pd.to_datetime(
        df["input_start"], utc=True
    )
    df["target_end"] = pd.to_datetime(
        df["target_end"], utc=True
    )

    sessions = (
        df.groupby("session_id")
        .agg(
            start=("input_start", "min"),
            end=("target_end", "max"),
            windows=("session_id", "size"),
        )
        .sort_values(["start", "end"])
        .reset_index()
    )

    total = len(df)
    cumulative = sessions["windows"].cumsum()

    train_cut = (
        int((cumulative < total * TRAIN_FRACTION).sum()) + 1
    )

    val_cut = (
        int((
            cumulative < total * (
                TRAIN_FRACTION + VALIDATION_FRACTION
            )
        ).sum()) + 1
    )

    if not (0 < train_cut < val_cut < len(sessions)):
        raise ValueError("Invalid split boundaries.")

    train_sessions = set(
        sessions.iloc[:train_cut]["session_id"]
    )
    val_sessions = set(
        sessions.iloc[train_cut:val_cut]["session_id"]
    )
    test_sessions = set(
        sessions.iloc[val_cut:]["session_id"]
    )

    train_mask = df["session_id"].isin(train_sessions)
    val_mask = df["session_id"].isin(val_sessions)
    test_mask = df["session_id"].isin(test_sessions)

    # Check chronological separation before applying embargo.
    train_end = df.loc[train_mask, "target_end"].max()
    val_start = df.loc[val_mask, "input_start"].min()
    val_end = df.loc[val_mask, "target_end"].max()
    test_start = df.loc[test_mask, "input_start"].min()

    if val_start <= train_end or test_start <= val_end:
        raise ValueError(
            "Sessions overlap across chronological splits."
        )

    embargo = pd.Timedelta(seconds=EMBARGO_SECONDS)

    # Require the beginning of every validation window to be
    # at least 70 seconds after the final training target.
    val_mask = val_mask & (
        df["input_start"] >= train_end + embargo
    )

    if not val_mask.any():
        raise ValueError("Embargo removed all validation windows.")

    # Use the retained validation end for the test embargo.
    retained_val_end = df.loc[val_mask, "target_end"].max()

    test_mask = test_mask & (
        df["input_start"] >= retained_val_end + embargo
    )

    if not test_mask.any():
        raise ValueError("Embargo removed all test windows.")

    masks = {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }

    indices = {
        name: np.flatnonzero(mask.to_numpy())
        for name, mask in masks.items()
    }

    all_indices = np.concatenate(list(indices.values()))

    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Duplicate indices across splits.")

    original_counts = {
        "train": int(df["session_id"].isin(train_sessions).sum()),
        "val": int(df["session_id"].isin(val_sessions).sum()),
        "test": int(df["session_id"].isin(test_sessions).sum()),
    }

    print("\n=== PHASE 7J SPLIT MANIFEST ===")
    print("Original windows:", total)
    print("Embargo:", EMBARGO_SECONDS, "seconds")

    for name, idx in indices.items():
        subset = df.iloc[idx]

        print(f"\n=== {name.upper()} ===")
        print("Windows:", len(idx))
        print("Sessions:", subset["session_id"].nunique())
        print("First:", subset["input_start"].min())
        print("Last:", subset["target_end"].max())
        print(
            "Removed by embargo:",
            original_counts[name] - len(idx),
        )

    train_final_end = df.iloc[indices["train"]]["target_end"].max()
    val_final_start = df.iloc[indices["val"]]["input_start"].min()
    val_final_end = df.iloc[indices["val"]]["target_end"].max()
    test_final_start = df.iloc[indices["test"]]["input_start"].min()

    gap1 = (
        val_final_start - train_final_end
    ).total_seconds()

    gap2 = (
        test_final_start - val_final_end
    ).total_seconds()

    print("\n=== FINAL BOUNDARY CHECK ===")
    print(f"Train -> Validation: {gap1:.2f} seconds")
    print(f"Validation -> Test: {gap2:.2f} seconds")

    if gap1 < EMBARGO_SECONDS or gap2 < EMBARGO_SECONDS:
        raise ValueError("Embargo verification failed.")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    for name, idx in indices.items():
        np.save(SPLIT_DIR / f"{name}_indices.npy", idx)

    print("\nSplit index files saved.")
    print("Test labels and outcomes were not evaluated.")


if __name__ == "__main__":
    main()
