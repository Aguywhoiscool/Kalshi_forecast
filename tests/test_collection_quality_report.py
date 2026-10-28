
import csv

from src.collection_quality_report import analyze_file


def test_collection_quality_counts(tmp_path):
    filepath = tmp_path / "features_test.csv"

    rows = [
        {
            "timestamp": "2026-10-28T18:00:00+00:00",
            "ticker": "TEST",
            "bid": "0.40",
            "ask": "0.42",
            "spread": "0.02",
            "midpoint": "0.41",
        },
        {
            "timestamp": "2026-10-28T18:00:01+00:00",
            "ticker": "TEST",
            "bid": "0.41",
            "ask": "0.43",
            "spread": "0.02",
            "midpoint": "0.42",
        },
        {
            "timestamp": "2026-10-28T18:00:02+00:00",
            "ticker": "TEST",
            "bid": "0.20",
            "ask": "0.40",
            "spread": "0.20",
            "midpoint": "0.30",
        },
    ]

    with filepath.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    result = analyze_file(filepath)

    assert result["rows"] == 3
    assert result["usable_rows"] == 2
    assert result["wide_rows"] == 1
    assert result["invalid_rows"] == 0
    assert result["meaningful_changes"] == 2
    assert result["meaningful_usable"] == 1

def test_wide_to_narrow_transition_is_not_usable(tmp_path):
    filepath = tmp_path / "features_transition.csv"

    rows = [
        {
            "timestamp": "2026-10-28T18:00:00+00:00",
            "ticker": "TEST",
            "bid": "0.05",
            "ask": "0.45",
            "spread": "0.40",
            "midpoint": "0.25",
        },
        {
            "timestamp": "2026-10-28T18:00:01+00:00",
            "ticker": "TEST",
            "bid": "0.34",
            "ask": "0.36",
            "spread": "0.02",
            "midpoint": "0.35",
        },
        {
            "timestamp": "2026-10-28T18:00:02+00:00",
            "ticker": "TEST",
            "bid": "0.35",
            "ask": "0.37",
            "spread": "0.02",
            "midpoint": "0.36",
        },
    ]

    with filepath.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    result = analyze_file(filepath)

    assert result["rows"] == 3
    assert result["usable_rows"] == 2
    assert result["wide_rows"] == 1

    # Both movements are meaningful.
    assert result["meaningful_changes"] == 2

    # Only the narrow-to-narrow movement qualifies.
    assert result["meaningful_usable"] == 1
