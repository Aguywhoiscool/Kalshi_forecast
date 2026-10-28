
"""Freeze completed feature recordings for a new dataset version.

Run:
    python -m src.freeze_dataset
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import shutil

SOURCE_DIR = Path("data/processed")
VERSION_DIR = SOURCE_DIR / "datasets" / "v2"
FROZEN_DIR = VERSION_DIR / "features"
MANIFEST = VERSION_DIR / "manifest.csv"

# Avoid copying the file currently being written by the collector.
MIN_FILE_AGE_MINUTES = 10


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    # Don't silently change an already frozen dataset.
    if MANIFEST.exists():
        print("Version 2 is already frozen.")
        print("Manifest:", MANIFEST)
        return

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=MIN_FILE_AGE_MINUTES
    )

    files = sorted(SOURCE_DIR.glob("features_*.csv"))

    eligible = [
        path for path in files
        if datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ) < cutoff
    ]

    print("Feature files found:", len(files))
    print("Completed files eligible:", len(eligible))

    if not eligible:
        print("No completed recordings found.")
        return

    manifest_rows = []
    total_bytes = 0

    for source in eligible:
        destination = FROZEN_DIR / source.name

        if destination.exists():
            if sha256_file(source) != sha256_file(destination):
                raise RuntimeError(
                    f"Existing frozen file differs: {destination}"
                )
        else:
            shutil.copy2(source, destination)

        digest = sha256_file(destination)
        size = destination.stat().st_size
        total_bytes += size

        manifest_rows.append({
            "filename": source.name,
            "size_bytes": size,
            "sha256": digest,
        })

    # Write manifest last so it marks a completed snapshot.
    with MANIFEST.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["filename", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nVERSION 2 SNAPSHOT COMPLETE")
    print("Frozen recordings:", len(manifest_rows))
    print("Total size (MB):", round(total_bytes / 1048576, 2))
    print("Folder:", FROZEN_DIR)
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
