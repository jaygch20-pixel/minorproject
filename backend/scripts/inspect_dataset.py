from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.dataset import inspect_archive

ARCHIVE_PATH = BACKEND_DIR.parent / "archive.zip"
OUTPUT_PATH = BACKEND_DIR / "data" / "dataset_inspection.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ISIC2020 metadata and images inside archive.zip.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    summary = inspect_archive(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nInspection report written to: {args.output}")
    print("Images were inspected in-place; the archive was not extracted or modified.")


if __name__ == "__main__":
    main()
