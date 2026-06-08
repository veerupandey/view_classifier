"""Build an identity-level UnsupCar manifest from Zenodo-harmonized labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from view_classifier.data import build_manifest

ZENOD_CSV = Path("data/unsupcar/annotations/freiburg_static_cars_52_v1.1/full_dataset.csv")
RAW_DIR = Path("data/unsupcar/raw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ZENOD_CSV)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/unsupcar"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    missing = 0
    with args.csv.open() as f:
        for rec in csv.DictReader(f):
            rel = rec["image_path"]
            src = args.raw_dir / rel
            if not src.exists():
                missing += 1
                continue
            rows.append(
                {
                    "identity": int(rec["identity"]),
                    "angle": float(rec["angle"]),
                    "x1": rec["x1"],
                    "y1": rec["y1"],
                    "x2": rec["x2"],
                    "y2": rec["y2"],
                    "source_path": rel,
                }
            )
    if missing:
        print(f"[unsupcar] skipped {missing} rows with missing PNGs", flush=True)
    if not rows:
        raise SystemExit(f"no UnsupCar rows from {args.csv} + {args.raw_dir}")

    manifest = build_manifest(
        rows,
        seed=args.seed,
        dataset="unsupcar",
        source="unsupcar",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "manifest.json"
    path.write_text(json.dumps(manifest))
    summary = {k: v for k, v in manifest.items() if k != "records"}
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[unsupcar] wrote {path} ({len(manifest['records'])} rows)", flush=True)


if __name__ == "__main__":
    main()
