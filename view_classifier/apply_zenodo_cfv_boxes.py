"""Merge Zenodo-harmonized CFV boxes into our identity-split manifest.

Keeps the locked seed-42 identity splits and JPEG paths. Matches Zenodo rows by
(identity, wrapped angle) because the CSV is not HF-index ordered.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from view_classifier.data import load_manifest
from view_classifier.labels import wrap_deg


def _angle_key(angle: float) -> float:
    return round(wrap_deg(angle), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/cfv/manifest.json"))
    parser.add_argument(
        "--zenodo-csv",
        type=Path,
        default=Path("data/unsupcar/annotations/CFV-Dataset/full_dataset.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/cfv/manifest_zenodo.json"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    by_key: dict[tuple[int, float], list[dict]] = defaultdict(list)
    with args.zenodo_csv.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["identity"]), _angle_key(float(row["angle"])))
            by_key[key].append(row)

    updated = 0
    missing = 0
    multi = 0
    for rec in manifest["records"]:
        key = (int(rec["identity"]), _angle_key(float(rec["angle"])))
        cands = by_key.get(key, [])
        if not cands:
            # Prefer exact match; fall back to nearest 0.1° within 1°.
            found = []
            for step in range(1, 11):
                for sign in (-1, 1):
                    k = (key[0], _angle_key(rec["angle"] + sign * step * 0.1))
                    if by_key.get(k):
                        found = by_key[k]
                        break
                if found:
                    break
            cands = found
        if not cands:
            missing += 1
            continue
        if len(cands) > 1:
            multi += 1
        z = cands.pop(0)
        rec["bbox"] = [
            int(float(z["x1"])),
            int(float(z["y1"])),
            int(float(z["x2"])),
            int(float(z["y2"])),
        ]
        rec["zenodo_path"] = z["image_path"]
        updated += 1

    manifest["bbox_source"] = "zenodo-20173357"
    args.out.write_text(json.dumps(manifest))
    summary = {
        "out": str(args.out),
        "updated": updated,
        "missing": missing,
        "ambiguous_angle_matches": multi,
        "n_records": len(manifest["records"]),
        "split_counts": manifest["split_counts"],
        "bbox_source": manifest["bbox_source"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    if missing:
        raise SystemExit(f"failed to match {missing} records to Zenodo boxes")


if __name__ == "__main__":
    main()
