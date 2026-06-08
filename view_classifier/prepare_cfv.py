"""Download CFV from Hugging Face and write an identity-level 70/20/10 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

from view_classifier.data import HF_DATASET, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/cfv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "hf_cache"
    print(f"[cfv] downloading {HF_DATASET} into {cache_dir}", flush=True)
    raw = load_dataset(HF_DATASET, cache_dir=str(cache_dir))
    parts = [raw[k] for k in ("train", "test") if k in raw]
    ds = concatenate_datasets(parts)
    meta = ds.remove_columns("image") if "image" in ds.column_names else ds
    print(f"[cfv] {len(ds)} images, columns={ds.column_names}", flush=True)

    identities = list(meta["identity"])
    angles = list(meta["angle"])
    x1 = list(meta["x1"]) if "x1" in meta.column_names else [None] * len(meta)
    y1 = list(meta["y1"]) if "y1" in meta.column_names else [None] * len(meta)
    x2 = list(meta["x2"]) if "x2" in meta.column_names else [None] * len(meta)
    y2 = list(meta["y2"]) if "y2" in meta.column_names else [None] * len(meta)
    rows = [
        {
            "identity": int(identities[i]),
            "angle": float(angles[i]),
            "x1": x1[i],
            "y1": y1[i],
            "x2": x2[i],
            "y2": y2[i],
        }
        for i in range(len(meta))
    ]

    manifest = build_manifest(rows, seed=args.seed)
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    summary = {k: v for k, v in manifest.items() if k != "records"}
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[cfv] wrote {manifest_path} ({len(manifest['records'])} rows)", flush=True)


if __name__ == "__main__":
    main()
