"""Decode CFV once from Hugging Face and write 256px square JPEGs for fast MPS training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from PIL import Image

from view_classifier.data import HF_DATASET, load_manifest
from view_classifier.transforms_cfv import square_reflect_crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/cfv/manifest.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/cfv/images"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--expand",
        type=float,
        default=1.0,
        help="Expand bbox before square crop (1.1 leaves room for train RandomResizedCrop)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing JPEGs")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] loading {HF_DATASET}", flush=True)
    raw = load_dataset(HF_DATASET)
    hf = concatenate_datasets([raw[k] for k in ("train", "test") if k in raw])
    print(
        f"[export] {len(hf)} source rows -> {args.out_dir} expand={args.expand} force={args.force}",
        flush=True,
    )

    for i, rec in enumerate(manifest["records"]):
        rel = Path(rec["split"]) / f"id{rec['identity']:03d}_{rec['index']:05d}.jpg"
        dest = args.out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.force or not dest.exists():
            row = hf[int(rec["index"])]
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
            bbox = tuple(rec["bbox"]) if rec.get("bbox") else None
            image = square_reflect_crop(image, bbox, expand=args.expand)
            image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
            image.save(dest, format="JPEG", quality=90, optimize=True)
        rec["path"] = str(rel)
        if (i + 1) % 500 == 0:
            print(f"[export] {i + 1}/{len(manifest['records'])}", flush=True)

    args.manifest.write_text(json.dumps(manifest))
    print(f"[export] done. updated {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
