"""Crop UnsupCar PNGs once and write 256px square JPEGs for fast MPS training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from view_classifier.data import load_manifest
from view_classifier.transforms_cfv import square_reflect_crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/unsupcar/manifest.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/unsupcar/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/unsupcar/images"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(manifest["records"])
    print(f"[export] {n} UnsupCar rows -> {args.out_dir}", flush=True)

    for i, rec in enumerate(manifest["records"]):
        rel = Path(rec["split"]) / f"id{rec['identity']:03d}_{rec['index']:05d}.jpg"
        dest = args.out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            src = args.raw_dir / rec["source_path"]
            image = Image.open(src).convert("RGB")
            bbox = tuple(rec["bbox"]) if rec.get("bbox") else None
            image = square_reflect_crop(image, bbox)
            image = image.resize((args.size, args.size), Image.Resampling.LANCZOS)
            image.save(dest, format="JPEG", quality=90, optimize=True)
        rec["path"] = str(rel)
        if (i + 1) % 500 == 0:
            print(f"[export] {i + 1}/{n}", flush=True)

    args.manifest.write_text(json.dumps(manifest))
    print(f"[export] done. updated {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
