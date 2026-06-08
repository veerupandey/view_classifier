"""Identity-level splits. Prefer local JPEGs; Hugging Face is CFV export-only."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from view_classifier.labels import POSE_CLASSES, angle_to_pose_idx, angle_to_sincos, wrap_deg
from view_classifier.transforms_cfv import eval_tensorize, square_reflect_crop, train_tensorize

HF_DATASET = "fort-cyber/CFV-Dataset"


def _bbox(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    keys = ("x1", "y1", "x2", "y2")
    if not all(k in row and row[k] is not None for k in keys):
        return None
    try:
        x1, y1, x2, y2 = (int(float(row[k])) for k in keys)
    except (TypeError, ValueError):
        return None
    return x1, y1, x2, y2


def split_identities(identities: list[int], seed: int = 42) -> dict[str, list[int]]:
    ids = sorted(set(identities))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.20)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    if not test:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


def build_manifest(
    rows: list[dict[str, Any]],
    seed: int = 42,
    dataset: str = HF_DATASET,
    source: str = "cfv",
) -> dict[str, Any]:
    identities = [int(r["identity"]) for r in rows]
    splits = split_identities(identities, seed=seed)
    id_to_split = {i: split for split, members in splits.items() for i in members}
    pose_counts = {name: 0 for name in POSE_CLASSES}
    records = []
    for idx, row in enumerate(rows):
        identity = int(row["identity"])
        angle = wrap_deg(float(row["angle"]))
        pose_idx = angle_to_pose_idx(angle)
        pose = POSE_CLASSES[pose_idx]
        pose_counts[pose] += 1
        rec = {
            "index": idx,
            "identity": identity,
            "angle": angle,
            "pose": pose,
            "pose_idx": pose_idx,
            "split": id_to_split[identity],
            "bbox": list(_bbox(row)) if _bbox(row) else None,
            "source": source,
        }
        if row.get("path"):
            rec["path"] = row["path"]
        if row.get("source_path"):
            rec["source_path"] = row["source_path"]
        records.append(rec)
    return {
        "dataset": dataset,
        "source": source,
        "seed": seed,
        "n_identities": len(set(identities)),
        "split_identities": splits,
        "split_counts": {
            split: sum(1 for r in records if r["split"] == split) for split in ("train", "val", "test")
        },
        "pose_counts": pose_counts,
        "records": records,
    }


class CfvViewDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        train: bool,
        image_root: Path | None = None,
        weak_crop: bool = True,
    ):
        missing = [r for r in records if not r.get("path")]
        if missing:
            raise ValueError(
                f"{len(missing)} records have no JPEG path. Run: python -m view_classifier.export_jpegs"
            )
        self.records = records
        self.image_root = image_root
        self.tensorize = train_tensorize(weak_crop=weak_crop) if train else eval_tensorize()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        path = Path(rec["path"])
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path
        image = Image.open(path).convert("RGB")
        bbox = tuple(rec["bbox"]) if rec.get("bbox") else None
        # Exported JPEGs are already square-cropped; still clip if a raw file sneaks in.
        if image.size[0] != image.size[1]:
            image = square_reflect_crop(image, bbox)
        x = self.tensorize(image)
        sin_v, cos_v = angle_to_sincos(rec["angle"])
        return {
            "image": x,
            "pose_idx": rec["pose_idx"],
            "angle": rec["angle"],
            "sincos": (sin_v, cos_v),
            "identity": rec["identity"],
        }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
