"""Per-car accuracy / CMAE report for fair identity-split scoring."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from view_classifier.data import CfvViewDataset, load_manifest
from view_classifier.labels import POSE_CLASSES
from view_classifier.model import ConvNeXtViewClassifier, circular_deg_error
from view_classifier.train import collate, resolve_device


@torch.no_grad()
def score(model, records, image_root, device, batch_size: int):
    ds = CfvViewDataset(records, train=False, image_root=image_root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    by = defaultdict(lambda: {"n": 0, "ok": 0, "err": 0.0, "adj": 0})
    for batch in loader:
        logits, sincos = model(batch["image"].to(device))
        pred = logits.argmax(1).cpu()
        errs = circular_deg_error(sincos.cpu(), batch["angle"])
        for i, identity in enumerate(batch["identity"].tolist()):
            p = int(pred[i])
            t = int(batch["pose_idx"][i])
            circ = min(abs(p - t), 8 - abs(p - t))
            b = by[identity]
            b["n"] += 1
            b["ok"] += int(p == t)
            b["err"] += float(errs[i])
            b["adj"] += int(circ <= 1)
    cars = []
    for identity, b in sorted(by.items()):
        cars.append(
            {
                "identity": identity,
                "n": b["n"],
                "accuracy": round(b["ok"] / b["n"], 4),
                "cmae_deg": round(b["err"] / b["n"], 3),
                "adjacent_or_correct": round(b["adj"] / b["n"], 4),
            }
        )
    n = sum(c["n"] for c in cars)
    micro_acc = sum(c["accuracy"] * c["n"] for c in cars) / n
    micro_cmae = sum(c["cmae_deg"] * c["n"] for c in cars) / n
    macro_acc = sum(c["accuracy"] for c in cars) / len(cars)
    macro_cmae = sum(c["cmae_deg"] for c in cars) / len(cars)
    return {
        "n_images": n,
        "n_cars": len(cars),
        "micro_accuracy": round(micro_acc, 4),
        "micro_cmae_deg": round(micro_cmae, 3),
        "mean_over_cars_accuracy": round(macro_acc, 4),
        "mean_over_cars_cmae_deg": round(macro_cmae, 3),
        "worst_cars": sorted(cars, key=lambda c: c["accuracy"])[:5],
        "cars": cars,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/cfv/manifest.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/cfv/images"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device()
    manifest = load_manifest(args.manifest)
    records = [r for r in manifest["records"] if r["split"] == args.split]
    model = ConvNeXtViewClassifier(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    report = {
        "ckpt": str(args.ckpt),
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "split": args.split,
        "classes": list(POSE_CLASSES),
        **score(model, records, args.image_root, device, args.batch_size),
    }
    text = json.dumps(
        {k: v for k, v in report.items() if k != "cars"},
        indent=2,
    )
    print(text, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
