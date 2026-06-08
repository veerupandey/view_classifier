"""Run view classifier on a folder of claim / phone photos (no GT required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from view_classifier.labels import POSE_CLASSES, sincos_to_deg
from view_classifier.model import ConvNeXtViewClassifier
from view_classifier.train import resolve_device
from view_classifier.transforms_cfv import eval_tensorize, square_reflect_crop

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True, help="Folder of photos")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--no-square-pad",
        action="store_true",
        help="Skip reflect square pad (use if images are already cropped/square)",
    )
    args = parser.parse_args()

    paths = sorted(
        p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    device = resolve_device(args.device)
    model = ConvNeXtViewClassifier(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tfm = eval_tensorize()

    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if not args.no_square_pad:
            image = square_reflect_crop(image, bbox=None)
        tensor = tfm(image).unsqueeze(0).to(device)
        logits, sincos = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_i = int(probs.argmax().item())
        rows.append(
            {
                "path": str(path.relative_to(args.images)),
                "pred": POSE_CLASSES[pred_i],
                "pred_ang": round(sincos_to_deg(float(sincos[0, 0]), float(sincos[0, 1])), 1),
                "conf": round(float(probs[pred_i].item()), 3),
                "probs": {
                    name: round(float(probs[i].item()), 3) for i, name in enumerate(POSE_CLASSES)
                },
            }
        )

    report = {
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt.get("epoch"),
        "images": str(args.images),
        "n": len(rows),
        "predictions": rows,
    }
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
