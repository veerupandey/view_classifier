"""Square-crop + mirrored pad, then 224 resize. No horizontal flip, no rotation."""

from __future__ import annotations

import random

from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def square_reflect_crop(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    expand: float = 1.0,
) -> Image.Image:
    """Crop to the car bbox when valid, then pad to a square with reflected borders."""
    img = image.convert("RGB")
    w, h = img.size
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if expand != 1.0:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = (x2 - x1) * expand, (y2 - y1) * expand
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        if (x2 - x1) >= 8 and (y2 - y1) >= 8:
            img = img.crop((x1, y1, x2, y2))
            w, h = img.size
    side = max(w, h)
    pad_x = side - w
    pad_y = side - h
    padding = (pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2)
    return transforms.functional.pad(img, padding, padding_mode="reflect")


def jitter_bbox(
    bbox: tuple[int, int, int, int],
    scale_range: tuple[float, float] = (0.9, 1.2),
) -> tuple[int, int, int, int]:
    """Paper-style independent-ish box scale jitter around the car."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    sx = random.uniform(*scale_range)
    sy = random.uniform(*scale_range)
    bw, bh = (x2 - x1) * sx, (y2 - y1) * sy
    return (
        int(round(cx - bw / 2.0)),
        int(round(cy - bh / 2.0)),
        int(round(cx + bw / 2.0)),
        int(round(cy + bh / 2.0)),
    )


def train_tensorize(weak_crop: bool = True) -> transforms.Compose:
    # Catruna/Pasaulis weak aug: random crop + color jitter. Never flip/rotate.
    ops: list = []
    if weak_crop:
        ops.append(transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.95, 1.05)))
    else:
        ops.append(transforms.Resize((224, 224)))
    ops.extend(
        [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def eval_tensorize() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
