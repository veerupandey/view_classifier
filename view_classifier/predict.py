"""Single-image inference helpers shared by the demo app and notebook."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from view_classifier.labels import POSE_CLASSES, sincos_to_deg
from view_classifier.model import ConvNeXtViewClassifier
from view_classifier.train import resolve_device
from view_classifier.transforms_cfv import eval_tensorize, square_reflect_crop

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "models" / "joint_v2_best.pt"
# Fallback if someone only has a local training run
_FALLBACK_CKPT = REPO_ROOT / "artifacts" / "joint_v2_zenodo_crop" / "view_classifier_best.pt"
DEMO_IMAGE_DIR = REPO_ROOT / "docs" / "demo_images"


@lru_cache(maxsize=2)
def load_model(ckpt: str, device: str = "auto"):
    requested = Path(ckpt)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(REPO_ROOT / requested)
    candidates.extend([DEFAULT_CKPT, _FALLBACK_CKPT])
    path = next((p for p in candidates if p.exists()), requested)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {requested}\n"
            "Expected models/joint_v2_best.pt (Git LFS) or a local artifacts/ path."
        )
    dev = resolve_device(device)
    model = ConvNeXtViewClassifier(pretrained=False).to(dev)
    loaded = torch.load(path, map_location=dev, weights_only=False)
    model.load_state_dict(loaded["model_state"])
    model.eval()
    return model, dev, loaded.get("epoch"), eval_tensorize()


@torch.no_grad()
def predict_image(
    image: Image.Image,
    *,
    ckpt: str | Path = DEFAULT_CKPT,
    device: str = "auto",
    square_pad: bool = True,
) -> dict:
    """Return pose, angle, confidence, and full probability dict."""
    model, dev, epoch, tfm = load_model(str(ckpt), device)
    rgb = image.convert("RGB")
    if square_pad:
        rgb = square_reflect_crop(rgb, bbox=None)
    tensor = tfm(rgb).unsqueeze(0).to(dev)
    logits, sincos = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_i = int(probs.argmax().item())
    conf = float(probs[pred_i].item())
    pred_ang = sincos_to_deg(float(sincos[0, 0]), float(sincos[0, 1]))
    return {
        "pred": POSE_CLASSES[pred_i],
        "pred_ang": round(pred_ang, 1),
        "conf": round(conf, 3),
        "probs": {name: round(float(probs[i].item()), 4) for i, name in enumerate(POSE_CLASSES)},
        "ckpt_epoch": epoch,
        "classes": list(POSE_CLASSES),
    }


def list_demo_images(demo_dir: Path | None = None) -> list[Path]:
    root = demo_dir or DEMO_IMAGE_DIR
    if not root.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def format_prediction(result: dict) -> str:
    conf_pct = 100.0 * result["conf"]
    lines = [
        f"**View:** `{result['pred']}`",
        f"**Angle:** `{result['pred_ang']:.1f}°`  (0°=front, clockwise, 90°=right)",
        f"**Confidence:** `{conf_pct:.1f}%`",
    ]
    if result.get("ckpt_epoch") is not None:
        lines.append(f"_Checkpoint epoch {result['ckpt_epoch']}_")
    return "\n\n".join(lines)
