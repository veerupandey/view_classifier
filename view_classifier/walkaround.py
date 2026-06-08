"""Render an 8-view walkaround grid for one car identity (demo / QA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from view_classifier.data import load_manifest
from view_classifier.labels import POSE_CLASSES, angle_to_pose_idx, sincos_to_deg, wrap_deg
from view_classifier.model import ConvNeXtViewClassifier, circular_deg_error
from view_classifier.train import resolve_device
from view_classifier.transforms_cfv import eval_tensorize


def _pick_bin_representatives(records: list[dict], n_bins: int = 8) -> list[dict]:
    """One image nearest each bin center (0°, 45°, …)."""
    centers = [i * (360.0 / n_bins) for i in range(n_bins)]
    by_bin: list[list[dict]] = [[] for _ in range(n_bins)]
    for rec in records:
        by_bin[angle_to_pose_idx(float(rec["angle"]))].append(rec)
    picks: list[dict] = []
    for i, center in enumerate(centers):
        pool = by_bin[i] or records
        best = min(
            pool,
            key=lambda r: min(
                abs(wrap_deg(float(r["angle"]) - center)),
                360.0 - abs(wrap_deg(float(r["angle"]) - center)),
            ),
        )
        picks.append(best)
    return picks


@torch.no_grad()
def walkaround(
    *,
    ckpt: Path,
    manifest: Path,
    image_root: Path,
    identity: int,
    split: str | None,
    out: Path,
    device: torch.device,
) -> dict:
    data = load_manifest(manifest)
    records = [
        r
        for r in data["records"]
        if int(r["identity"]) == identity and (split is None or r["split"] == split)
    ]
    if not records:
        raise SystemExit(f"no records for identity={identity} split={split}")

    picks = _pick_bin_representatives(records)
    model = ConvNeXtViewClassifier(pretrained=False).to(device)
    loaded = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(loaded["model_state"])
    model.eval()
    tfm = eval_tensorize()

    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    for ax, rec in zip(axes.flat, picks):
        rel = Path(rec["path"])
        img_path = image_root / rel
        image = Image.open(img_path).convert("RGB")
        tensor = tfm(image).unsqueeze(0).to(device)
        logits, sincos = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_i = int(probs.argmax().item())
        conf = float(probs[pred_i].item())
        pred_ang = sincos_to_deg(float(sincos[0, 0]), float(sincos[0, 1]))
        true_i = angle_to_pose_idx(float(rec["angle"]))
        true_ang = wrap_deg(float(rec["angle"]))
        err = float(
            circular_deg_error(sincos.cpu(), torch.tensor([true_ang])).item()
        )
        ok = pred_i == true_i
        color = "#1b7f3a" if ok else "#b00020"
        ax.imshow(image)
        ax.set_title(
            f"true {POSE_CLASSES[true_i]} {true_ang:.0f}°\n"
            f"pred {POSE_CLASSES[pred_i]} {pred_ang:.0f}°  ({conf:.0%})\n"
            f"err {err:.1f}°",
            fontsize=9,
            color=color,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        rows.append(
            {
                "path": str(rel),
                "true": POSE_CLASSES[true_i],
                "true_ang": round(true_ang, 1),
                "pred": POSE_CLASSES[pred_i],
                "pred_ang": round(pred_ang, 1),
                "conf": round(conf, 3),
                "err": round(err, 2),
                "ok": ok,
            }
        )

    n_ok = sum(1 for r in rows if r["ok"])
    mean_err = sum(r["err"] for r in rows) / len(rows)
    fig.suptitle(
        f"v2 walkaround id{identity:03d}  "
        f"({n_ok}/8 bins correct, mean CMAE {mean_err:.1f}°)  "
        f"ckpt={ckpt.name} ep={loaded.get('epoch')}",
        fontsize=12,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)

    summary = {
        "identity": identity,
        "split": split or records[0]["split"],
        "ckpt": str(ckpt),
        "ckpt_epoch": loaded.get("epoch"),
        "out": str(out),
        "bins_correct": n_ok,
        "mean_cmae_deg": round(mean_err, 2),
        "views": rows,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/cfv/manifest_zenodo.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/cfv/images_v2"))
    parser.add_argument("--identity", type=int, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    out = args.out or Path(f"artifacts/examples/walkaround_v2_id{args.identity:03d}.png")
    device = resolve_device(args.device)
    summary = walkaround(
        ckpt=args.ckpt,
        manifest=args.manifest,
        image_root=args.image_root,
        identity=args.identity,
        split=args.split,
        out=out,
        device=device,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "views"}, indent=2), flush=True)
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()
