"""Build a labeled prediction grid from infer_folder JSON + image folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--max", type=int, default=12)
    args = parser.parse_args()

    report = json.loads(args.preds.read_text())
    rows = report["predictions"][: args.max]
    n = len(rows)
    cols = args.cols
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.4 * cols, 3.6 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, row in zip(axes, rows):
        path = args.images / row["path"]
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        conf = row["conf"]
        color = "#1b7f3a" if conf >= 0.6 else "#b36b00" if conf >= 0.35 else "#b00020"
        ax.set_title(
            f"{row['pred']}  {row['pred_ang']:.0f}°\nconf {conf:.0%}",
            fontsize=10,
            color=color,
        )
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(
        f"View classifier on real damage photos  (n={n})  "
        f"green=confident ≥60%, amber=35–60%, red=<35%",
        fontsize=12,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
