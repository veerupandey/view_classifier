# Training notes

## Hardware

Developed on Apple Silicon (M4 Max, MPS). Defaults keep system RAM near **`--mem-fraction 0.55`**. Long runs use `--batch-size 8 --accum 4` (probe may reduce batch further).

## Checkpoints (local only)

| File | Role |
|---|---|
| `view_classifier_best.pt` | Best CFV val accuracy |
| `view_classifier_last.pt` | Full train state each epoch (resume) |

Both are gitignored. Copy them yourself if you need weights on another machine.

## Resume / auto-resume

```bash
# single resume
PYTHONPATH=. .venv/bin/python -m view_classifier.train ... --resume --out-dir artifacts/joint_v2_zenodo_crop

# loop until history has N epochs and metrics.json exists
PYTHONPATH=. .venv/bin/python -m view_classifier.auto_resume --sleep 30 -- \
  ...same train args... --resume --out-dir artifacts/joint_v2_zenodo_crop
```

## CFV-only baseline

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.train \
  --batch-size 8 --accum 4 --epochs 30 \
  --out-dir artifacts/cfv_convnext_s
```

## Joint v1

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.train \
  --extra-manifest data/unsupcar/manifest.json \
  --extra-image-root data/unsupcar/images \
  --init artifacts/cfv_convnext_s/view_classifier_best.pt \
  --freeze-epochs 0 --epochs 20 --lr 3e-5 \
  --batch-size 8 --accum 4 \
  --out-dir artifacts/joint_cfv_unsupcar
```

## Joint v2 (recommended)

See root [`README.md`](../README.md). Requires Zenodo-box JPEGs in `data/cfv/images_v2/`.

## Smoke

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.train --smoke --batch-size 4 --accum 2 --out-dir artifacts/cfv_smoke
```
