# Datasets

Raw images and large manifests are **gitignored** under `data/`. Rebuild locally with the package entry points below.

## CFV (primary)

- Source: Hugging Face dataset id `fort-cyber/CFV-Dataset` (Apache 2.0).
- Content: ~23.8k images, 66 vehicle identities, continuous 0–360° azimuth.
- Local paths:
  - `data/cfv/images/` — v1 export (original boxes)
  - `data/cfv/images_v2/` — Zenodo-box export, expand 1.1
  - `data/cfv/manifest.json` / `manifest_zenodo.json`

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.prepare_cfv
PYTHONPATH=. .venv/bin/python -m view_classifier.apply_zenodo_cfv_boxes
PYTHONPATH=. .venv/bin/python -m view_classifier.export_jpegs \
  --manifest data/cfv/manifest_zenodo.json \
  --out-dir data/cfv/images_v2 \
  --expand 1.1
```

Split: identity-level **70/20/10**, seed **42** (locked for comparability).

## UnsupCar (joint)

- Freiburg Static Cars 52 v1.1 images + Zenodo harmonized angles ([doi:10.5281/zenodo.20173357](https://doi.org/10.5281/zenodo.20173357)).
- Use **harmonized** angles (same 0°=front clockwise convention as CFV). Do not use original Freiburg angles.
- Local: `data/unsupcar/images/`, `data/unsupcar/manifest.json`

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.prepare_unsupcar
PYTHONPATH=. .venv/bin/python -m view_classifier.export_unsupcar
```

## Angle convention

See [`ANGLE.md`](ANGLE.md).

## Demo photo set (optional)

`data/claim_demo_clean/` is a small local folder of Wikimedia damage photos used for `docs/figures/claim_demo_grid.png`. Not required for training.
