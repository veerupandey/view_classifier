# View Classifier

Personal research project: a **single-image vehicle view** model for a vehicle-damage photo stack (pairs with part segmentation and damage segmentation elsewhere).

The model predicts:

- an **8-way pose bin**: `front`, `front_right`, `right`, `rear_right`, `rear`, `rear_left`, `left`, `front_left`
- a **continuous azimuth** as `(sin θ, cos θ)`

**Angle convention** (Catruna / CFV): **0° = vehicle front**, increasing **clockwise**, **90° = vehicle right** (passenger side on LHD). No horizontal flip — it would swap left and right.

Published evaluation numbers, figures, and demo outputs live under [`docs/`](docs/).  
Release weights: [`models/joint_v2_best.pt`](models/joint_v2_best.pt) via **Git LFS**. Raw training data and `artifacts/` stay **out of git** (see [`.gitignore`](.gitignore)).

## Results (v2)

| Run | Honest score |
|---|---|
| **Joint v2** (Zenodo boxes + weak crop + circular loss) | CFV val **93.6% / 3.81°**; CFV test **94.1% / 3.77°**; UnsupCar test **82.9% / 17.0°** |
| Joint v1 | CFV val **83.3% / 16.8°**; CFV test **94.5% / 4.2°** |
| CFV-only | Test **94.0% / 4.61°**; val **82.4% / 17.5°** |
| Catruna 2023 (published, different protocol) | **93.97% / 3.39°** |
| Pasaulis 2026 (published joint) | CFV **3.77°** / UnsupCar **5.38°** |

Hard val cars improved under v2: pickup `id001` **90%** (was ~54%), `id054` **92%** (was ~30%). Full tables: [`docs/RESULTS.md`](docs/RESULTS.md).

Demo figures:

- Walkaround grids: [`docs/figures/walkaround_v2_id003.png`](docs/figures/walkaround_v2_id003.png) (and `id001`, `id008`, `id054`)
- Real damage photos: [`docs/figures/claim_demo_grid.png`](docs/figures/claim_demo_grid.png)

## Repo layout

```
view_classifier/     Python package (train, eval, infer, walkaround)
tests/               Unit tests
docs/                Results, dataset notes, figures (tracked)
data/                Local JPEGs + manifests (gitignored)
artifacts/           Local checkpoints + run logs (gitignored)
```

## Quick start

Python **3.12** recommended (Torch wheels). From repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-demo.txt   # Gradio app + notebook
```

### Live demo app (best for showing people)

See the timed script in [`docs/DEMO.md`](docs/DEMO.md). Before guests arrive:

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.demo_selftest   # must PASS
PYTHONPATH=. .venv/bin/python -m view_classifier.app
# open http://127.0.0.1:7860
# optional temporary public link:
PYTHONPATH=. .venv/bin/python -m view_classifier.app --share
```

Upload a photo or click a sample in `docs/demo_images/`. Needs checkpoint  
`models/joint_v2_best.pt` (Git LFS — run `git lfs pull` after clone).

### Notebook walkthrough

```bash
PYTHONPATH=. .venv/bin/jupyter notebook notebooks/view_classifier_demo.ipynb
```

Infer on a folder of photos (CLI):

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.infer_folder \
  --ckpt artifacts/joint_v2_zenodo_crop/view_classifier_best.pt \
  --images /path/to/photos \
  --out docs/results/my_preds.json
```

8-view walkaround for one CFV car:

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.walkaround \
  --ckpt artifacts/joint_v2_zenodo_crop/view_classifier_best.pt \
  --manifest data/cfv/manifest_zenodo.json \
  --image-root data/cfv/images_v2 \
  --identity 3 --split test \
  --out docs/figures/walkaround_v2_id003.png
```

## Train (local MPS / CUDA)

Data prep and full recipes: [`docs/DATASETS.md`](docs/DATASETS.md) and [`docs/TRAINING.md`](docs/TRAINING.md).

Short form for the **v2** recipe (after JPEG export exists):

```bash
PYTHONPATH=. .venv/bin/python -m view_classifier.auto_resume -- \
  --manifest data/cfv/manifest_zenodo.json \
  --image-root data/cfv/images_v2 \
  --extra-manifest data/unsupcar/manifest.json \
  --extra-image-root data/unsupcar/images \
  --init artifacts/joint_cfv_unsupcar/view_classifier_best.pt \
  --freeze-epochs 2 --epochs 40 --lr 5e-5 \
  --angle-loss circular --angle-w 2 \
  --device mps --mem-fraction 0.55 --batch-size 8 --accum 4 \
  --resume --out-dir artifacts/joint_v2_zenodo_crop
```

Checkpoints: release weight `models/joint_v2_best.pt` (LFS); local `artifacts/*/view_classifier_last.pt` for resume only.

## License

Code in this repository: see [`LICENSE`](LICENSE).  
Datasets remain under their own licenses (CFV Apache 2.0; UnsupCar / Zenodo terms — see [`docs/DATASETS.md`](docs/DATASETS.md)).

## Not in scope (yet)

Utility / proximity heads (VIN, odometer, interior, document, wide vs close-up). Pose alone is wrong for those frames — use a gate before the view model.
