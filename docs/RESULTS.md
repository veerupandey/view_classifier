# Evaluation results

All numbers below are from this repo’s identity-split protocol (CFV seed **42**, 70/20/10 by vehicle ID: **46 / 13 / 7** cars → **16606 / 4693 / 2527** images). That is **not** the Hugging Face official CFV train/test split.

Raw JSON copies live in [`results/`](results/). Figures in [`figures/`](figures/).

## Summary table

| Run | Checkpoint (local) | CFV val | CFV test | UnsupCar test |
|---|---|---|---|---|
| CFV-only | `artifacts/cfv_convnext_s/view_classifier_best.pt` | 82.4% / 17.5° | **93.95% / 4.61°** | — |
| Joint v1 | `artifacts/joint_cfv_unsupcar/view_classifier_best.pt` | 83.3% / 16.8° | **94.5% / 4.2°** | 82.9% / 17.0°* |
| **Joint v2** | `artifacts/joint_v2_zenodo_crop/view_classifier_best.pt` (epoch **10**) | **93.6% / 3.81°** | **94.1% / 3.77°** | **82.9% / 17.0°** |

\*v1 UnsupCar test from final eval; see `results/joint_v1_metrics.json`.

### Published references (different protocols — not apples-to-apples)

| Paper | Reported |
|---|---|
| Catruna et al. 2023 (CFV) | 93.97% / 3.39° |
| Pasaulis et al. 2026 (joint) | CFV 3.77° CMAE / UnsupCar 5.38° CMAE |

## Joint v2 detail

- Recipe: Zenodo CFV boxes, expand 1.1, `RandomResizedCrop`, circular angle loss, warm-start from joint v1, 40 epochs, freeze 2.
- Best by CFV val accuracy: **epoch 10** (deploy this).
- Late epochs trade a little accuracy for lower CMAE (~3.50° on val at epoch 32–40) without beating epoch-10 accuracy.
- Per-car val (`results/joint_v2_per_car_val.json`): mean-over-cars **93.6% / 3.81°**.
  - Worst cars: `id001` 90.0%, `id059` 90.3%, `id043` 90.9%, `id054` 92.0%, `id061` 92.2%.
  - Adjacent-or-correct on those cars ≈ **99–100%**.

## Real-photo smoke demo

Twelve Wikimedia vehicle-damage photos (CC-licensed), inferred with v2:

| Confidence | Count |
|---|---|
| ≥ 60% | 8 / 12 |
| 35–60% | 3 / 12 |
| &lt; 35% | 1 / 12 (extreme wreck) |

See `figures/claim_demo_grid.png` and `results/claim_demo_preds.json`.

## Confusion matrices

- CFV-only: `figures/cfv_only_confusion.png`
- Joint v2 CFV test: `figures/joint_v2_confusion_cfv.png`
- Joint v2 UnsupCar test: `figures/joint_v2_confusion_unsupcar.png`
