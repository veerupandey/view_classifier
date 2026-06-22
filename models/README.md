# Release checkpoints (Git LFS)

| File | What |
|---|---|
| `joint_v2_best.pt` | **Recommended** — joint CFV+UnsupCar v2 (Zenodo boxes, circular loss), best CFV val epoch |

Clone with LFS:

```bash
git lfs install
git clone <repo-url>
# or after a normal clone:
git lfs pull
```

Size ≈ **189 MB** per file. Training-time `artifacts/**/view_classifier_last.pt` stay local-only (not in git).
