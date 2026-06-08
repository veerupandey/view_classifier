Local training outputs and weights live here (gitignored).

Expected layout after training:

```
artifacts/
  cfv_convnext_s/view_classifier_best.pt
  joint_cfv_unsupcar/view_classifier_best.pt
  joint_v2_zenodo_crop/view_classifier_best.pt   # recommended
  joint_v2_zenodo_crop/view_classifier_last.pt   # resume
```

Published metrics and figures are copied to `docs/results/` and `docs/figures/` for git.
