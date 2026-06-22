Local training outputs live here (gitignored).

Release weights for demos / inference are under **`models/`** (Git LFS), e.g. `models/joint_v2_best.pt`.

Expected layout after local training:

```
artifacts/
  cfv_convnext_s/view_classifier_best.pt
  joint_cfv_unsupcar/view_classifier_best.pt
  joint_v2_zenodo_crop/view_classifier_best.pt
  joint_v2_zenodo_crop/view_classifier_last.pt   # resume only
```
