# Demo plan (stakeholder walkthrough)

**Goal:** In ~8 minutes, show that the view model works on real damage photos, explain angle/confidence, and be honest about failure modes.

**Props:** Gradio app at http://127.0.0.1:7860 · checkpoint `models/joint_v2_best.pt` (Git LFS) · samples in `docs/demo_images/`.

## Runbook (before people arrive)

```bash
cd /path/to/view_classifier
source .venv/bin/activate
pip install -r requirements-demo.txt   # once
PYTHONPATH=. python -m view_classifier.demo_selftest   # must pass
PYTHONPATH=. python -m view_classifier.app             # leave running
```

Open http://127.0.0.1:7860. Click one sample once so the model is warm.

## Script (~8 min)

| Min | What you do | What you say |
|---|---|---|
| 0–1 | Show README numbers briefly | “v2: CFV val **93.6% / 3.8°**, test **94.1% / 3.77°**. Hard trucks fixed.” |
| 1–2 | Draw compass on whiteboard / ANGLE.md | “0° = front, clockwise, 90° = right. Eight bins of 45°.” |
| 2–4 | App → sample **01** (side-impact sedan) | “Predicts **left** ~296°, high confidence. Wide exterior → model is sure.” |
| 4–5 | Sample **05** (front-crashed SLK) | “**front_left** ~297°, ~90% conf. Damage doesn’t break pose if the car shape is visible.” |
| 5–6 | Sample **09** (wrapped around tree) | “Confidence **drops**. Extreme wrecks — don’t trust pose blindly.” |
| 6–7 | Upload / drag any phone photo if available | “Same path we’d use in a claim pipeline.” |
| 7–8 | Close | “Next gate: utility/proximity (VIN, docs, close-ups). Pose alone isn’t enough for those.” |

## Sample cheat sheet

| File | Expect (approx) | Talking point |
|---|---|---|
| `01_…Side-Impact…` | `left`, high conf | Clean side view |
| `05_…Front_crashed_SLK…` | `front_left`, high conf | Front-¾ despite heavy damage |
| `07_…Dubai_02…` | `rear_left`, high conf | Rear-¾ |
| `09_…WrappedAroundTree…` | any, **low conf** | Failure mode — good |
| `11_…Howth…` | mid conf OK | Dent / tight framing |

## Don’t

- Don’t claim SOTA vs papers without noting **different splits**.
- Don’t show document-only images as “car views.”
- Don’t open training logs mid-demo.

## Fallback if app dies

```bash
PYTHONPATH=. python -m view_classifier.demo_selftest
# or notebook:
PYTHONPATH=. jupyter notebook notebooks/view_classifier_demo.ipynb
```
