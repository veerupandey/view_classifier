"""Self-test the demo path: checkpoint, sample images, Gradio predict API.

Exit 0 only if all critical checks pass. Run before a live demo:

  PYTHONPATH=. .venv/bin/python -m view_classifier.demo_selftest
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from view_classifier.predict import (
    DEFAULT_CKPT,
    DEMO_IMAGE_DIR,
    list_demo_images,
    predict_image,
)

# Soft expectations for curated Wikimedia samples (pose must match; conf band).
EXPECTATIONS: dict[str, dict] = {
    "01_wiki_00050_Vehicle_After_a_Side-Impact_C.jpg": {
        "pred": "left",
        "min_conf": 0.55,
        "ang_lo": 250,
        "ang_hi": 330,
    },
    "05_wiki_00022_Front_crashed_SLK.jpg.jpg": {
        "pred": "front_left",
        "min_conf": 0.55,
        "ang_lo": 270,
        "ang_hi": 340,
    },
    "07_wiki_00018_Damaged_car_on_road_Dubai_02..jpg": {
        "pred": "rear_left",
        "min_conf": 0.55,
        "ang_lo": 190,
        "ang_hi": 260,
    },
    "09_wiki_00147_CarWrappedAroundTreeBauregard.jpg": {
        # Extreme wreck — we only require low confidence.
        "max_conf": 0.45,
    },
}


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


def _fail(msg: str, errors: list[str]) -> None:
    print(f"  FAIL  {msg}", flush=True)
    errors.append(msg)


def check_assets(errors: list[str]) -> list[Path]:
    print("[1/4] Assets", flush=True)
    if not DEFAULT_CKPT.exists():
        _fail(f"missing checkpoint {DEFAULT_CKPT}", errors)
    else:
        _ok(f"checkpoint {DEFAULT_CKPT.name} ({DEFAULT_CKPT.stat().st_size // 10**6} MB)")
    images = list_demo_images()
    if len(images) < 8:
        _fail(f"need ≥8 demo images in {DEMO_IMAGE_DIR}, found {len(images)}", errors)
    else:
        _ok(f"{len(images)} demo images in {DEMO_IMAGE_DIR.name}")
    return images


def check_predictions(images: list[Path], errors: list[str]) -> None:
    print("[2/4] Inference on curated samples", flush=True)
    by_name = {p.name: p for p in images}
    t0 = time.time()
    # Warmup
    first = images[0]
    predict_image(Image.open(first))
    warm = time.time() - t0
    _ok(f"warmup predict {warm:.1f}s on {first.name}")

    for name, exp in EXPECTATIONS.items():
        path = by_name.get(name)
        if path is None:
            _fail(f"missing expected sample {name}", errors)
            continue
        t1 = time.time()
        result = predict_image(Image.open(path))
        dt = time.time() - t1
        pred, conf, ang = result["pred"], result["conf"], result["pred_ang"]
        detail = f"{name}: {pred} {ang:.0f}° conf={conf:.0%} ({dt:.1f}s)"
        if "pred" in exp and pred != exp["pred"]:
            _fail(f"{detail} — expected pred={exp['pred']}", errors)
            continue
        if "min_conf" in exp and conf < exp["min_conf"]:
            _fail(f"{detail} — conf below {exp['min_conf']}", errors)
            continue
        if "max_conf" in exp and conf > exp["max_conf"]:
            _fail(f"{detail} — expected low conf ≤{exp['max_conf']}", errors)
            continue
        if "ang_lo" in exp:
            # Circular-ish band for left/front_left spanning 0 wrap not needed for these
            if not (exp["ang_lo"] <= ang <= exp["ang_hi"]):
                _fail(f"{detail} — angle outside [{exp['ang_lo']},{exp['ang_hi']}]", errors)
                continue
        _ok(detail)


def check_gallery(images: list[Path], errors: list[str]) -> None:
    print("[3/4] Full gallery smoke (all demo images)", flush=True)
    n_hi = n_mid = n_lo = 0
    for path in images:
        try:
            r = predict_image(Image.open(path))
        except Exception as exc:  # noqa: BLE001
            _fail(f"{path.name} raised {exc}", errors)
            continue
        if r["conf"] >= 0.6:
            n_hi += 1
        elif r["conf"] >= 0.35:
            n_mid += 1
        else:
            n_lo += 1
    _ok(f"gallery n={len(images)}  high≥60%={n_hi}  mid={n_mid}  low={n_lo}")
    if n_hi < 4:
        _fail("expected at least 4 high-confidence exterior shots", errors)


def check_gradio(errors: list[str], base: str = "http://127.0.0.1:7860") -> None:
    print("[4/4] Gradio HTTP", flush=True)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            code = resp.status
            body = resp.read(200)
    except urllib.error.URLError as exc:
        _fail(f"Gradio not reachable at {base} ({exc})", errors)
        print("       Start with: PYTHONPATH=. .venv/bin/python -m view_classifier.app", flush=True)
        return
    if code != 200:
        _fail(f"Gradio GET / -> {code}", errors)
    else:
        _ok(f"Gradio GET / -> {code} ({base})")
    # config endpoint exists on Gradio apps
    try:
        with urllib.request.urlopen(base + "/config", timeout=5) as resp:
            cfg = json.loads(resp.read().decode())
        title = cfg.get("title") or cfg.get("space_id") or "ok"
        _ok(f"Gradio /config title={title!r}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Gradio /config failed: {exc}", errors)


def main() -> None:
    print("=== view_classifier demo self-test ===", flush=True)
    errors: list[str] = []
    images = check_assets(errors)
    if images and DEFAULT_CKPT.exists():
        check_predictions(images, errors)
        check_gallery(images, errors)
    check_gradio(errors)
    print(flush=True)
    if errors:
        print(f"RESULT: FAIL ({len(errors)} issue(s))", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        raise SystemExit(1)
    print("RESULT: PASS — ready to demo", flush=True)


if __name__ == "__main__":
    main()
