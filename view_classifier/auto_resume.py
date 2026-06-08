"""Keep restarting view_classifier.train until epochs finish and metrics.json exists.

Usage:
  PYTHONPATH=. .venv/bin/python -m view_classifier.auto_resume -- \\
    --manifest ... --resume --out-dir artifacts/joint_v2_zenodo_crop --epochs 40 ...

Pass train args after ``--``. This wrapper always injects ``--resume`` on retries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _history_epochs(out_dir: Path) -> int:
    path = out_dir / "history.json"
    if not path.exists():
        return 0
    try:
        history = json.loads(path.read_text())
    except json.JSONDecodeError:
        return 0
    return len(history) if isinstance(history, list) else 0


def _parse_out_epochs(train_argv: list[str]) -> tuple[Path, int]:
    out_dir = Path("artifacts/cfv_convnext_s")
    epochs = 30
    i = 0
    while i < len(train_argv):
        arg = train_argv[i]
        if arg == "--out-dir" and i + 1 < len(train_argv):
            out_dir = Path(train_argv[i + 1])
            i += 2
            continue
        if arg.startswith("--out-dir="):
            out_dir = Path(arg.split("=", 1)[1])
            i += 1
            continue
        if arg == "--epochs" and i + 1 < len(train_argv):
            epochs = int(train_argv[i + 1])
            i += 2
            continue
        if arg.startswith("--epochs="):
            epochs = int(arg.split("=", 1)[1])
            i += 1
            continue
        i += 1
    return out_dir, epochs


def _ensure_resume(train_argv: list[str]) -> list[str]:
    if "--resume" in train_argv:
        return list(train_argv)
    return list(train_argv) + ["--resume"]


def _done(out_dir: Path, epochs: int) -> bool:
    return _history_epochs(out_dir) >= epochs and (out_dir / "metrics.json").exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sleep",
        type=float,
        default=20.0,
        help="Seconds to wait after a crash before retrying (default 20)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Stop after N failed attempts (0 = unlimited)",
    )
    parser.add_argument(
        "train_argv",
        nargs=argparse.REMAINDER,
        help="Train args; put them after --",
    )
    args = parser.parse_args()
    train_argv = list(args.train_argv)
    if train_argv and train_argv[0] == "--":
        train_argv = train_argv[1:]
    if not train_argv:
        parser.error("pass train args after --")

    out_dir, epochs = _parse_out_epochs(train_argv)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "auto_resume.log"
    attempt = 0

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    log(f"auto-resume out={out_dir} target_epochs={epochs}")
    while True:
        done_epochs = _history_epochs(out_dir)
        if _done(out_dir, epochs):
            log(f"complete: history={done_epochs}/{epochs}, metrics.json present")
            return

        attempt += 1
        if args.max_retries and attempt > args.max_retries:
            log(f"giving up after {args.max_retries} attempts (history={done_epochs}/{epochs})")
            raise SystemExit(1)

        cmd_argv = _ensure_resume(train_argv)
        # First attempt may be a cold start if user omitted --resume and no last.pt yet.
        if attempt == 1 and not (out_dir / "view_classifier_last.pt").exists() and "--resume" not in train_argv:
            cmd_argv = list(train_argv)

        cmd = [sys.executable, "-m", "view_classifier.train", *cmd_argv]
        log(f"attempt={attempt} history={done_epochs}/{epochs} cmd={' '.join(cmd)}")
        t0 = time.time()
        # Inherit stdout/stderr so tee / pipeline.log still gets live epoch lines.
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        done_epochs = _history_epochs(out_dir)
        log(
            f"exit={proc.returncode} elapsed={elapsed:.0f}s "
            f"history={done_epochs}/{epochs} metrics={(out_dir / 'metrics.json').exists()}"
        )

        if _done(out_dir, epochs):
            log("complete after successful run")
            return

        # Train finished epochs but eval may have died; force another pass for metrics.
        if done_epochs >= epochs and not (out_dir / "metrics.json").exists():
            log("epochs done but metrics.json missing; retrying for eval")

        log(f"sleeping {args.sleep:.0f}s before resume")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
