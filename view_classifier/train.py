"""Train ConvNeXt-Small on CFV, optionally joint with UnsupCar."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from view_classifier.data import CfvViewDataset, load_manifest
from view_classifier.labels import POSE_CLASSES
from view_classifier.memory import cap_batch_size, system_memory_fraction
from view_classifier.model import ConvNeXtViewClassifier, angle_regression_loss, circular_deg_error


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but torch.backends.mps.is_available() is False")
        torch.set_float32_matmul_precision("high")
        return torch.device("mps")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "pose_idx": torch.tensor([b["pose_idx"] for b in batch], dtype=torch.long),
        "angle": torch.tensor([b["angle"] for b in batch], dtype=torch.float32),
        "sincos": torch.tensor([b["sincos"] for b in batch], dtype=torch.float32),
        "identity": torch.tensor([b["identity"] for b in batch], dtype=torch.long),
    }


def run_epoch(
    model,
    loader,
    device,
    pose_crit,
    angle_w,
    optimizer=None,
    max_batches=None,
    accum=1,
    angle_loss: str = "mse",
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_ce = total_ang = 0.0
    correct = seen = 0
    cmae_sum = 0.0
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for bi, batch in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=False)
            pose = batch["pose_idx"].to(device)
            sincos_t = batch["sincos"].to(device)
            angle = batch["angle"].to(device)
            logits, sincos_p = model(images)
            loss_ce = pose_crit(logits, pose)
            loss_ang = angle_regression_loss(sincos_p, sincos_t, angle, kind=angle_loss)
            loss = (loss_ce + angle_w * loss_ang) / accum
            if training:
                loss.backward()
                if (bi + 1) % accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            bs = images.size(0)
            total_loss += loss.item() * accum * bs
            total_ce += loss_ce.item() * bs
            total_ang += loss_ang.item() * bs
            correct += (logits.argmax(1) == pose).sum().item()
            cmae_sum += circular_deg_error(sincos_p.detach(), angle).sum().item()
            seen += bs
        if training and (bi + 1) % accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    n = max(seen, 1)
    return {
        "loss": total_loss / n,
        "ce": total_ce / n,
        "ang": total_ang / n,
        "acc": correct / n,
        "cmae": cmae_sum / n,
    }


@torch.no_grad()
def confusion_and_cmae(model, loader, device):
    model.eval()
    n = len(POSE_CLASSES)
    confusion = [[0] * n for _ in range(n)]
    correct = seen = 0
    cmae_sum = 0.0
    adj = 0
    for batch in loader:
        logits, sincos_p = model(batch["image"].to(device))
        pred = logits.argmax(1).cpu()
        pose = batch["pose_idx"]
        cmae_sum += circular_deg_error(sincos_p.cpu(), batch["angle"]).sum().item()
        for t, p in zip(pose.tolist(), pred.tolist()):
            confusion[t][p] += 1
            correct += int(t == p)
            adj += int(min(abs(t - p), 8 - abs(t - p)) <= 1)
            seen += 1
    return correct / max(seen, 1), cmae_sum / max(seen, 1), adj / max(seen, 1), confusion


def split_records(manifest: dict, source: str) -> dict[str, list]:
    by = {"train": [], "val": [], "test": []}
    for rec in manifest["records"]:
        rec.setdefault("source", source)
        by[rec["split"]].append(rec)
    return by


def eval_split(model, records, image_root, device, batch_size, smoke: bool):
    if not records:
        return None
    ds = CfvViewDataset(records, train=False, image_root=image_root)
    if smoke:
        ds = torch.utils.data.Subset(ds, range(min(32, len(ds))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    acc, cmae, adj, confusion = confusion_and_cmae(model, loader, device)
    return {
        "n": len(ds),
        "accuracy": round(acc, 4),
        "cmae_deg": round(cmae, 3),
        "adjacent_or_correct": round(adj, 4),
        "confusion_matrix": confusion,
    }


def plot_confusion(confusion, out_path: Path):
    n = len(POSE_CLASSES)
    row_sums = [max(sum(row), 1) for row in confusion]
    norm = [[c / s for c in row] for row, s in zip(confusion, row_sums)]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), POSE_CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(n), POSE_CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(confusion[i][j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


LAST_CKPT = "view_classifier_last.pt"
BEST_CKPT = "view_classifier_best.pt"


def _args_dict(args: argparse.Namespace) -> dict:
    return {
        k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
        for k, v in vars(args).items()
    }


def save_last_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    best_epoch: int,
    history: list,
    unfrozen: bool,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "history": history,
        "unfrozen": unfrozen,
        "classes": list(POSE_CLASSES),
        "args": _args_dict(args),
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_training_state(
    path: Path,
    *,
    model,
    device: torch.device,
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def maybe_save_best(
    path: Path,
    *,
    model,
    epoch: int,
    val_m: dict,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_m["acc"],
            "val_cmae": val_m["cmae"],
            "classes": list(POSE_CLASSES),
            "args": _args_dict(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/cfv/manifest.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/cfv/images"))
    parser.add_argument("--extra-manifest", type=Path, default=None)
    parser.add_argument("--extra-image-root", type=Path, default=Path("data/unsupcar/images"))
    parser.add_argument("--init", type=Path, default=None, help="Warm-start checkpoint (joint finetune)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume from {LAST_CKPT} in --out-dir (falls back to {BEST_CKPT})",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/cfv_convnext_s"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="auto picks MPS on Apple Silicon (default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Per-step batch; 0 = auto from --mem-fraction on MPS (default)",
    )
    parser.add_argument("--accum", type=int, default=2, help="Grad accum; effective batch = batch-size × accum")
    parser.add_argument(
        "--mem-fraction",
        type=float,
        default=0.55,
        help="On MPS/macOS: cap train step so system RAM stays at or below this fraction (default 0.55)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--angle-w", type=float, default=2.0)
    parser.add_argument(
        "--angle-loss",
        choices=("mse", "circular"),
        default="mse",
        help="mse=sin/cos MSE; circular=normalized CMAE (paper-style)",
    )
    parser.add_argument(
        "--no-weak-crop",
        action="store_true",
        help="Disable RandomResizedCrop train aug (legacy color-jitter-only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.freeze_epochs = 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    last_path = args.out_dir / LAST_CKPT
    best_path = args.out_dir / BEST_CKPT

    resume_batch: int | None = None
    if args.resume:
        for ckpt_path in (last_path, best_path):
            if ckpt_path.exists():
                meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                saved = meta.get("args", {}).get("batch_size")
                if saved is not None:
                    resume_batch = int(saved)
                break

    if args.batch_size > 0:
        if device.type == "mps" and sys.platform == "darwin":
            requested = args.batch_size
            args.batch_size = cap_batch_size(
                args.batch_size,
                device,
                mem_fraction=args.mem_fraction,
                angle_loss=args.angle_loss,
            )
            if args.batch_size != requested:
                print(
                    f"[mem] reduced batch {requested}->{args.batch_size} "
                    f"(cap {args.mem_fraction:.0%})",
                    flush=True,
                )
            elif resume_batch is not None and requested != resume_batch:
                print(
                    f"[mem] overriding checkpoint batch {resume_batch}->{args.batch_size} "
                    f"(cap {args.mem_fraction:.0%})",
                    flush=True,
                )
        else:
            print(f"[mem] batch={args.batch_size} (explicit)", flush=True)
    elif resume_batch is not None:
        args.batch_size = resume_batch
        if device.type == "mps" and sys.platform == "darwin":
            before = system_memory_fraction()
            capped = cap_batch_size(
                args.batch_size,
                device,
                mem_fraction=args.mem_fraction,
                angle_loss=args.angle_loss,
            )
            if capped != args.batch_size:
                print(
                    f"[mem] resume batch {args.batch_size}->{capped} "
                    f"(cap {args.mem_fraction:.0%}, system {before:.0%})",
                    flush=True,
                )
                args.batch_size = capped
            else:
                print(f"[resume] batch={args.batch_size} from checkpoint", flush=True)
        else:
            print(f"[resume] batch={args.batch_size} from checkpoint", flush=True)
    elif device.type == "mps" and sys.platform == "darwin":
        before = system_memory_fraction()
        args.batch_size = cap_batch_size(
            0,
            device,
            mem_fraction=args.mem_fraction,
            angle_loss=args.angle_loss,
        )
        print(
            f"[mem] auto batch={args.batch_size} (cap {args.mem_fraction:.0%}, "
            f"system {before:.0%}->{system_memory_fraction():.0%})",
            flush=True,
        )
    else:
        args.batch_size = 8

    eff = args.batch_size * args.accum
    print(
        f"[device] {device} batch={args.batch_size} accum={args.accum} effective={eff}",
        flush=True,
    )

    manifest = load_manifest(args.manifest)
    by_split = split_records(manifest, "cfv")
    extra = None
    extra_by = None
    if args.extra_manifest is not None:
        extra = load_manifest(args.extra_manifest)
        extra_by = split_records(extra, extra.get("source", "unsupcar"))

    weak_crop = not args.no_weak_crop
    train_parts = [
        CfvViewDataset(by_split["train"], train=True, image_root=args.image_root, weak_crop=weak_crop)
    ]
    if extra_by is not None:
        train_parts.append(
            CfvViewDataset(
                extra_by["train"],
                train=True,
                image_root=args.extra_image_root,
                weak_crop=weak_crop,
            )
        )
    train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_ds = CfvViewDataset(by_split["val"], train=False, image_root=args.image_root)
    test_ds = CfvViewDataset(by_split["test"], train=False, image_root=args.image_root)
    extra_val_ds = extra_val_loader = None
    if extra_by is not None:
        extra_val_ds = CfvViewDataset(extra_by["val"], train=False, image_root=args.extra_image_root)
    print(
        f"[data] train={len(train_ds)} cfv_val={len(val_ds)} cfv_test={len(test_ds)} "
        f"cfv_ids={manifest['n_identities']}"
        + (
            f" extra_val={len(extra_val_ds)} extra_ids={extra['n_identities']}"
            if extra_val_ds is not None
            else ""
        ),
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate
    )
    if extra_val_ds is not None:
        extra_val_loader = DataLoader(
            extra_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )

    model = ConvNeXtViewClassifier(pretrained=args.init is None and not args.resume).to(device)
    pose_crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    history: list = []
    history_path = args.out_dir / "history.json"
    best_acc = -1.0
    best_epoch = 0
    start_epoch = 1
    unfrozen = args.freeze_epochs <= 0

    if args.resume:
        resume_path = last_path if last_path.exists() else best_path if best_path.exists() else None
        if resume_path is None:
            raise SystemExit(f"--resume requested but no {LAST_CKPT} or {BEST_CKPT} in {args.out_dir}")
        ckpt = load_training_state(resume_path, model=model, device=device)
        if resume_path == last_path:
            history = ckpt.get("history", [])
            best_acc = float(ckpt.get("best_acc", -1.0))
            best_epoch = int(ckpt.get("best_epoch", 0))
            unfrozen = bool(ckpt.get("unfrozen", unfrozen))
            start_epoch = int(ckpt["epoch"]) + 1
            print(f"[resume] {resume_path} epoch={ckpt['epoch']} -> {start_epoch}", flush=True)
        else:
            if history_path.exists():
                history = json.loads(history_path.read_text())
            start_epoch = len(history) + 1
            best_acc = max((r["val"]["acc"] for r in history), default=-1.0)
            best_epoch = max(
                (r["epoch"] for r in history if r["val"]["acc"] == best_acc),
                default=0,
            )
            print(
                f"[resume] {resume_path} (no {LAST_CKPT}); history={len(history)} -> epoch {start_epoch}",
                flush=True,
            )
        if start_epoch > args.epochs:
            print(f"[resume] already finished ({start_epoch - 1}/{args.epochs})", flush=True)
    elif args.init is not None:
        loaded = torch.load(args.init, map_location=device, weights_only=False)
        model.load_state_dict(loaded["model_state"])
        print(f"[init] {args.init} epoch={loaded.get('epoch')}", flush=True)

    if args.freeze_epochs > 0 and not unfrozen:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    remaining = max(1, args.epochs - start_epoch + 1)
    if unfrozen and args.freeze_epochs > 0:
        remaining = max(1, args.epochs - max(args.freeze_epochs, start_epoch - 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        if ckpt.get("optimizer_state"):
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            except ValueError:
                print("[resume] scheduler state mismatch; using fresh schedule", flush=True)

    max_batches = 3 if args.smoke else None

    if start_epoch <= args.epochs:
        for epoch in range(start_epoch, args.epochs + 1):
            if not unfrozen and epoch == args.freeze_epochs + 1:
                model.unfreeze_backbone()
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr * 0.3, weight_decay=args.weight_decay
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - args.freeze_epochs)
                )
                unfrozen = True
                print(f"[epoch {epoch}] unfroze backbone", flush=True)
            t0 = time.time()
            train_m = run_epoch(
                model,
                train_loader,
                device,
                pose_crit,
                args.angle_w,
                optimizer,
                max_batches,
                args.accum,
                args.angle_loss,
            )
            val_m = run_epoch(
                model, val_loader, device, pose_crit, args.angle_w, None, max_batches, 1, args.angle_loss
            )
            extra_m = None
            if extra_val_loader is not None:
                extra_m = run_epoch(
                    model,
                    extra_val_loader,
                    device,
                    pose_crit,
                    args.angle_w,
                    None,
                    max_batches,
                    1,
                    args.angle_loss,
                )
            scheduler.step()
            row = {
                "epoch": epoch,
                "train": train_m,
                "val": val_m,
                "seconds": round(time.time() - t0, 1),
                "backbone": "frozen" if epoch <= args.freeze_epochs else "unfrozen",
            }
            if extra_m is not None:
                row["val_unsupcar"] = extra_m
            history.append(row)
            marker = ""
            if val_m["acc"] > best_acc:
                best_acc = val_m["acc"]
                best_epoch = epoch
                maybe_save_best(best_path, model=model, epoch=epoch, val_m=val_m, args=args)
                marker = "  <- best"
            save_last_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                best_epoch=best_epoch,
                history=history,
                unfrozen=unfrozen,
                args=args,
            )
            extra_txt = ""
            if extra_m is not None:
                extra_txt = f" unsup_val={extra_m['acc']:.3f}/{extra_m['cmae']:.1f}°"
            print(
                f"[epoch {epoch:02d}/{args.epochs}] "
                f"train_acc={train_m['acc']:.3f} cfv_val={val_m['acc']:.3f} "
                f"cfv_cmae={val_m['cmae']:.2f}{extra_txt} ({row['seconds']}s){marker}",
                flush=True,
            )
            history_path.write_text(json.dumps(history, indent=2))
            if device.type == "mps":
                torch.mps.empty_cache()

    eval_ckpt_path = best_path if best_path.exists() else last_path
    ckpt = torch.load(eval_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] using {eval_ckpt_path.name} epoch={ckpt.get('epoch', best_epoch)}", flush=True)
    by_source = {
        "cfv": eval_split(
            model,
            by_split["val"] if args.smoke else by_split["test"],
            args.image_root,
            device,
            args.batch_size,
            args.smoke,
        )
    }
    if extra_by is not None:
        by_source["unsupcar"] = eval_split(
            model,
            extra_by["val"] if args.smoke else extra_by["test"],
            args.extra_image_root,
            device,
            args.batch_size,
            args.smoke,
        )
    cfv = by_source["cfv"]
    plot_confusion(cfv["confusion_matrix"], args.out_dir / "confusion_matrix.png")
    if by_source.get("unsupcar"):
        plot_confusion(
            by_source["unsupcar"]["confusion_matrix"],
            args.out_dir / "confusion_matrix_unsupcar.png",
        )
    metrics = {
        "device": str(device),
        "best_epoch": ckpt.get("epoch", best_epoch),
        "eval_split": "val" if args.smoke else "test",
        "accuracy": cfv["accuracy"],
        "cmae_deg": cfv["cmae_deg"],
        "adjacent_or_correct": cfv["adjacent_or_correct"],
        "classes": list(POSE_CLASSES),
        "confusion_matrix": cfv["confusion_matrix"],
        "by_source": by_source,
        "history": history,
        "split_summary": {k: v for k, v in manifest.items() if k != "records"},
    }
    if extra is not None:
        metrics["extra_split_summary"] = {k: v for k, v in extra.items() if k != "records"}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[eval] cfv acc={cfv['accuracy']:.4f} cmae={cfv['cmae_deg']:.2f}° "
        f"adjacent_or_correct={cfv['adjacent_or_correct']:.3f}",
        flush=True,
    )
    if by_source.get("unsupcar"):
        u = by_source["unsupcar"]
        print(
            f"[eval] unsupcar acc={u['accuracy']:.4f} cmae={u['cmae_deg']:.2f}° "
            f"adjacent_or_correct={u['adjacent_or_correct']:.3f}",
            flush=True,
        )
    print(f"[eval] -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
