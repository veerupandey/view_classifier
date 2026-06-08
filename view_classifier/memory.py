"""System memory helpers for capping MPS training on Apple Silicon."""

from __future__ import annotations

import gc
import os
import re
import subprocess
import sys

import torch
import torch.nn as nn

from view_classifier.model import ConvNeXtViewClassifier, angle_regression_loss


def total_ram_bytes() -> int:
    if sys.platform == "darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
    page = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page * pages)


def system_memory_fraction() -> float:
    """Approximate fraction of physical RAM in active + wired + compressed pages."""
    if sys.platform != "darwin":
        return 0.0
    page = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]).decode().strip())
    txt = subprocess.check_output(["vm_stat"]).decode()

    def pages(label: str) -> int:
        match = re.search(rf"{label}:\s+(\d+)", txt)
        return int(match.group(1)) if match else 0

    used = (
        pages("Pages active")
        + pages("Pages wired down")
        + pages("Pages occupied by compressor")
    ) * page
    return used / max(total_ram_bytes(), 1)


def _train_step_peak_fraction(batch_size: int, device: torch.device, angle_loss: str) -> float:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    model = ConvNeXtViewClassifier(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pose_crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    pose = torch.zeros(batch_size, dtype=torch.long, device=device)
    sincos_t = torch.randn(batch_size, 2, device=device)
    angle = torch.randn(batch_size, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, sincos_p = model(images)
    loss = pose_crit(logits, pose) + 2.0 * angle_regression_loss(
        sincos_p, sincos_t, angle, kind=angle_loss
    )
    loss.backward()
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()

    frac = system_memory_fraction()
    del model, optimizer, images, pose, sincos_t, angle, logits, sincos_p, loss
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return frac


def probe_batch_size(
    device: torch.device,
    *,
    mem_fraction: float,
    angle_loss: str = "mse",
    candidates: tuple[int, ...] = (4, 8, 12, 16, 20, 24, 28, 32),
    headroom: float = 0.02,
) -> int:
    """Pick the largest batch size whose train step stays at or below mem_fraction."""
    target = max(0.1, mem_fraction - headroom)
    best = candidates[0]
    for batch_size in candidates:
        frac = _train_step_peak_fraction(batch_size, device, angle_loss)
        if frac <= target:
            best = batch_size
        else:
            break
    return best


def cap_batch_size(
    batch_size: int,
    device: torch.device,
    *,
    mem_fraction: float,
    angle_loss: str = "mse",
    headroom: float = 0.02,
) -> int:
    """Reduce batch_size if a probe step would exceed mem_fraction."""
    target = max(0.1, mem_fraction - headroom)
    if batch_size <= 0:
        return probe_batch_size(
            device, mem_fraction=mem_fraction, angle_loss=angle_loss, headroom=headroom
        )
    frac = _train_step_peak_fraction(batch_size, device, angle_loss)
    if frac <= target:
        return batch_size
    for candidate in range(batch_size - 1, 0, -1):
        frac = _train_step_peak_fraction(candidate, device, angle_loss)
        if frac <= target:
            return candidate
    return 4
