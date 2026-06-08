from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Small_Weights, convnext_small

from view_classifier.labels import POSE_CLASSES


class ConvNeXtViewClassifier(nn.Module):
    """ConvNeXt-Small with an 8-way pose head and a sin/cos azimuth head."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_small(weights=weights)
        in_ch = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Identity()
        self.head_pose = nn.Linear(in_ch, len(POSE_CLASSES))
        self.head_angle = nn.Linear(in_ch, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        pose_logits = self.head_pose(feats)
        sincos = F.normalize(self.head_angle(feats), dim=-1)
        return pose_logits, sincos

    def freeze_backbone(self) -> None:
        for p in self.backbone.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.features.parameters():
            p.requires_grad = True


def circular_deg_error(sincos: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
    pred = torch.atan2(sincos[:, 0], sincos[:, 1]) * (180.0 / torch.pi)
    pred = pred % 360.0
    target = angle_deg % 360.0
    diff = torch.abs(pred - target)
    return torch.minimum(diff, 360.0 - diff)


def angle_regression_loss(
    sincos_p: torch.Tensor,
    sincos_t: torch.Tensor,
    angle_deg: torch.Tensor,
    kind: str = "mse",
) -> torch.Tensor:
    """MSE on unit sincos, or normalized circular degrees (Catruna-style CMAE)."""
    if kind == "circular":
        return circular_deg_error(sincos_p, angle_deg).mean() / 180.0
    if kind == "mse":
        return F.mse_loss(sincos_p, sincos_t)
    raise ValueError(f"unknown angle loss kind: {kind}")
