"""CFV / Catruna 8-way pose bins and circular-angle helpers.

Bins match Catruna et al., Electronics 2023: 0° is vehicle front, increasing
clockwise, so 90° is vehicle right (passenger side on LHD vehicles).
"""

from __future__ import annotations

import math

POSE_CLASSES: tuple[str, ...] = (
    "front",
    "front_right",
    "right",
    "rear_right",
    "rear",
    "rear_left",
    "left",
    "front_left",
)

POSE_TO_IDX = {name: i for i, name in enumerate(POSE_CLASSES)}


def wrap_deg(angle: float) -> float:
    return float(angle) % 360.0


def angle_to_pose_idx(angle: float) -> int:
    """Map [0, 360) degrees onto the 8 bins of width 45°, front centered at 0°."""
    return int((wrap_deg(angle) + 22.5) % 360.0 // 45.0)


def angle_to_sincos(angle: float) -> tuple[float, float]:
    rad = math.radians(wrap_deg(angle))
    return math.sin(rad), math.cos(rad)


def sincos_to_deg(sin_v: float, cos_v: float) -> float:
    return wrap_deg(math.degrees(math.atan2(sin_v, cos_v)))
