"""
HandMocap (21 landmarks) -> 16 Aero Hand joint angles in degrees.

The per-joint scale factors are tuned on hardware.

* Works on ``HandMocap.points`` (consistent 3-D units).
* Calibration is computed from *raw* angles, so it carries no EMA lag and
  recalibrating never compounds.
* Calibration rescales instead of only subtracting an offset, so the full
  0..max range is still reachable after calibrating.
* No MediaPipe import, so this module is testable without a camera.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .joints import JOINT_MAX_DEG, NUM_JOINTS, Landmark as L, clamp_joints

# Per-joint output scale: maps MediaPipe's under-ranged geometry to the robot's
# full range. Tuned on hardware so a closed human fist reads ~90 deg.
#   thumb: abd 1.8, cmc_flex 2.5, mcp 2.0, ip 2.2 (short, foreshortened segments)
#   fingers: mcp 1.5, pip 1.7, dip 1.8
DEFAULT_JOINT_SCALE: List[float] = [
    1.8, 2.5, 2.0, 2.2,
    1.5, 1.7, 1.8,
    1.5, 1.7, 1.8,
    1.5, 1.7, 1.8,
    1.5, 1.7, 1.8,
]

# (proximal, joint, distal) landmark triples for every flexion joint (1..15).
_FLEXION_TRIPLES = {
    1: (L.WRIST, L.THUMB_CMC, L.THUMB_MCP),
    2: (L.THUMB_CMC, L.THUMB_MCP, L.THUMB_IP),
    3: (L.THUMB_MCP, L.THUMB_IP, L.THUMB_TIP),
    4: (L.WRIST, L.INDEX_MCP, L.INDEX_PIP),
    5: (L.INDEX_MCP, L.INDEX_PIP, L.INDEX_DIP),
    6: (L.INDEX_PIP, L.INDEX_DIP, L.INDEX_TIP),
    7: (L.WRIST, L.MIDDLE_MCP, L.MIDDLE_PIP),
    8: (L.MIDDLE_MCP, L.MIDDLE_PIP, L.MIDDLE_DIP),
    9: (L.MIDDLE_PIP, L.MIDDLE_DIP, L.MIDDLE_TIP),
    10: (L.WRIST, L.RING_MCP, L.RING_PIP),
    11: (L.RING_MCP, L.RING_PIP, L.RING_DIP),
    12: (L.RING_PIP, L.RING_DIP, L.RING_TIP),
    13: (L.WRIST, L.PINKY_MCP, L.PINKY_PIP),
    14: (L.PINKY_MCP, L.PINKY_PIP, L.PINKY_DIP),
    15: (L.PINKY_PIP, L.PINKY_DIP, L.PINKY_TIP),
}


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Unsigned angle between two vectors, in radians (0 for degenerate input)."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    return float(math.acos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def palm_normal(points: np.ndarray) -> np.ndarray:
    """Unit normal of the plane through wrist, index MCP and pinky MCP."""
    w = points[L.WRIST]
    n = np.cross(points[L.INDEX_MCP] - w, points[L.PINKY_MCP] - w)
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])


def _to_joint_range(rad: float, joint: int, scale: float) -> float:
    """Map [0, pi] -> [0, joint_max], apply scale, clamp."""
    deg = (rad / math.pi) * JOINT_MAX_DEG[joint] * scale
    return float(np.clip(deg, 0.0, JOINT_MAX_DEG[joint]))


def raw_joint_angles(points: np.ndarray, joint_scale: Sequence[float] = DEFAULT_JOINT_SCALE) -> List[float]:
    """Uncalibrated, unsmoothed 16 joint angles (deg) from a (21, 3) point array."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (21, 3):
        raise ValueError(f"Expected (21, 3) points, got {p.shape}")
    angles = [0.0] * NUM_JOINTS

    # Thumb abduction: angle between thumb and index rays, projected on the palm plane.
    n = palm_normal(p)
    thumb = p[L.THUMB_CMC] - p[L.WRIST]
    index = p[L.INDEX_MCP] - p[L.WRIST]
    thumb -= np.dot(thumb, n) * n
    index -= np.dot(index, n) * n
    angles[0] = _to_joint_range(angle_between(thumb, index), 0, joint_scale[0])

    # Flexion joints: angle between consecutive bone vectors.
    for j, (a, b, c) in _FLEXION_TRIPLES.items():
        angles[j] = _to_joint_range(angle_between(p[b] - p[a], p[c] - p[b]), j, joint_scale[j])
    return angles


@dataclass
class RetargetingConfig:
    """Per-user calibration.

    ``open_reference`` holds the raw angles read with an open, flat palm.
    Calibrated angle = (raw - open) / (max - open) * max.
    """

    open_reference: List[float] = field(default_factory=lambda: [0.0] * NUM_JOINTS)
    joint_scale: List[float] = field(default_factory=lambda: list(DEFAULT_JOINT_SCALE))

    @property
    def is_calibrated(self) -> bool:
        return any(v > 0 for v in self.open_reference)


class Retargeter:
    """HandMocap -> 16 smoothed, calibrated joint angles (degrees).

    Parameters
    ----------
    config          : calibration and scale factors
    smoothing_alpha : EMA weight on the newest sample in [0, 1];
                      lower = smoother but laggier (0.35 is a good start)
    """

    def __init__(self, config: Optional[RetargetingConfig] = None, smoothing_alpha: float = 0.35):
        self.config = config or RetargetingConfig()
        self.alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self._prev: Optional[List[float]] = None

    def apply_calibration(self, raw: Sequence[float]) -> List[float]:
        out = []
        for j, (r, ref) in enumerate(zip(raw, self.config.open_reference)):
            span = JOINT_MAX_DEG[j] - ref
            out.append((r - ref) / span * JOINT_MAX_DEG[j] if span > 1e-6 else 0.0)
        return clamp_joints(out)

    def retarget(self, mocap) -> List[float]:
        raw = raw_joint_angles(mocap.points, self.config.joint_scale)
        angles = self.apply_calibration(raw)
        if self._prev is not None:
            a = self.alpha
            angles = [a * x + (1.0 - a) * p for x, p in zip(angles, self._prev)]
        self._prev = list(angles)
        return angles

    def calibrate(self, open_palm_mocap) -> List[float]:
        """Record an open, flat palm as the zero reference."""
        self.config.open_reference = raw_joint_angles(open_palm_mocap.points, self.config.joint_scale)
        self.reset()
        print("[retargeting] Calibrated. Open-palm reference:",
              [round(v, 1) for v in self.config.open_reference])
        print("[retargeting] Now make a fist and check the bars reach the top.")
        return list(self.config.open_reference)

    def reset(self) -> None:
        """Forget smoothing history (call when the hand is lost)."""
        self._prev = None
