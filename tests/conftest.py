"""Shared test fixtures: a synthetic hand model and fake hardware."""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import numpy as np
import pytest

from aero_webcam_teleop.tracking import HandMocap

SEGMENTS = (0.045, 0.025, 0.02)  # proximal, middle, distal phalanx lengths (m)
FINGER_BASES = {  # MCP position (x, y) in the palm plane, wrist at origin
    "index": (0.025, 0.09), "middle": (0.005, 0.095),
    "ring": (-0.015, 0.088), "pinky": (-0.033, 0.075),
}
FINGER_LANDMARKS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                    "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _bend(direction: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a unit vector in the palm plane by `deg` toward the palm (-z)."""
    a = math.radians(deg)
    return direction * math.cos(a) + np.array([0.0, 0.0, -1.0]) * math.sin(a)


def synthetic_hand(flex: Optional[Dict[str, List[float]]] = None) -> np.ndarray:
    """(21, 3) metric points for a right hand. Palm lies in z=0, fingers point +y.

    flex maps finger -> [mcp, pip, dip] flexion in degrees (bending toward -z).
    Thumb stays straight and abducted ~40 deg from the index ray.
    """
    flex = flex or {}
    pts = np.zeros((21, 3))
    for name, (x, y) in FINGER_BASES.items():
        ids = FINGER_LANDMARKS[name]
        base = np.array([x, y, 0.0])
        pts[ids[0]] = base
        # Direction of the metacarpal (wrist -> MCP) is the "straight" direction.
        direction = base / np.linalg.norm(base)
        bend = 0.0
        prev = base
        for seg, angle, lm in zip(SEGMENTS, flex.get(name, [0, 0, 0]), ids[1:]):
            bend += angle
            prev = prev + seg * _bend(direction, bend)
            pts[lm] = prev
    # Thumb: straight chain from a CMC near the wrist, ~30 deg from the index ray.
    cmc = np.array([0.02, 0.02, 0.0])
    cmc_dir = cmc / np.linalg.norm(cmc)
    pts[1] = cmc
    pts[2] = cmc + 0.04 * cmc_dir
    pts[3] = pts[2] + 0.03 * cmc_dir
    pts[4] = pts[3] + 0.025 * cmc_dir
    return pts


def mocap_from(points: np.ndarray, t: Optional[float] = None) -> HandMocap:
    return HandMocap(side="right", points=points, points_source="world",
                     timestamp=time.monotonic() if t is None else t)


class FakeHand:
    """Records SDK calls; mimics the aero_open_sdk.AeroHand surface used here."""

    def __init__(self):
        self.calls: List[tuple] = []
        self.actuations = [0.0] * 7
        self.currents = [0.0] * 7
        self.closed = False

    def set_joint_positions(self, angles):
        self.calls.append(("set_joint_positions", list(angles)))

    def set_speed(self, i, speed):
        self.calls.append(("set_speed", i, speed))

    def send_homing(self):
        self.calls.append(("send_homing",))

    def set_actuations(self, a):
        self.actuations = list(a)
        self.calls.append(("set_actuations", list(a)))

    def get_actuations(self):
        return list(self.actuations)

    def ctrl_torque(self, t):
        self.calls.append(("ctrl_torque", list(t)))

    def get_actuator_currents(self):
        return list(self.currents)

    def get_actuator_speeds(self):
        return [0.0] * 7

    def get_actuator_temperatures(self):
        return [30.0] * 7

    def close(self):
        self.closed = True

    def sent(self, name):
        return [c for c in self.calls if c[0] == name]


@pytest.fixture
def fake_hand():
    return FakeHand()
