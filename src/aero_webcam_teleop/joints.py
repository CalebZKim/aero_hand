"""
Shared constants: the Aero Hand joint layout and the MediaPipe hand landmark
layout. Every other module imports from here so the two definitions can never
drift apart.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Aero Hand Open joints (SDK order, degrees, 0 = fully extended)
# ---------------------------------------------------------------------------

JOINT_NAMES: Tuple[str, ...] = (
    "thumb_cmc_abd",   # 0
    "thumb_cmc_flex",  # 1
    "thumb_mcp",       # 2
    "thumb_ip",        # 3
    "index_mcp",       # 4
    "index_pip",       # 5
    "index_dip",       # 6
    "middle_mcp",      # 7
    "middle_pip",      # 8
    "middle_dip",      # 9
    "ring_mcp",        # 10
    "ring_pip",        # 11
    "ring_dip",        # 12
    "pinky_mcp",       # 13
    "pinky_pip",       # 14
    "pinky_dip",       # 15
)
NUM_JOINTS = len(JOINT_NAMES)

JOINT_MIN_DEG: Tuple[float, ...] = (0.0,) * NUM_JOINTS
JOINT_MAX_DEG: Tuple[float, ...] = (
    100.0, 55.0, 90.0, 90.0,  # thumb
    90.0, 90.0, 90.0,         # index
    90.0, 90.0, 90.0,         # middle
    90.0, 90.0, 90.0,         # ring
    90.0, 90.0, 90.0,         # pinky
)

NUM_ACTUATORS = 7
ACTUATOR_NAMES: Tuple[str, ...] = (
    "thumb_cmc_abd",
    "thumb_cmc_flex",
    "thumb_tendon",
    "index_tendon",
    "middle_tendon",
    "ring_tendon",
    "pinky_tendon",
)

# Actuator speed range accepted by the SDK.
SPEED_MAX = 32766

OPEN_POSE: Tuple[float, ...] = (0.0,) * NUM_JOINTS
FIST_POSE: Tuple[float, ...] = (
    50.0, 27.5, 45.0, 45.0,   # thumb tucked, not fully opposed
    90.0, 90.0, 90.0,
    90.0, 90.0, 90.0,
    90.0, 90.0, 90.0,
    90.0, 90.0, 90.0,
)


def clamp_joints(angles: Sequence[float]) -> List[float]:
    """Clamp 16 joint angles to the SDK limits."""
    if len(angles) != NUM_JOINTS:
        raise ValueError(f"Expected {NUM_JOINTS} joint angles, got {len(angles)}")
    return [
        float(min(max(a, lo), hi))
        for a, lo, hi in zip(angles, JOINT_MIN_DEG, JOINT_MAX_DEG)
    ]


# ---------------------------------------------------------------------------
# MediaPipe hand landmarks (21 points)
# ---------------------------------------------------------------------------

class Landmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


NUM_LANDMARKS = 21

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
)
