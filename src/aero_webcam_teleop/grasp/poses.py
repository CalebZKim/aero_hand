"""
Grasp pose library: grasp type -> 16 joint angles, force level -> speed.

Note on "force": the Aero Hand is position-controlled here, so a force level
is applied as a scale on actuator speed (slower closing = gentler contact).
It is not a measured or controlled grip force.

This module is also the single source of per-object fallback plans.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

from ..joints import NUM_JOINTS


class GraspPose(NamedTuple):
    name: str
    angles: Tuple[float, ...]  # 16 joint angles (deg), SDK joint order
    description: str


class ForceProfile(NamedTuple):
    level: str
    speed_scalar: float  # multiplier on the bridge's normal speed
    description: str


FORCE_PROFILES: Dict[str, ForceProfile] = {
    "light": ForceProfile("light", 0.3, "Slow close: fragile or compliant objects"),
    "medium": ForceProfile("medium", 0.6, "Moderate close: everyday rigid objects"),
    "firm": ForceProfile("firm", 1.0, "Full-speed close: heavy or slippery objects"),
}

#                      thumb abd/flex/mcp/ip     index          middle         ring           pinky
GRASP_POSES: Dict[str, GraspPose] = {
    "power": GraspPose("Power Grasp", (
        60.0, 35.0, 55.0, 40.0, 75.0, 70.0, 60.0, 75.0, 70.0, 60.0, 70.0, 65.0, 55.0, 65.0, 60.0, 50.0,
    ), "Full-hand wrap: cylinders, large objects"),
    "pinch": GraspPose("Pinch Grasp", (
        40.0, 30.0, 50.0, 45.0, 55.0, 50.0, 40.0, 15.0, 10.0, 5.0, 10.0, 5.0, 0.0, 5.0, 0.0, 0.0,
    ), "Index-thumb tip pinch: small, precise objects"),
    "tripod": GraspPose("Tripod Grasp", (
        45.0, 32.0, 52.0, 45.0, 55.0, 50.0, 40.0, 55.0, 50.0, 40.0, 15.0, 10.0, 5.0, 5.0, 0.0, 0.0,
    ), "Thumb + index + middle: cubes, medium objects"),
    "lateral_pinch": GraspPose("Lateral Pinch", (
        20.0, 25.0, 40.0, 35.0, 45.0, 40.0, 30.0, 20.0, 15.0, 10.0, 10.0, 5.0, 0.0, 5.0, 0.0, 0.0,
    ), "Thumb against side of index: flat or thin objects"),
    "palmar": GraspPose("Palmar Grasp", (
        55.0, 30.0, 45.0, 35.0, 60.0, 55.0, 45.0, 65.0, 60.0, 50.0, 60.0, 55.0, 45.0, 55.0, 50.0, 40.0,
    ), "Palm cradle: soft objects, large spheres"),
    "open": GraspPose("Open Hand", (0.0,) * NUM_JOINTS, "Fully open: approach or release"),
}

# object label -> (grasp type, force level, reason)
OBJECT_DEFAULTS: Dict[str, Tuple[str, str, str]] = {
    "sphere": ("power", "medium", "Spherical: power wrap"),
    "wooden_sphere": ("power", "firm", "Rigid sphere: firm power wrap"),
    "cylinder": ("power", "medium", "Cylindrical: power wrap"),
    "cube": ("tripod", "medium", "Cubic: tripod on flat faces"),
    "rectangular_prism": ("lateral_pinch", "medium", "Flat object: lateral pinch"),
    "squish_ball": ("palmar", "light", "Soft ball: gentle palmar cradle"),
    "pyramid": ("tripod", "medium", "Tripod on pyramid faces"),
    "small_object": ("pinch", "light", "Generic small object"),
    "large_object": ("power", "firm", "Generic large object"),
    "flat_object": ("lateral_pinch", "medium", "Generic flat object"),
    "unknown": ("tripod", "medium", "Unknown object: safe tripod default"),
}

GRASP_TYPES = frozenset(GRASP_POSES)
FORCE_LEVELS = frozenset(FORCE_PROFILES)


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def get_pose(grasp_type: str) -> GraspPose:
    return GRASP_POSES.get(grasp_type, GRASP_POSES["tripod"])


def get_force(force_level: str) -> ForceProfile:
    return FORCE_PROFILES.get(force_level, FORCE_PROFILES["medium"])


def default_for_object(label: str) -> Tuple[str, str, str]:
    return OBJECT_DEFAULTS.get(normalize_label(label), OBJECT_DEFAULTS["unknown"])


def interpolate(start: Sequence[float], end: Sequence[float], t: float) -> List[float]:
    """Linear blend between two poses, t clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return [s + (e - s) * t for s, e in zip(start, end)]
