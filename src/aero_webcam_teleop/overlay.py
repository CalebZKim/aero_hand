"""
On-screen display: hand skeleton, status panel, joint bars, detections.

OpenCV's Hershey fonts only draw ASCII, so all text here is ASCII
(anything else renders as '???').
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .joints import HAND_CONNECTIONS, JOINT_MAX_DEG

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
GREEN = (100, 220, 100)
AMBER = (0, 190, 255)
CYAN = (255, 200, 0)

STATE_COLORS = {
    "teleop": GREEN,
    "manual": GREY,
    "detecting": AMBER,
    "planning": AMBER,
    "executing": (0, 255, 120),
    "holding": CYAN,
}

KEY_HELP = "c=calibrate h=home o=open/teleop f=fist g=grasp a=auto-grasp q=quit"

# Compact bars: thumb abduction, thumb flexion (max of 3), then each finger's MCP.
_BAR_LABELS = ("TH-ABD", "TH-FLX", "INDEX", "MIDDLE", "RING", "PINKY")


def compact_angles(angles: Sequence[float]):
    """(value, max) pairs for the six display bars."""
    return [
        (angles[0], JOINT_MAX_DEG[0]),
        (max(angles[1], angles[2], angles[3]), 90.0),
        (angles[4], 90.0), (angles[7], 90.0), (angles[10], 90.0), (angles[13], 90.0),
    ]


def draw_hand(img: np.ndarray, keypoints, side: str) -> None:
    h, w = img.shape[:2]
    pts = [(int(k.x * w), int(k.y * h)) for k in keypoints]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], (230, 230, 230), 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, p, 4 if i % 4 == 0 else 3, (60, 60, 230), -1, cv2.LINE_AA)
    cv2.putText(img, f"{side.upper()} hand", (pts[9][0] - 40, pts[9][1] - 20),
                FONT, 0.55, GREEN, 2, cv2.LINE_AA)


def draw_detection(img: np.ndarray, det) -> None:
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 150), 2)
    extra = f" {det.size_cm:.1f}cm" if det.size_cm else ""
    extra += f" @{det.depth_m:.2f}m" if det.depth_m else ""
    cv2.putText(img, f"{det.label} {det.confidence:.0%}{extra}", (x1, max(y1 - 8, 14)),
                FONT, 0.45, (0, 255, 150), 1, cv2.LINE_AA)


def draw_status(
    img: np.ndarray,
    *,
    side: str,
    hardware: str,
    fps: float,
    camera: str,
    state: str,
    auto_grasp: bool,
    llm: bool,
    calibrated: bool,
    homing: bool,
    angles: Optional[Sequence[float]],
    plan=None,
) -> None:
    h, w = img.shape[:2]
    panel_h = 150
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, dst=img)

    cv2.putText(img, "Aero Hand | webcam teleop + grasp", (10, 22), FONT, 0.6, CYAN, 2, cv2.LINE_AA)
    cv2.putText(img, KEY_HELP, (10, 42), FONT, 0.36, GREY, 1, cv2.LINE_AA)

    status = (f"[{side.upper()}] {hardware}  FPS {fps:4.1f}  CAM {camera}  "
              f"{'AUTO-GRASP' if auto_grasp else 'manual grasp'}  "
              f"{'LLM' if llm else 'fallback table'}  "
              f"{'calibrated' if calibrated else 'NOT CALIBRATED (press c)'}")
    cv2.putText(img, status, (10, 60), FONT, 0.36, GREEN, 1, cv2.LINE_AA)

    label = "HOMING..." if homing else state.upper()
    cv2.putText(img, label, (10, 82), FONT, 0.5, STATE_COLORS.get(state, GREY), 1, cv2.LINE_AA)
    if plan is not None:
        tag = "LLM" if plan.source == "llm" else "fallback"
        cv2.putText(img, f"{plan.object_label}: {plan.grasp_type} / {plan.force_level} [{tag}]",
                    (140, 82), FONT, 0.42, (255, 180, 50), 1, cv2.LINE_AA)
        reason = plan.reasoning if len(plan.reasoning) <= 90 else plan.reasoning[:87] + "..."
        cv2.putText(img, reason.encode("ascii", "replace").decode(), (10, 100),
                    FONT, 0.34, (210, 210, 210), 1, cv2.LINE_AA)

    if angles is not None:
        bar_w = (w - 20) // len(_BAR_LABELS)
        y = panel_h - 30
        for i, ((val, mx), name) in enumerate(zip(compact_angles(angles), _BAR_LABELS)):
            x = 10 + i * bar_w
            fill = int(max(0.0, min(1.0, val / mx)) * (bar_w - 6))
            cv2.rectangle(img, (x, y), (x + bar_w - 6, y + 18), (60, 60, 60), -1)
            cv2.rectangle(img, (x, y), (x + fill, y + 18), (0, 200, 100), -1)
            cv2.putText(img, name, (x + 3, y + 13), FONT, 0.32, WHITE, 1, cv2.LINE_AA)
