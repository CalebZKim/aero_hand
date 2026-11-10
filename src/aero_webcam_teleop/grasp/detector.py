"""
Object detection for grasp mode (YOLOv8 via ``ultralytics``, optional).

YOLO's 80 COCO classes are mapped onto the demo object set. With a D405,
the object's distance and approximate size come from the depth image.

Frames are passed to YOLO as BGR (ultralytics treats numpy input as BGR)
and the camera field of view is a parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

# COCO label -> demo object label. Extend alongside OBJECT_DEFAULTS in poses.py.
YOLO_TO_OBJECT: Dict[str, str] = {
    "sports ball": "wooden_sphere", "orange": "wooden_sphere", "apple": "wooden_sphere",
    "cup": "cylinder", "bottle": "cylinder", "vase": "cylinder",
    "bowl": "cylinder", "wine glass": "cylinder",
    "book": "rectangular_prism", "laptop": "rectangular_prism",
    "cell phone": "rectangular_prism", "remote": "rectangular_prism",
    "suitcase": "rectangular_prism",
}
# Labels a custom-trained model may emit directly.
DIRECT_LABELS = frozenset({
    "sphere", "wooden_sphere", "cylinder", "cube", "rectangular_prism", "squish_ball", "pyramid",
})

D405_COLOR_HFOV_DEG = 87.0


def map_label(yolo_label: str) -> str:
    label = yolo_label.lower()
    if label.replace(" ", "_") in DIRECT_LABELS:
        return label.replace(" ", "_")
    return YOLO_TO_OBJECT.get(label, "unknown")


def estimate_size_cm(size_px: float, depth_m: float, image_width_px: int, hfov_deg: float) -> float:
    """Pinhole estimate of an object's extent from its pixel size and distance."""
    focal_px = image_width_px / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return size_px * depth_m / focal_px * 100.0


@dataclass
class Detection:
    label: str                       # demo object label
    yolo_label: str                  # raw model class name
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    size_px: float
    size_cm: Optional[float] = None
    depth_m: Optional[float] = None


class GraspDetector:
    """
    Parameters
    ----------
    model_size : YOLOv8 variant, "n" (fastest) .. "x"
    confidence : minimum detection confidence
    device     : "cpu" | "cuda" | "mps" | None (auto)
    hfov_deg   : horizontal FOV of the colour camera, for size estimates
    """

    def __init__(
        self,
        model_size: str = "n",
        confidence: float = 0.25,
        device: Optional[str] = None,
        hfov_deg: float = D405_COLOR_HFOV_DEG,
    ):
        self.confidence = confidence
        self.hfov_deg = hfov_deg
        self._model = None
        if not YOLO_AVAILABLE:
            print("[detector] ultralytics not installed; grasp mode will fall back to 'unknown'. "
                  "Install with: pip install 'aero-webcam-teleop[grasp]'")
            return
        self._model = YOLO(f"yolov8{model_size}.pt")
        if device:
            self._model.to(device)

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray, depth_m: Optional[np.ndarray] = None) -> List[Detection]:
        """Detect objects; results sorted by confidence (highest first)."""
        if self._model is None:
            return []
        h, w = frame_bgr.shape[:2]
        result = self._model(frame_bgr, conf=self.confidence, verbose=False)[0]

        detections: List[Detection] = []
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            yolo_label = result.names[int(box.cls[0])]
            size_px = float(max(x2 - x1, y2 - y1))
            size_cm = depth = None
            if depth_m is not None:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                patch = depth_m[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6]
                valid = patch[patch > 0]
                if valid.size:
                    depth = float(np.median(valid))
                    size_cm = estimate_size_cm(size_px, depth, w, self.hfov_deg)
            detections.append(Detection(
                label=map_label(yolo_label),
                yolo_label=yolo_label,
                confidence=float(box.conf[0]),
                bbox=(x1, y1, x2, y2),
                size_px=size_px,
                size_cm=size_cm,
                depth_m=depth,
            ))
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
