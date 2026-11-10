"""
Camera capture + MediaPipe hand tracking.

* Uses the MediaPipe **Tasks** ``HandLandmarker`` API. Current MediaPipe
  releases (0.10.3x) no longer ship ``mp.solutions``; the legacy API is used
  only if it is the only one available.
* Joint angles are computed from a geometrically consistent 3-D point set
  (``HandMocap.points``), never from normalised image x/y mixed with depth in
  other units. The default is MediaPipe's metric *world* landmarks; with a
  D405 and ``use_depth_for_hand=True`` the pixels are deprojected to metres
  with the camera intrinsics.
* Tracking is separated from display: frames go to an ``on_frame`` callback
  (with the depth image, so grasp mode can use it) and the caller decides
  what to draw and show.
"""

from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .joints import NUM_LANDMARKS

try:
    import pyrealsense2 as rs  # type: ignore
    REALSENSE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on hardware SDK
    rs = None
    REALSENSE_AVAILABLE = False


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Keypoint3D:
    x: float = 0.0  # normalised [0, 1] across image width
    y: float = 0.0  # normalised [0, 1] across image height
    z: float = 0.0  # MediaPipe relative depth (for drawing only)


def _empty_keypoints() -> List[Keypoint3D]:
    return [Keypoint3D() for _ in range(NUM_LANDMARKS)]


def _empty_points() -> np.ndarray:
    return np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)


@dataclass
class HandMocap:
    """One tracked hand.

    ``keypoints`` are image-space landmarks (for drawing). ``points`` is a
    (21, 3) array in consistent units, which is what retargeting uses.
    """

    side: str = "right"
    keypoints: List[Keypoint3D] = field(default_factory=_empty_keypoints)
    points: np.ndarray = field(default_factory=_empty_points)
    points_source: str = "image"  # "world" | "depth" | "image"
    timestamp: float = 0.0
    camera: str = "webcam"  # "webcam" | "realsense"


@dataclass
class Frame:
    bgr: np.ndarray
    depth_m: Optional[np.ndarray] = None  # aligned depth in metres (D405 only)


# ---------------------------------------------------------------------------
# Camera backends
# ---------------------------------------------------------------------------

class WebcamCamera:
    name = "webcam"
    label = "WEBCAM"

    def __init__(self, index: int = 0):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")

    def read(self) -> Optional[Frame]:
        ok, bgr = self._cap.read()
        return Frame(bgr=bgr) if ok else None

    def deproject(self, u: float, v: float, depth_m: float) -> Optional[np.ndarray]:
        return None

    def release(self) -> None:
        self._cap.release()


class RealSenseCamera:  # pragma: no cover - needs hardware
    """Intel RealSense D405: colour + depth aligned to colour."""

    name = "realsense"
    label = "REALSENSE D405"

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not installed")
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._intrinsics = (
            profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        )

    def read(self) -> Optional[Frame]:
        frames = self._align.process(self._pipeline.wait_for_frames(timeout_ms=5000))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        bgr = np.asanyarray(color.get_data())
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale
        return Frame(bgr=bgr, depth_m=depth_m)

    def deproject(self, u: float, v: float, depth_m: float) -> Optional[np.ndarray]:
        x, y, z = rs.rs2_deproject_pixel_to_point(self._intrinsics, [u, v], depth_m)
        return np.array([x, y, z], dtype=np.float64)

    def release(self) -> None:
        self._pipeline.stop()


def open_camera(index: int = 0, force_webcam: bool = False):
    """Prefer a RealSense D405 if one is connected, otherwise use a webcam."""
    if not force_webcam and REALSENSE_AVAILABLE:
        try:
            cam = RealSenseCamera()
            print("[tracking] RealSense D405 detected, using depth camera.")
            return cam
        except Exception as exc:  # pragma: no cover
            print(f"[tracking] RealSense unavailable ({exc}), falling back to webcam.")
    print(f"[tracking] Using webcam index {index}.")
    return WebcamCamera(index)


def depth_at(depth_m: np.ndarray, u: float, v: float, radius: int = 2) -> float:
    """Median of non-zero depth in a (2r+1)^2 patch around pixel (u, v)."""
    h, w = depth_m.shape[:2]
    px, py = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    patch = depth_m[max(0, py - radius):py + radius + 1, max(0, px - radius):px + radius + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


# ---------------------------------------------------------------------------
# Landmark detectors (Tasks API, legacy fallback)
# ---------------------------------------------------------------------------

# Raw detector output: (mediapipe handedness label, image landmarks, world landmarks)
RawHand = Tuple[str, List[Tuple[float, float, float]], Optional[List[Tuple[float, float, float]]]]


def default_model_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "aero_webcam_teleop" / "hand_landmarker.task"


def ensure_hand_model(path: Optional[Path] = None) -> Path:
    """Return the hand landmarker model path, downloading it on first use."""
    path = Path(path) if path else default_model_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[tracking] Downloading hand landmark model to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
        tmp.replace(path)
    return path


class _TasksDetector:
    def __init__(self, model_path: Optional[Path], max_hands: int):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_hand_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb: np.ndarray) -> List[RawHand]:
        ts = int(time.monotonic() * 1000)
        ts = max(ts, self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        hands: List[RawHand] = []
        for i, lms in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name
            img = [(p.x, p.y, p.z) for p in lms]
            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[i]]
            hands.append((label, img, world))
        return hands

    def close(self) -> None:
        self._landmarker.close()


class _LegacyDetector:  # pragma: no cover - only on old MediaPipe
    def __init__(self, max_hands: int):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb: np.ndarray) -> List[RawHand]:
        res = self._hands.process(rgb)
        hands: List[RawHand] = []
        if not res.multi_hand_landmarks:
            return hands
        worlds = res.multi_hand_world_landmarks or [None] * len(res.multi_hand_landmarks)
        for lms, handed, world in zip(res.multi_hand_landmarks, res.multi_handedness, worlds):
            img = [(p.x, p.y, p.z) for p in lms.landmark]
            w = [(p.x, p.y, p.z) for p in world.landmark] if world is not None else None
            hands.append((handed.classification[0].label, img, w))
        return hands

    def close(self) -> None:
        self._hands.close()


def make_detector(model_path: Optional[Path] = None, max_hands: int = 2):
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return _TasksDetector(model_path, max_hands)
    except ImportError:  # pragma: no cover
        return _LegacyDetector(max_hands)


# ---------------------------------------------------------------------------
# Building HandMocap from raw detections
# ---------------------------------------------------------------------------

def physical_side(mediapipe_label: str, mirrored: bool = False) -> str:
    """MediaPipe labels handedness as if the image were mirrored (selfie view).

    A raw, un-flipped webcam frame therefore reports the opposite hand.
    """
    label = mediapipe_label.lower()
    if mirrored:
        return label
    return "left" if label == "right" else "right"


def build_points(
    image_lms: List[Tuple[float, float, float]],
    world_lms: Optional[List[Tuple[float, float, float]]],
    frame_shape: Tuple[int, ...],
    depth_m: Optional[np.ndarray] = None,
    deproject: Optional[Callable[[float, float, float], Optional[np.ndarray]]] = None,
) -> Tuple[np.ndarray, str]:
    """Pick the best consistent 3-D point set for retargeting.

    Order of preference: deprojected depth (if requested and every landmark
    has a valid depth reading), MediaPipe world landmarks (metric), and
    finally aspect-corrected image landmarks.
    """
    h, w = frame_shape[:2]

    if depth_m is not None and deproject is not None:
        pts = []
        for x, y, _ in image_lms:
            u, v = x * w, y * h
            d = depth_at(depth_m, u, v)
            p = deproject(u, v, d) if d > 0 else None
            if p is None:
                pts = None
                break
            pts.append(p)
        if pts is not None:
            return np.asarray(pts, dtype=np.float64), "depth"

    if world_lms is not None:
        return np.asarray(world_lms, dtype=np.float64), "world"

    # MediaPipe image z uses roughly the same scale as x (image width), so
    # rescale y from "fraction of height" to "fraction of width".
    arr = np.asarray(image_lms, dtype=np.float64)
    arr[:, 1] *= h / w
    return arr, "image"


# ---------------------------------------------------------------------------
# Tracker loop
# ---------------------------------------------------------------------------

class HandTracker:
    """Capture loop: camera -> MediaPipe -> ``on_mocap`` / ``on_frame``.

    Parameters
    ----------
    camera_index       : webcam index (ignored when a D405 is used)
    target_fps         : capture-rate cap (Hz)
    hand_side          : "right" | "left" | None (first hand found)
    on_mocap           : callback(HandMocap) for each tracked hand
    on_frame           : callback(Frame, Optional[HandMocap]) once per frame,
                         after on_mocap. Return False to stop the loop.
    force_webcam       : skip RealSense even if connected
    use_depth_for_hand : with a D405, build hand points from depth
    mirrored           : set True if the camera already flips the image
    model_path         : custom hand_landmarker.task path
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_fps: float = 30.0,
        hand_side: Optional[str] = "right",
        on_mocap: Optional[Callable[[HandMocap], None]] = None,
        on_frame: Optional[Callable[[Frame, Optional[HandMocap]], Optional[bool]]] = None,
        force_webcam: bool = False,
        use_depth_for_hand: bool = False,
        mirrored: bool = False,
        model_path: Optional[Path] = None,
    ):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.hand_side = hand_side
        self.on_mocap = on_mocap
        self.on_frame = on_frame
        self.force_webcam = force_webcam
        self.use_depth_for_hand = use_depth_for_hand
        self.mirrored = mirrored
        self.model_path = model_path
        self.camera_label = ""
        self._running = False

    def _select(self, hands: List[RawHand]) -> Optional[Tuple[str, RawHand]]:
        for raw in hands:
            side = physical_side(raw[0], self.mirrored)
            if self.hand_side is None or side == self.hand_side:
                return side, raw
        return None

    def run(self) -> None:
        """Blocking capture loop. Call ``stop()`` (or return False from on_frame) to end."""
        self._running = True
        camera = open_camera(self.camera_index, self.force_webcam)
        self.camera_label = camera.label
        detector = make_detector(self.model_path)
        interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        try:
            while self._running:
                t0 = time.monotonic()
                frame = camera.read()
                if frame is None:
                    continue

                rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
                picked = self._select(detector.process(rgb))

                mocap: Optional[HandMocap] = None
                if picked is not None:
                    side, (_, image_lms, world_lms) = picked
                    use_depth = self.use_depth_for_hand and frame.depth_m is not None
                    points, source = build_points(
                        image_lms,
                        world_lms,
                        frame.bgr.shape,
                        depth_m=frame.depth_m if use_depth else None,
                        deproject=camera.deproject if use_depth else None,
                    )
                    mocap = HandMocap(
                        side=side,
                        keypoints=[Keypoint3D(x, y, z) for x, y, z in image_lms],
                        points=points,
                        points_source=source,
                        timestamp=time.monotonic(),
                        camera=camera.name,
                    )
                    if self.on_mocap:
                        self.on_mocap(mocap)

                if self.on_frame and self.on_frame(frame, mocap) is False:
                    break

                sleep = interval - (time.monotonic() - t0)
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            detector.close()
            camera.release()
            self._running = False

    def stop(self) -> None:
        self._running = False
