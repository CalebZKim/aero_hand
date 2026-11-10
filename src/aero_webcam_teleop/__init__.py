"""
aero_webcam_teleop
==================
Webcam teleoperation for the TetherIA Aero Hand Open, without ROS 2.

    camera -> MediaPipe hand landmarks -> 16 joint angles -> aero_open_sdk

plus an optional grasp mode (YOLOv8 detection -> Claude grasp planner ->
pose library).

Quick start::

    aero-webcam-teleop                    # mock mode, no hardware
    aero-webcam-teleop --port COM3        # real hand
"""

from .bridge import AeroHandBridge
from .retargeting import Retargeter, RetargetingConfig, raw_joint_angles
from .tracking import HandMocap, HandTracker, Keypoint3D

__version__ = "0.2.0"

__all__ = [
    "AeroHandBridge",
    "HandMocap",
    "HandTracker",
    "Keypoint3D",
    "Retargeter",
    "RetargetingConfig",
    "WebcamTeleopPipeline",
    "raw_joint_angles",
]


def __getattr__(name):
    # Import lazily so `import aero_webcam_teleop` doesn't load YOLO/OpenCV windows.
    if name == "WebcamTeleopPipeline":
        from .pipeline import WebcamTeleopPipeline
        return WebcamTeleopPipeline
    raise AttributeError(name)
