"""
Command line entry point.

    aero-webcam-teleop [run] [options]   # teleop + grasp mode (default)
    aero-webcam-teleop track [options]   # tracking + joint angles only, no hand
    aero-webcam-teleop home --port COM3  # tendon-safe manual homing
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

COMMANDS = ("run", "track", "home")


def _add_camera_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--fps", type=float, default=30.0, help="capture rate cap in Hz")
    p.add_argument("--force-webcam", action="store_true", help="ignore an attached RealSense")
    p.add_argument("--depth-hand", action="store_true",
                   help="with a D405, build hand points from depth instead of MediaPipe world landmarks")
    p.add_argument("--smoothing", type=float, default=0.35,
                   help="EMA weight on the newest sample, 0..1 (lower = smoother)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aero-webcam-teleop",
                                     description="Webcam teleoperation for the TetherIA Aero Hand Open")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="teleop + grasp mode (default)")
    run.add_argument("--port", help="serial port, e.g. COM3 or /dev/ttyACM0 (omit for mock mode)")
    run.add_argument("--baudrate", type=int, default=921_600)
    _add_camera_args(run)
    run.add_argument("--control-hz", type=float, default=20.0, help="command rate to the hand")
    run.add_argument("--auto-grasp-s", type=float, default=3.0,
                     help="seconds between detections in auto-grasp mode")
    run.add_argument("--no-grasp", action="store_true", help="disable YOLO + planner")
    run.add_argument("--verbose", action="store_true", help="print every command in mock mode")

    track = sub.add_parser("track", help="show tracking and joint angles without a hand")
    _add_camera_args(track)

    home = sub.add_parser("home", help="tendon-safe manual homing")
    home.add_argument("--port", required=True)
    home.add_argument("--actuators", type=int, nargs="+", choices=range(7), metavar="ID",
                      help="actuator ids 0-6 to home (default all)")
    home.add_argument("--backoff", type=float, default=6.0, help="degrees to back off after contact")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, "run")
    args = build_parser().parse_args(argv)

    if args.command == "home":
        from .bridge import SDK_AVAILABLE, AeroHand
        from .homing import safe_home
        if not SDK_AVAILABLE:
            print("aero_open_sdk is not installed (pip install aero-open-sdk).")
            return 1
        safe_home(AeroHand(port=args.port), actuators=args.actuators, backoff_deg=args.backoff)
        return 0

    if args.command == "track":
        return _track(args)

    from .pipeline import WebcamTeleopPipeline
    WebcamTeleopPipeline(
        port=args.port,
        baudrate=args.baudrate,
        side=args.side,
        camera_index=args.camera,
        target_fps=args.fps,
        control_hz=args.control_hz,
        smoothing_alpha=args.smoothing,
        verbose=args.verbose,
        force_webcam=args.force_webcam,
        use_depth_for_hand=args.depth_hand,
        auto_grasp_s=args.auto_grasp_s,
        enable_grasp=not args.no_grasp,
    ).run()
    return 0


def _track(args) -> int:
    import cv2

    from .joints import JOINT_NAMES
    from .overlay import draw_hand, draw_status
    from .retargeting import Retargeter
    from .tracking import HandTracker

    retargeter = Retargeter(smoothing_alpha=args.smoothing)
    state = {"angles": None, "calibrate": False, "last_print": 0.0}

    def on_mocap(m):
        if state["calibrate"]:
            retargeter.calibrate(m)
            state["calibrate"] = False
        state["angles"] = retargeter.retarget(m)
        if m.timestamp - state["last_print"] > 0.5:
            state["last_print"] = m.timestamp
            print(f"[{m.points_source}] " + " ".join(
                f"{n.split('_', 1)[0][:2]}{n.split('_', 1)[1]}={a:.0f}"
                for n, a in zip(JOINT_NAMES, state["angles"])))

    def on_frame(frame, m):
        img = frame.bgr.copy()
        if m is not None:
            draw_hand(img, m.keypoints, m.side)
        draw_status(img, side=args.side, hardware="NO HAND", fps=0.0, camera=tracker.camera_label,
                    state="teleop", auto_grasp=False, llm=False,
                    calibrated=retargeter.config.is_calibrated, homing=False, angles=state["angles"])
        cv2.imshow("Aero Hand - tracking", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            state["calibrate"] = True
        return key not in (ord("q"), 27)

    tracker = HandTracker(camera_index=args.camera, target_fps=args.fps, hand_side=args.side,
                          on_mocap=on_mocap, on_frame=on_frame, force_webcam=args.force_webcam,
                          use_depth_for_hand=args.depth_hand)
    try:
        tracker.run()
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
