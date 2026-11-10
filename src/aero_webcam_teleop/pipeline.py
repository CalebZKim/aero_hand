"""
Top-level pipeline:

    camera -> HandTracker -> Retargeter --------------------> AeroHandBridge
                                                                  ^
    'g' key -> GraspDetector -> GraspPlanner -> pose library -----+

Threads: the camera/display loop runs on the main thread (OpenCV windows
need it), the control loop runs at ``control_hz`` in a worker thread, and
detection/planning/homing each run in short-lived worker threads.

State machine
-------------
    TELEOP     hand follows your hand (after a short warm-up)
    MANUAL     holds a fixed pose ('f' = fist) until 'o'
    DETECTING  YOLO running on a snapshot of the current frame
    PLANNING   waiting on the grasp planner
    EXECUTING  blending from the current pose to the grasp pose
    HOLDING    holding the grasp until 'g' or 'o'

Behaviour notes:
* 'f' (fist) holds until you press 'o'; teleop doesn't override it.
* Losing the hand stops commands, resets smoothing, and requires a fresh
  warm-up when it comes back.
* A planner request rejected by its cooldown returns to teleop.
* 'g' snapshots the newest frame directly.
* A plan that arrives after you cancelled with 'o' is ignored.
* The RealSense depth image reaches the detector for size/distance.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import cv2

from .bridge import NORMAL_SPEED, AeroHandBridge
from .grasp import GraspDetector, GraspPlan, GraspPlanner, get_force, get_pose, interpolate
from .joints import FIST_POSE, OPEN_POSE
from .overlay import draw_detection, draw_hand, draw_status
from .retargeting import RetargetingConfig, Retargeter
from .tracking import Frame, HandMocap, HandTracker


class State:
    TELEOP = "teleop"
    MANUAL = "manual"
    DETECTING = "detecting"
    PLANNING = "planning"
    EXECUTING = "executing"
    HOLDING = "holding"


GRASP_ACTIVE = (State.DETECTING, State.PLANNING, State.EXECUTING, State.HOLDING)
WINDOW_NAME = "Aero Hand - Webcam Teleop"


class WebcamTeleopPipeline:
    """
    Parameters
    ----------
    port, baudrate, side      : hand connection (port=None -> mock)
    camera_index, target_fps  : capture settings
    control_hz                : command rate to the hand
    smoothing_alpha           : EMA weight on newest sample (lower = smoother)
    verbose                   : log every command in mock mode
    force_webcam              : ignore an attached RealSense
    use_depth_for_hand        : build hand points from D405 depth
    auto_grasp_s              : interval between detections in auto mode
    grasp_duration_s          : time to blend into a grasp pose
    enable_grasp              : set False to skip loading YOLO/planner
    bridge, detector, planner : inject components (used by tests)
    """

    WARMUP_FRAMES = 5      # stable frames before teleop sends commands
    HAND_LOST_S = 0.5      # no detection for this long = hand lost

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 921_600,
        side: str = "right",
        camera_index: int = 0,
        target_fps: float = 30.0,
        control_hz: float = 20.0,
        smoothing_alpha: float = 0.35,
        verbose: bool = False,
        force_webcam: bool = False,
        use_depth_for_hand: bool = False,
        auto_grasp_s: float = 3.0,
        grasp_duration_s: float = 1.5,
        enable_grasp: bool = True,
        bridge: Optional[AeroHandBridge] = None,
        detector: Optional[GraspDetector] = None,
        planner: Optional[GraspPlanner] = None,
    ):
        self.side = side
        self.control_hz = control_hz
        self.auto_grasp_s = auto_grasp_s
        self.grasp_duration_s = grasp_duration_s

        self._lock = threading.RLock()
        self._running = False
        self._state = State.TELEOP

        # teleop
        self._angles: Optional[List[float]] = None
        self._last_mocap_t = -float("inf")
        self._stable_frames = 0
        self._calibrate_next = False
        self._last_sent: List[float] = list(OPEN_POSE)
        self._manual_pose: List[float] = list(OPEN_POSE)

        # grasp
        self._latest_frame: Optional[Frame] = None
        self._detection = None
        self._plan: Optional[GraspPlan] = None
        self._grasp_from: List[float] = list(OPEN_POSE)
        self._grasp_to: List[float] = list(OPEN_POSE)
        self._grasp_t0 = 0.0
        self._auto_grasp = False
        self._last_auto_t = 0.0

        # display
        self._last_mocap: Optional[HandMocap] = None
        self._fps = 0.0
        self._fps_n = 0
        self._fps_t = time.monotonic()

        self.retargeter = Retargeter(RetargetingConfig(), smoothing_alpha)
        self.bridge = bridge or AeroHandBridge(port=port, baudrate=baudrate, side=side,
                                               verbose=verbose)
        if enable_grasp:
            self.detector = detector or GraspDetector()
            self.planner = planner or GraspPlanner()
        else:
            self.detector, self.planner = detector, planner
        if self.planner is not None:
            self.planner.on_plan = self._on_plan

        self.tracker = HandTracker(
            camera_index=camera_index,
            target_fps=target_fps,
            hand_side=side,
            on_mocap=self._on_mocap,
            on_frame=self._on_frame,
            force_webcam=force_webcam,
            use_depth_for_hand=use_depth_for_hand,
        )

    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    # ------------------------------------------------------------------
    # Tracking callbacks (camera thread)
    # ------------------------------------------------------------------

    def _on_mocap(self, mocap: HandMocap) -> None:
        with self._lock:
            if mocap.timestamp - self._last_mocap_t > self.HAND_LOST_S:
                # Hand (re)appeared: start a fresh warm-up and drop stale smoothing.
                self._stable_frames = 0
                self.retargeter.reset()
            if self._calibrate_next:
                self.retargeter.calibrate(mocap)
                self._calibrate_next = False
            self._angles = self.retargeter.retarget(mocap)
            self._last_mocap_t = mocap.timestamp
            self._stable_frames += 1
            self._last_mocap = mocap

    def _on_frame(self, frame: Frame, mocap: Optional[HandMocap]) -> bool:
        now = time.monotonic()
        with self._lock:
            self._latest_frame = frame
            state = self._state
        self._fps_n += 1
        if now - self._fps_t >= 1.0:
            self._fps = self._fps_n / (now - self._fps_t)
            self._fps_n, self._fps_t = 0, now

        if self._auto_grasp and state == State.TELEOP and now - self._last_auto_t >= self.auto_grasp_s:
            self._last_auto_t = now
            self.trigger_grasp()

        img = frame.bgr.copy()
        self._draw(img, mocap)
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            return self.handle_key(key)
        return True

    def _draw(self, img, mocap: Optional[HandMocap]) -> None:
        with self._lock:
            state, plan, det = self._state, self._plan, self._detection
            angles = self._last_sent if state != State.TELEOP else self._angles
        if mocap is not None:
            draw_hand(img, mocap.keypoints, mocap.side)
        if det is not None and state in GRASP_ACTIVE:
            draw_detection(img, det)
        draw_status(
            img,
            side=self.side,
            hardware="MOCK" if self.bridge.is_mock else "HARDWARE",
            fps=self._fps,
            camera=self.tracker.camera_label,
            state=state,
            auto_grasp=self._auto_grasp,
            llm=bool(self.planner and self.planner.uses_llm),
            calibrated=self.retargeter.config.is_calibrated,
            homing=self.bridge.is_homing,
            angles=angles,
            plan=plan if state in GRASP_ACTIVE else None,
        )

    # ------------------------------------------------------------------
    # Control loop (worker thread)
    # ------------------------------------------------------------------

    def control_step(self, now: Optional[float] = None) -> Optional[List[float]]:
        """One control tick. Returns the command sent (None if nothing sent)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            state = self._state
            if state == State.TELEOP:
                fresh = now - self._last_mocap_t <= self.HAND_LOST_S
                if not (fresh and self._stable_frames >= self.WARMUP_FRAMES and self._angles):
                    return None  # hand lost or warming up: hold last command
                cmd = list(self._angles)
            elif state == State.MANUAL:
                cmd = list(self._manual_pose)
            elif state == State.EXECUTING:
                t = (now - self._grasp_t0) / self.grasp_duration_s if self.grasp_duration_s > 0 else 1.0
                cmd = interpolate(self._grasp_from, self._grasp_to, t)
                if t >= 1.0:
                    self._state = State.HOLDING
                    print("[pipeline] Grasp pose reached, holding. Press g or o to release.")
            elif state == State.HOLDING:
                cmd = list(self._grasp_to)
            else:  # DETECTING / PLANNING: keep the hand where it is
                return None
        if self.bridge.send_joint_positions(cmd):
            with self._lock:
                self._last_sent = cmd
            return cmd
        return None

    def _control_loop(self) -> None:
        interval = 1.0 / self.control_hz
        while self._running:
            t0 = time.monotonic()
            try:
                self.control_step(t0)
            except Exception as exc:  # keep the loop alive; report and continue
                print(f"[pipeline] control error: {exc}")
            sleep = interval - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------------
    # Grasp mode
    # ------------------------------------------------------------------

    def trigger_grasp(self) -> bool:
        """Snapshot the newest frame and start detection. False if not possible."""
        if self.detector is None or self.planner is None:
            print("[pipeline] Grasp mode is disabled.")
            return False
        with self._lock:
            if self._state not in (State.TELEOP, State.MANUAL) or self._latest_frame is None:
                return False
            frame = self._latest_frame
            self._state = State.DETECTING
            self._plan = self._detection = None
        print("[pipeline] Detecting objects...")
        threading.Thread(target=self._detect_and_plan, args=(frame,), daemon=True).start()
        return True

    def _detect_and_plan(self, frame: Frame) -> None:
        try:
            detections = self.detector.detect(frame.bgr, frame.depth_m)
        except Exception as exc:
            print(f"[pipeline] Detection failed: {exc}")
            detections = []
        with self._lock:
            if self._state != State.DETECTING:
                return  # cancelled
            if not detections:
                print("[pipeline] No objects detected, back to teleop.")
                self._state = State.TELEOP
                return
            top = detections[0]
            self._detection = top
            self._state = State.PLANNING
        print(f"[pipeline] Detected {top.label} (YOLO '{top.yolo_label}', {top.confidence:.0%})")
        if not self.planner.request_plan(top.label, top.size_cm, top.depth_m, top.confidence):
            print("[pipeline] Planner busy or cooling down, back to teleop.")
            with self._lock:
                if self._state == State.PLANNING:
                    self._state = State.TELEOP

    def _on_plan(self, plan: GraspPlan) -> None:
        with self._lock:
            if self._state != State.PLANNING:
                return  # cancelled while the planner was thinking
            self._plan = plan
            self._grasp_from = list(self._last_sent)
            self._grasp_to = list(get_pose(plan.grasp_type).angles)
            self._grasp_t0 = time.monotonic()
            self._state = State.EXECUTING
        print(f"[pipeline] Plan: {plan.grasp_type} / {plan.force_level} [{plan.source}] {plan.reasoning}")
        self.bridge.set_speed(int(get_force(plan.force_level).speed_scalar * self.bridge.normal_speed))

    # ------------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------------

    def return_to_teleop(self) -> None:
        """Open the hand and resume following (after warm-up)."""
        with self._lock:
            self._state = State.TELEOP
            self._plan = self._detection = None
            self._stable_frames = 0
            self._last_sent = list(OPEN_POSE)
        self.bridge.set_speed(self.bridge.normal_speed)
        self.bridge.open_hand()

    def handle_key(self, key: int) -> bool:
        """Handle a key press. Returns False to quit."""
        ch = chr(key).lower() if key < 128 else ""
        if key == 27 or ch == "q":
            return False
        if ch == "c":
            print("[pipeline] Calibrating on the next frame: hold your hand open and flat.")
            self._calibrate_next = True
        elif ch == "h":
            threading.Thread(target=self.bridge.home, daemon=True).start()
        elif ch == "o":
            print("[pipeline] Open hand, teleop resumed.")
            self.return_to_teleop()
        elif ch == "f":
            print("[pipeline] Holding fist. Press o to resume teleop.")
            with self._lock:
                self._state = State.MANUAL
                self._manual_pose = list(FIST_POSE)
        elif ch == "g":
            if self._state in (State.EXECUTING, State.HOLDING):
                print("[pipeline] Releasing grasp.")
                self.return_to_teleop()
            else:
                self.trigger_grasp()
        elif ch == "a":
            self._auto_grasp = not self._auto_grasp
            self._last_auto_t = time.monotonic()
            print(f"[pipeline] Auto-grasp {'on' if self._auto_grasp else 'off'} "
                  f"(every {self.auto_grasp_s:g}s)")
        return True

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start everything and block until 'q'/ESC or Ctrl+C."""
        self._running = True
        if not self.bridge.is_mock:
            print("[pipeline] Moving hand to open palm before starting...")
            self.bridge.open_hand()
            time.sleep(0.8)
        ctrl = threading.Thread(target=self._control_loop, daemon=True)
        ctrl.start()
        print(f"\nAero Hand teleop | {self.side} hand | "
              f"{'MOCK' if self.bridge.is_mock else 'HARDWARE'} | {self.control_hz:g} Hz control")
        print("Click the video window, then: c=calibrate h=home o=open/teleop f=fist "
              "g=grasp a=auto-grasp q=quit\n")
        try:
            self.tracker.run()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            ctrl.join(timeout=1.0)
            self.bridge.close()
            cv2.destroyAllWindows()
