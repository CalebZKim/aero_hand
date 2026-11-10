"""
Hardware bridge for the TetherIA Aero Hand Open (wraps ``aero_open_sdk``).

Falls back to a mock that logs commands when the SDK is missing, no port is
given, or the connection fails, so everything else runs without hardware.

* All SDK calls go through one lock, since the control loop, the key
  handler and the homing thread each talk to the serial port.
* Joint commands are suppressed while homing is running, so teleop can't
  fight the homing routine.
* ``close()`` releases the serial connection when the SDK exposes a way to
  do that.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Sequence

from .joints import FIST_POSE, JOINT_NAMES, NUM_ACTUATORS, OPEN_POSE, SPEED_MAX, clamp_joints

try:
    from aero_open_sdk.aero_hand import AeroHand  # type: ignore
    SDK_AVAILABLE = True
except ImportError:
    AeroHand = None  # type: ignore
    SDK_AVAILABLE = False

NORMAL_SPEED = 20000      # ~60% of max: responsive teleop
SDK_HOMING_SPEED = 500    # ~1.5% of max: tendon-safe, tested on hardware
SDK_HOMING_SPEED_CAP = 1500  # don't exceed without checking cable tension


class AeroHandBridge:
    """
    Parameters
    ----------
    port         : serial port ("COM3", "/dev/ttyACM0"); None = mock mode
    baudrate     : serial baud (the SDK default is 921600)
    side         : "right" | "left" (informational)
    verbose      : in mock mode, print every joint command
    normal_speed : actuator speed (0..32766) used for teleop
    hand         : inject an SDK-like object (used by tests)
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 921_600,
        side: str = "right",
        verbose: bool = False,
        normal_speed: int = NORMAL_SPEED,
        hand=None,
    ):
        self.side = side
        self.verbose = verbose
        self.normal_speed = normal_speed
        self._io = threading.RLock()
        self._homing = threading.Event()
        self._hand = hand

        if self._hand is None and port:
            if not SDK_AVAILABLE:
                print("[bridge] aero_open_sdk not installed, running in MOCK mode.")
            else:
                try:
                    self._hand = AeroHand(port=port, baudrate=baudrate)
                except TypeError:
                    self._hand = AeroHand(port=port)  # SDK versions without baudrate kwarg
                except Exception as exc:
                    print(f"[bridge] Could not connect on {port}: {exc}. Running in MOCK mode.")
        elif self._hand is None:
            print("[bridge] No port given, running in MOCK mode.")

        if self._hand is not None:
            print(f"[bridge] Connected to {side} hand" + (f" on {port}" if port else ""))
            self.set_speed(self.normal_speed)

    # ------------------------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self._hand is None

    @property
    def is_homing(self) -> bool:
        return self._homing.is_set()

    @property
    def sdk(self):
        """The underlying SDK object (None in mock mode)."""
        return self._hand

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def send_joint_positions(self, angles_deg: Sequence[float]) -> bool:
        """Clamp and send 16 joint angles. Returns False if suppressed (homing)."""
        clamped = clamp_joints(angles_deg)
        if self._homing.is_set():
            return False
        if self._hand is None:
            if self.verbose:
                print("[MOCK] " + " ".join(f"{n}={a:.0f}" for n, a in zip(JOINT_NAMES, clamped)))
            return True
        with self._io:
            self._hand.set_joint_positions(clamped)
        return True

    def set_speed(self, speed: int) -> None:
        """Set all actuator speeds (0..32766)."""
        speed = int(max(0, min(SPEED_MAX, speed)))
        if self._hand is None:
            return
        with self._io:
            for i in range(NUM_ACTUATORS):
                self._hand.set_speed(i, speed)

    def open_hand(self) -> None:
        self.send_joint_positions(OPEN_POSE)

    def close_hand(self) -> None:
        self.send_joint_positions(FIST_POSE)

    def home(self, speed: int = SDK_HOMING_SPEED) -> None:
        """Run the firmware homing routine at a tendon-safe speed (blocking).

        Teleop commands are ignored until homing finishes.
        """
        speed = min(speed, SDK_HOMING_SPEED_CAP)
        if self._hand is None:
            print("[MOCK] home")
            return
        if self._homing.is_set():
            print("[bridge] Homing already running.")
            return
        self._homing.set()
        try:
            with self._io:
                print(f"[bridge] Homing at speed {speed}/{SPEED_MAX}. "
                      "Unplug USB if anything looks wrong.")
                self.set_speed(speed)
                time.sleep(0.2)
                self._hand.send_homing()
                self.set_speed(self.normal_speed)
            print("[bridge] Homing complete.")
        finally:
            self._homing.clear()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_actuator_states(self) -> Dict[str, List[float]]:
        if self._hand is None:
            return {
                "actuations": [0.0] * NUM_ACTUATORS,
                "speeds": [0.0] * NUM_ACTUATORS,
                "currents": [0.0] * NUM_ACTUATORS,
                "temperatures": [25.0] * NUM_ACTUATORS,
            }
        with self._io:
            return {
                "actuations": list(self._hand.get_actuations()),
                "speeds": list(self._hand.get_actuator_speeds()),
                "currents": list(self._hand.get_actuator_currents()),
                "temperatures": list(self._hand.get_actuator_temperatures()),
            }

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Open the hand and release the serial port."""
        if self._hand is not None:
            try:
                self._homing.clear()
                self.open_hand()
                time.sleep(0.5)
            finally:
                with self._io:
                    for name in ("close", "disconnect"):
                        fn = getattr(self._hand, name, None)
                        if callable(fn):
                            try:
                                fn()
                            except Exception as exc:
                                print(f"[bridge] {name}() failed: {exc}")
                            break
                self._hand = None
        print("[bridge] Closed.")
