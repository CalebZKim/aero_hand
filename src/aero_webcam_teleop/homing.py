"""
Tendon-safe manual homing for the Aero Hand Open (standalone routine).

Strategy per actuator, chosen from measured current-sensor reliability:

  ID  Actuator         Sensor    Strategy
  0   thumb_cmc_abd    noisy     drive to a known safe position
  1   thumb_cmc_flex   noisy     drive to a known safe position
  2   thumb_tendon     inverted  torque until |current| crosses threshold
  3   index_tendon     noisy     drive to a known safe position
  4   middle_tendon    clean     torque until current crosses threshold
  5   ring_tendon      clean     torque until current crosses threshold
  6   pinky_tendon     clean     torque until current crosses threshold

A current drop to near zero after the tendon was confirmed loaded is treated
as a snapped tendon: torque is cut on everything and homing stops.

Run with:  aero-webcam-teleop home --port COM3
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .joints import ACTUATOR_NAMES, NUM_ACTUATORS, OPEN_POSE

POSITION = "position"
CURRENT = "current"


@dataclass(frozen=True)
class ActuatorHoming:
    strategy: str
    torque: int = 0
    threshold_ma: Optional[float] = None
    target_deg: Optional[float] = None
    current_sign: int = +1  # -1 where the current sensor reads inverted


ACTUATOR_HOMING: Dict[int, ActuatorHoming] = {
    0: ActuatorHoming(POSITION, torque=80, target_deg=85.0),
    1: ActuatorHoming(POSITION, torque=80, target_deg=45.0),
    2: ActuatorHoming(CURRENT, torque=80, threshold_ma=400, current_sign=-1),
    3: ActuatorHoming(POSITION, torque=100, target_deg=240.0),
    4: ActuatorHoming(CURRENT, torque=100, threshold_ma=500),
    5: ActuatorHoming(CURRENT, torque=100, threshold_ma=600),
    6: ActuatorHoming(CURRENT, torque=100, threshold_ma=200),
}

BACKOFF_DEG = 6.0
POLL_S = 0.05
TIMEOUT_S = 40.0
HOMING_SPEED = 1200
POSITION_TOLERANCE_DEG = 3.0
LOADED_FRACTION = 0.25    # current above 25% of threshold = tendon under load
LOADED_CONFIRM_CYCLES = 3
SNAP_DROP_MA = 35.0


class Result:
    CONTACT = "contact"
    POSITION = "position_target"
    TIMEOUT = "timeout"
    SNAP = "snap"
    SKIPPED = "skipped"


def _zero_torque(hand) -> None:
    try:
        hand.ctrl_torque([0] * NUM_ACTUATORS)
    except Exception:
        pass


def _home_by_position(hand, act: int, cfg: ActuatorHoming, timeout_s: float) -> str:
    target = list(hand.get_actuations())
    target[act] = cfg.target_deg
    hand.set_actuations(target)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        time.sleep(POLL_S)
        try:
            if abs(hand.get_actuations()[act] - cfg.target_deg) < POSITION_TOLERANCE_DEG:
                print(f"  reached {cfg.target_deg} deg ({time.monotonic() - t0:.1f}s)")
                return Result.POSITION
        except Exception:
            pass
    print("  timeout before reaching target")
    return Result.TIMEOUT


def _home_by_current(hand, act: int, cfg: ActuatorHoming, timeout_s: float, backoff_deg: float) -> str:
    cmd = [0] * NUM_ACTUATORS
    cmd[act] = cfg.torque
    hand.ctrl_torque(cmd)

    loaded_cycles, was_loaded = 0, False
    result = Result.TIMEOUT
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            current = hand.get_actuator_currents()[act] * cfg.current_sign
        except Exception:
            current = 0.0

        if current >= cfg.threshold_ma * LOADED_FRACTION:
            loaded_cycles += 1
            was_loaded = was_loaded or loaded_cycles >= LOADED_CONFIRM_CYCLES
        else:
            loaded_cycles = 0

        if was_loaded and current < SNAP_DROP_MA:
            _zero_torque(hand)
            print(f"\n  !!! EMERGENCY STOP: possible snapped tendon on "
                  f"{ACTUATOR_NAMES[act]} (current fell to {current:.0f} mA) !!!\n")
            return Result.SNAP

        if current >= cfg.threshold_ma:
            print(f"  contact at {current:.0f} mA ({time.monotonic() - t0:.1f}s)")
            result = Result.CONTACT
            break
        time.sleep(POLL_S)

    _zero_torque(hand)
    time.sleep(0.15)

    if result == Result.CONTACT:
        try:
            target = list(hand.get_actuations())
            target[act] += backoff_deg
            hand.set_actuations(target)
            time.sleep(0.3)
            print(f"  backed off {backoff_deg} deg")
        except Exception as exc:
            print(f"  back-off failed: {exc}")
    elif result == Result.TIMEOUT:
        print("  timeout without contact")
    return result


def safe_home(
    hand,
    actuators: Optional[List[int]] = None,
    backoff_deg: float = BACKOFF_DEG,
    timeout_s: float = TIMEOUT_S,
) -> Dict[int, str]:
    """Home the given actuators (default all) one at a time.

    ``hand`` is an ``aero_open_sdk.AeroHand`` (or anything with the same
    methods). Returns {actuator_id: Result}.
    """
    actuators = list(range(NUM_ACTUATORS)) if actuators is None else list(actuators)
    results = {i: Result.SKIPPED for i in range(NUM_ACTUATORS)}

    for i in range(NUM_ACTUATORS):
        hand.set_speed(i, HOMING_SPEED)
    time.sleep(0.1)

    snapped = False
    for act in actuators:
        cfg = ACTUATOR_HOMING[act]
        if snapped:
            results[act] = Result.SNAP
            continue
        print(f"[homing] {act} {ACTUATOR_NAMES[act]} ({cfg.strategy})")
        try:
            if cfg.strategy == POSITION:
                results[act] = _home_by_position(hand, act, cfg, timeout_s)
            else:
                results[act] = _home_by_current(hand, act, cfg, timeout_s, backoff_deg)
        except Exception as exc:
            print(f"  failed: {exc}")
            results[act] = Result.TIMEOUT
        snapped = results[act] == Result.SNAP
        time.sleep(0.2)

    _zero_torque(hand)
    time.sleep(0.3)
    if not snapped:
        try:
            for _ in range(2):  # seat the position register at open palm
                hand.set_joint_positions(list(OPEN_POSE))
                time.sleep(0.4)
        except Exception as exc:
            print(f"[homing] Could not command open palm: {exc}")

    print("\n[homing] Summary")
    for act, res in results.items():
        if res != Result.SKIPPED:
            print(f"  {ACTUATOR_NAMES[act]:<16} {res}")
    if snapped:
        print("\n  Stopped after a suspected snap. Inspect the tendons before continuing.")
    else:
        print("\n[homing] Done. Hand is at open palm; safe to start teleop.")
    return results
