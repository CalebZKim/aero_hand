"""State-machine tests: no camera, no window; control_step() is driven directly."""

import time

import numpy as np
import pytest
from conftest import mocap_from, synthetic_hand

from aero_webcam_teleop.bridge import AeroHandBridge
from aero_webcam_teleop.grasp import Detection, GraspPlan
from aero_webcam_teleop.grasp.poses import GRASP_POSES
from aero_webcam_teleop.joints import FIST_POSE, OPEN_POSE
from aero_webcam_teleop.pipeline import State, WebcamTeleopPipeline
from aero_webcam_teleop.tracking import Frame


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections
        self.seen = []

    def detect(self, bgr, depth_m=None):
        self.seen.append((bgr, depth_m))
        return list(self.detections)


class FakePlanner:
    uses_llm = False

    def __init__(self, accept=True):
        self.accept, self.requests, self.on_plan = accept, [], None

    def request_plan(self, label, size_cm=None, depth_m=None, confidence=1.0):
        self.requests.append(label)
        return self.accept

    def respond(self, grasp="power", force="medium"):
        self.on_plan(GraspPlan(self.requests[-1], grasp, force, "test", "fallback"))


CUP = Detection("cylinder", "cup", 0.9, (10, 10, 50, 50), 40.0, size_cm=8.0, depth_m=0.3)


def make(fake_hand, detections=(CUP,), accept=True):
    p = WebcamTeleopPipeline(bridge=AeroHandBridge(hand=fake_hand),
                             detector=FakeDetector(detections), planner=FakePlanner(accept))
    p.retargeter.alpha = 1.0
    return p


def feed(p, points, n=1, t0=None):
    t = time.monotonic() if t0 is None else t0
    for i in range(n):
        p._on_mocap(mocap_from(points, t + i * 0.03))
    return t + (n - 1) * 0.03


def give_frame(p):
    p._latest_frame = Frame(bgr=np.zeros((48, 64, 3), np.uint8), depth_m=np.ones((48, 64)))


def wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cond()


# ---------------------------------------------------------------- teleop
def test_no_commands_during_warmup(fake_hand):
    p = make(fake_hand)
    t = feed(p, synthetic_hand(), n=p.WARMUP_FRAMES - 1)
    assert p.control_step(t) is None
    t = feed(p, synthetic_hand(), n=1, t0=t + 0.03)
    assert p.control_step(t) is not None


def test_hand_lost_stops_commands_and_requires_new_warmup(fake_hand):
    p = make(fake_hand)
    t = feed(p, synthetic_hand(), n=p.WARMUP_FRAMES)
    assert p.control_step(t) is not None
    assert p.control_step(t + p.HAND_LOST_S + 0.1) is None      # hand gone
    t2 = feed(p, synthetic_hand(), n=1, t0=t + 2.0)              # hand back
    assert p.control_step(t2) is None                             # warming up again
    t2 = feed(p, synthetic_hand(), n=p.WARMUP_FRAMES, t0=t2 + 0.03)
    assert p.control_step(t2) is not None


def test_fist_holds_until_open(fake_hand):
    p = make(fake_hand)
    t = feed(p, synthetic_hand(), n=p.WARMUP_FRAMES)
    p.handle_key(ord("f"))
    assert p.control_step(t) == list(FIST_POSE)   # teleop no longer overrides it
    assert p.control_step(t) == list(FIST_POSE)
    p.handle_key(ord("o"))
    assert p.state == State.TELEOP
    assert fake_hand.sent("set_joint_positions")[-1][1] == list(OPEN_POSE)


def test_calibrate_key_applies_on_next_frame(fake_hand):
    p = make(fake_hand)
    p.handle_key(ord("c"))
    feed(p, synthetic_hand())
    assert p.retargeter.config.is_calibrated
    assert p._angles[0] == pytest.approx(0.0)


def test_quit_keys(fake_hand):
    p = make(fake_hand)
    assert p.handle_key(ord("q")) is False
    assert p.handle_key(27) is False
    assert p.handle_key(ord("x")) is True


# ---------------------------------------------------------------- grasp
def test_full_grasp_cycle(fake_hand):
    p = make(fake_hand)
    t = feed(p, synthetic_hand(), n=p.WARMUP_FRAMES)
    p.control_step(t)
    give_frame(p)

    assert p.trigger_grasp()
    wait_for(lambda: p.planner.requests)
    assert p.detector.seen[0][1] is not None        # depth reached the detector
    assert p.control_step() is None                   # hand holds still while planning

    p.planner.respond("power", "light")
    assert p.state == State.EXECUTING
    assert fake_hand.sent("set_speed")[-1][2] == int(0.3 * p.bridge.normal_speed)

    p.grasp_duration_s = 0.0
    p.control_step()
    assert p.state == State.HOLDING
    assert p.control_step() == list(GRASP_POSES["power"].angles)

    p.handle_key(ord("g"))                            # release
    assert p.state == State.TELEOP
    assert fake_hand.sent("set_speed")[-1][2] == p.bridge.normal_speed


def test_no_detection_returns_to_teleop(fake_hand):
    p = make(fake_hand, detections=())
    give_frame(p)
    assert p.trigger_grasp()
    wait_for(lambda: p.state == State.TELEOP)


def test_planner_rejection_does_not_strand_pipeline(fake_hand):
    p = make(fake_hand, accept=False)
    give_frame(p)
    p.trigger_grasp()
    wait_for(lambda: p.planner.requests)
    wait_for(lambda: p.state == State.TELEOP)


def test_late_plan_after_cancel_is_ignored(fake_hand):
    p = make(fake_hand)
    give_frame(p)
    p.trigger_grasp()
    wait_for(lambda: p.planner.requests)
    p.handle_key(ord("o"))
    p.planner.respond()
    assert p.state == State.TELEOP


def test_grasp_needs_a_frame(fake_hand):
    assert make(fake_hand).trigger_grasp() is False


def test_grasp_blend_starts_from_last_command(fake_hand):
    p = make(fake_hand)
    p.handle_key(ord("f"))
    p.control_step()
    give_frame(p)
    p.trigger_grasp()
    wait_for(lambda: p.planner.requests)
    p.planner.respond("pinch")
    p.grasp_duration_s = 10.0
    first = p.control_step(p._grasp_t0)
    assert first == pytest.approx(list(FIST_POSE))
