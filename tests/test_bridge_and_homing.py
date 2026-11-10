import threading

import pytest
from conftest import FakeHand

from aero_webcam_teleop import homing
from aero_webcam_teleop.bridge import NORMAL_SPEED, AeroHandBridge
from aero_webcam_teleop.joints import JOINT_MAX_DEG


def test_mock_mode_without_port():
    b = AeroHandBridge()
    assert b.is_mock
    assert b.send_joint_positions([10.0] * 16)
    b.close()


def test_connect_sets_normal_speed(fake_hand):
    AeroHandBridge(hand=fake_hand)
    speeds = fake_hand.sent("set_speed")
    assert len(speeds) == 7 and all(s[2] == NORMAL_SPEED for s in speeds)


def test_commands_are_clamped(fake_hand):
    b = AeroHandBridge(hand=fake_hand)
    b.send_joint_positions([-10.0] + [500.0] * 15)
    sent = fake_hand.sent("set_joint_positions")[-1][1]
    assert sent[0] == 0.0
    assert sent[1:] == list(JOINT_MAX_DEG[1:])


def test_wrong_length_rejected(fake_hand):
    with pytest.raises(ValueError):
        AeroHandBridge(hand=fake_hand).send_joint_positions([0.0] * 7)


def test_commands_suppressed_while_homing(fake_hand):
    gate, release = threading.Event(), threading.Event()

    def slow_homing():
        gate.set()
        release.wait(2.0)

    fake_hand.send_homing = slow_homing
    b = AeroHandBridge(hand=fake_hand)
    t = threading.Thread(target=b.home)
    t.start()
    assert gate.wait(2.0)
    assert b.is_homing
    assert b.send_joint_positions([0.0] * 16) is False
    release.set()
    t.join(2.0)
    assert not b.is_homing
    assert b.send_joint_positions([0.0] * 16) is True
    # speed restored after homing
    assert fake_hand.sent("set_speed")[-1][2] == NORMAL_SPEED


def test_close_opens_hand_and_releases_port(fake_hand):
    b = AeroHandBridge(hand=fake_hand)
    b.close()
    assert fake_hand.sent("set_joint_positions")[-1][1] == [0.0] * 16
    assert fake_hand.closed and b.is_mock


# ---------------------------------------------------------------- homing
class ScriptedHand(FakeHand):
    """Current on the torqued actuator follows a script (one value per poll)."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self.active = None

    def ctrl_torque(self, t):
        super().ctrl_torque(t)
        self.active = next((i for i, v in enumerate(t) if v), None)

    def get_actuator_currents(self):
        cur = [0.0] * 7
        if self.active is not None and self.script:
            cur[self.active] = self.script.pop(0)
        return cur

    def set_actuations(self, a):  # position moves arrive instantly
        super().set_actuations(a)


@pytest.fixture(autouse=True)
def fast_homing(monkeypatch):
    monkeypatch.setattr(homing, "POLL_S", 0.0)
    monkeypatch.setattr(homing.time, "sleep", lambda s: None)


def test_current_homing_contact_and_backoff():
    hand = ScriptedHand([50, 200, 300, 450, 520])  # middle threshold 500
    res = homing.safe_home(hand, actuators=[4], timeout_s=1.0)
    assert res[4] == homing.Result.CONTACT
    assert hand.actuations[4] == pytest.approx(homing.BACKOFF_DEG)
    assert hand.sent("ctrl_torque")[-1][1] == [0] * 7


def test_inverted_sensor_uses_sign():
    hand = ScriptedHand([-100, -200, -300, -420])  # thumb tendon, threshold 400, inverted
    assert homing.safe_home(hand, actuators=[2], timeout_s=1.0)[2] == homing.Result.CONTACT


def test_snap_detection_stops_everything():
    hand = ScriptedHand([200, 200, 200, 10])  # loaded, then current collapses
    res = homing.safe_home(hand, actuators=[5, 6], timeout_s=1.0)
    assert res[5] == homing.Result.SNAP
    assert res[6] == homing.Result.SNAP  # remaining actuators skipped
    assert not hand.sent("set_joint_positions")  # no open-palm command after a snap


def test_position_homing_reaches_target():
    hand = ScriptedHand([])
    res = homing.safe_home(hand, actuators=[0, 3], timeout_s=1.0)
    assert res[0] == res[3] == homing.Result.POSITION
    assert hand.actuations[3] == 240.0
