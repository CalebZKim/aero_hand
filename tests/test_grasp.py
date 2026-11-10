import time
from types import SimpleNamespace

import pytest

from aero_webcam_teleop.grasp import detector as det
from aero_webcam_teleop.grasp.planner import GraspPlanner, fallback_plan, parse_reply
from aero_webcam_teleop.grasp.poses import (
    FORCE_LEVELS, GRASP_POSES, GRASP_TYPES, OBJECT_DEFAULTS, default_for_object, interpolate,
)
from aero_webcam_teleop.joints import JOINT_MAX_DEG, NUM_JOINTS


# ---------------------------------------------------------------- poses
@pytest.mark.parametrize("name", sorted(GRASP_POSES))
def test_poses_are_within_joint_limits(name):
    angles = GRASP_POSES[name].angles
    assert len(angles) == NUM_JOINTS
    assert all(0.0 <= a <= hi for a, hi in zip(angles, JOINT_MAX_DEG))


def test_every_object_default_is_valid():
    for label, (grasp, force, _) in OBJECT_DEFAULTS.items():
        assert grasp in GRASP_TYPES, label
        assert force in FORCE_LEVELS, label


def test_every_detector_label_has_a_default():
    for label in set(det.YOLO_TO_OBJECT.values()) | det.DIRECT_LABELS:
        assert label in OBJECT_DEFAULTS, label


def test_default_lookup_normalises_and_falls_back():
    assert default_for_object("Squish Ball")[0] == "palmar"
    assert default_for_object("toaster") == OBJECT_DEFAULTS["unknown"]


def test_interpolate_clamps_t():
    assert interpolate([0, 0], [10, 20], 0.5) == [5, 10]
    assert interpolate([0], [10], 2.0) == [10]
    assert interpolate([0], [10], -1.0) == [0]


# ---------------------------------------------------------------- detector helpers
def test_label_mapping():
    assert det.map_label("bottle") == "cylinder"
    assert det.map_label("cube") == "cube"
    assert det.map_label("Squish Ball") == "squish_ball"
    assert det.map_label("giraffe") == "unknown"


def test_size_estimate_pinhole():
    # 90 deg HFOV, 640 px wide -> focal 320 px. 64 px at 0.5 m -> 10 cm.
    assert det.estimate_size_cm(64, 0.5, 640, 90.0) == pytest.approx(10.0)


# ---------------------------------------------------------------- planner
@pytest.mark.parametrize("text", [
    '{"grasp_type": "pinch", "force_level": "light", "reasoning": "small"}',
    '```json\n{"grasp_type": "Pinch", "force_level": "LIGHT", "reasoning": "small"}\n```',
    'Sure! {"grasp_type": "pinch", "force_level": "light", "reasoning": "small"} Hope that helps.',
])
def test_parse_reply_tolerates_wrapping(text):
    assert parse_reply(text) == {"grasp_type": "pinch", "force_level": "light", "reasoning": "small"}


@pytest.mark.parametrize("text", [
    "no json here",
    '{"grasp_type": "claw", "force_level": "light"}',
    '{"grasp_type": "pinch", "force_level": "crushing"}',
])
def test_parse_reply_rejects_bad_output(text):
    with pytest.raises(ValueError):
        parse_reply(text)


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.requests = reply, error, []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply)])


def test_plan_uses_llm_reply():
    client = FakeClient('{"grasp_type": "power", "force_level": "firm", "reasoning": "heavy"}')
    plan = GraspPlanner(client=client).plan("cylinder", size_cm=7.0, depth_m=0.3)
    assert (plan.grasp_type, plan.force_level, plan.source) == ("power", "firm", "llm")
    prompt = client.requests[0]["messages"][0]["content"]
    assert "7.0 cm" in prompt and "0.30 m" in prompt


def test_plan_falls_back_on_api_error():
    plan = GraspPlanner(client=FakeClient(error=RuntimeError("boom"))).plan("squish_ball")
    assert (plan.grasp_type, plan.source) == ("palmar", "fallback")


def test_fallback_without_client():
    assert fallback_plan("pyramid").grasp_type == "tripod"


def test_request_plan_reports_rejection_during_cooldown():
    results = []
    planner = GraspPlanner(on_plan=results.append, cooldown_s=60.0, client=FakeClient(
        '{"grasp_type": "tripod", "force_level": "medium", "reasoning": "ok"}'))
    assert planner.request_plan("cube") is True
    assert planner.request_plan("cube") is False
    deadline = time.monotonic() + 2.0
    while not results and time.monotonic() < deadline:
        time.sleep(0.01)
    assert results and results[0].grasp_type == "tripod"
    assert not planner.is_pending
