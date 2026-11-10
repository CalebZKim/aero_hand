import numpy as np
import pytest
from conftest import mocap_from, synthetic_hand

from aero_webcam_teleop.joints import JOINT_MAX_DEG, JOINT_NAMES
from aero_webcam_teleop.retargeting import (
    DEFAULT_JOINT_SCALE,
    RetargetingConfig,
    Retargeter,
    raw_joint_angles,
)

J = {name: i for i, name in enumerate(JOINT_NAMES)}


def test_open_hand_reads_zero_flexion():
    angles = raw_joint_angles(synthetic_hand())
    for j in range(1, 16):
        assert angles[j] == pytest.approx(0.0, abs=1e-3), JOINT_NAMES[j]


def test_thumb_abduction_measured_in_palm_plane():
    angles = raw_joint_angles(synthetic_hand())
    assert 20 < angles[J["thumb_cmc_abd"]] < JOINT_MAX_DEG[0]


@pytest.mark.parametrize("joint, finger, slot", [
    ("index_mcp", "index", 0), ("index_pip", "index", 1), ("middle_dip", "middle", 2),
    ("ring_pip", "ring", 1), ("pinky_mcp", "pinky", 0),
])
def test_single_joint_flexion_matches_geometry(joint, finger, slot):
    flex = [0, 0, 0]
    flex[slot] = 40.0
    angles = raw_joint_angles(synthetic_hand({finger: flex}))
    j = J[joint]
    expected = 40.0 / 180.0 * JOINT_MAX_DEG[j] * DEFAULT_JOINT_SCALE[j]
    assert angles[j] == pytest.approx(expected, abs=0.5)
    # Neighbouring joints stay straight.
    for k in range(1, 16):
        if k != j:
            assert angles[k] == pytest.approx(0.0, abs=0.5), JOINT_NAMES[k]


def test_fist_saturates_at_joint_max():
    fist = {f: [120, 110, 100] for f in ("index", "middle", "ring", "pinky")}
    angles = raw_joint_angles(synthetic_hand(fist))
    for j in range(4, 16):
        assert angles[j] == pytest.approx(JOINT_MAX_DEG[j])


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        raw_joint_angles(np.zeros((20, 3)))


def test_calibration_rescales_and_keeps_full_range():
    # A user whose "open" hand still reads 20 deg on index MCP.
    slightly_bent = synthetic_hand({"index": [20 * 180 / 90 / 1.5, 0, 0]})
    r = Retargeter(smoothing_alpha=1.0)
    ref = r.calibrate(mocap_from(slightly_bent))
    assert ref[J["index_mcp"]] == pytest.approx(20.0, abs=0.5)

    assert r.retarget(mocap_from(slightly_bent))[J["index_mcp"]] == pytest.approx(0.0, abs=1e-6)
    fist = synthetic_hand({"index": [120, 120, 120]})
    assert r.retarget(mocap_from(fist))[J["index_mcp"]] == pytest.approx(90.0)


def test_recalibrating_does_not_compound():
    hand = synthetic_hand({"middle": [15, 10, 5]})
    r = Retargeter(smoothing_alpha=0.35)
    r.retarget(mocap_from(synthetic_hand({"middle": [80, 80, 80]})))  # pollute EMA history
    first = r.calibrate(mocap_from(hand))
    second = r.calibrate(mocap_from(hand))
    assert first == pytest.approx(second)


def test_ema_smoothing_and_reset():
    open_hand, fist = synthetic_hand(), synthetic_hand({"index": [60, 0, 0]})
    r = Retargeter(smoothing_alpha=0.5)
    r.retarget(mocap_from(open_hand))
    target = raw_joint_angles(fist)[J["index_mcp"]]
    assert r.retarget(mocap_from(fist))[J["index_mcp"]] == pytest.approx(target / 2)
    r.reset()
    assert r.retarget(mocap_from(fist))[J["index_mcp"]] == pytest.approx(target)


def test_config_reports_calibration_state():
    cfg = RetargetingConfig()
    assert not cfg.is_calibrated
    Retargeter(cfg).calibrate(mocap_from(synthetic_hand()))
    assert cfg.is_calibrated  # thumb abduction is non-zero on an open hand
