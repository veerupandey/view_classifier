from view_classifier.labels import POSE_CLASSES, angle_to_pose_idx, sincos_to_deg, wrap_deg


def test_front_wraps_around_zero():
    assert POSE_CLASSES[angle_to_pose_idx(0)] == "front"
    assert POSE_CLASSES[angle_to_pose_idx(359)] == "front"
    assert POSE_CLASSES[angle_to_pose_idx(22.4)] == "front"


def test_cardinal_and_diagonal_bins():
    assert POSE_CLASSES[angle_to_pose_idx(45)] == "front_right"
    assert POSE_CLASSES[angle_to_pose_idx(90)] == "right"
    assert POSE_CLASSES[angle_to_pose_idx(180)] == "rear"
    assert POSE_CLASSES[angle_to_pose_idx(270)] == "left"
    assert POSE_CLASSES[angle_to_pose_idx(315)] == "front_left"


def test_circular_wrap_and_sincos_roundtrip():
    assert wrap_deg(-10) == 350
    assert abs(sincos_to_deg(0.0, 1.0) - 0.0) < 1e-6
    assert abs(sincos_to_deg(1.0, 0.0) - 90.0) < 1e-6
