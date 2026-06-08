def test_jitter_bbox_stays_centered():
    from view_classifier.transforms_cfv import jitter_bbox

    x1, y1, x2, y2 = jitter_bbox((100, 100, 200, 200), scale_range=(1.0, 1.0))
    assert (x1, y1, x2, y2) == (100, 100, 200, 200)


def test_circular_angle_loss_zero_when_perfect():
    import torch
    from view_classifier.model import angle_regression_loss

    sincos = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    angle = torch.tensor([0.0, 90.0])
    loss = angle_regression_loss(sincos, sincos, angle, kind="circular")
    assert float(loss) < 1e-5
