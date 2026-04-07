import torch

from stable_worldmodel.wm.lewm import SIGReg


def test_sigreg_forward_respects_input_device():
    sigreg = SIGReg(knots=5, num_proj=8)
    proj = torch.randn(3, 4, 6)

    out = sigreg(proj)

    assert out.ndim == 0
    assert out.device == proj.device
    assert torch.isfinite(out)
