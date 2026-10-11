import torch

from loralei.train.objective import apply_flow_shift, sample_sigmas


def test_flow_shift_pushes_towards_high_noise():
    u = torch.linspace(0.05, 0.95, 9)
    shifted = apply_flow_shift(u, 3.0)
    assert torch.all(shifted >= u)
    assert torch.allclose(apply_flow_shift(u, 1.0), u)


def test_flow_shift_keeps_the_unit_interval():
    u = torch.rand(1000)
    for shift in (1.0, 2.2, 3.0, 6.0):
        shifted = apply_flow_shift(u, shift)
        assert shifted.min() >= 0.0 and shifted.max() <= 1.0


def test_sampled_sigmas_avoid_the_endpoints():
    for scheme in ("uniform", "shift", "logit_normal"):
        sigmas = sample_sigmas(256, scheme, 3.0, "none", 0.0, 1.0, 1.29, torch.device("cpu"))
        assert sigmas.shape == (256,)
        assert sigmas.min() > 0.0 and sigmas.max() < 1.0


def test_pack_unpack_roundtrip():
    from loralei.families import pack_2x2, unpack_2x2

    latents = torch.randn(2, 16, 8, 12)
    packed = pack_2x2(latents)
    assert packed.shape == (2, (8 // 2) * (12 // 2), 16 * 4)
    assert torch.equal(unpack_2x2(packed, 8, 12), latents)
