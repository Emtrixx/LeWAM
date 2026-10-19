from types import MethodType

import pytest
import torch

from lewam.losses import SIGReg
from lewam.models import LeWAM, ModelConfig
from lewam.training.trainer import build_model


@pytest.mark.parametrize("mode,masked", [("forward", -1), ("backward", 0), ("policy", -1)])
def test_masked_frame_cannot_leak(config, mode, masked):
    m = build_model(config).eval()
    z, actions = torch.randn(2, 4, 64), torch.randn(2, 3, 10)
    original = m.tokens(z, actions, mode)
    z[:, masked] = 10000
    torch.testing.assert_close(original, m.tokens(z, actions, mode))


def test_policy_hides_past_actions_but_inverse_does_not(config):
    m = build_model(config).eval()
    z, actions = torch.randn(2, 4, 64), torch.randn(2, 3, 10)
    policy = m.tokens(z, actions, "policy")
    inverse = m.tokens(z, actions, "inverse")
    actions[:, :-1] += 100
    torch.testing.assert_close(policy, m.tokens(z, actions, "policy"))
    assert not torch.equal(inverse, m.tokens(z, actions, "inverse"))


def test_live_encoder_target_has_gradient(config):
    config["train"]["mode_weights"] = {
        "forward": 1.0,
        "backward": 0.0,
        "inverse": 0.0,
        "policy": 0.0,
    }
    config["train"]["sigreg"]["weight"] = 0.0
    m = build_model(config)
    m.random_context = False
    z = torch.randn(2, 4, 64, requires_grad=True)
    m.encode = lambda _: z
    m({"pixels": None, "actions": torch.randn(2, 3, 10)})["loss"].backward()
    assert z.grad[:, -1].abs().sum() > 0  # Masked input receives gradients through the live target.


def test_all_modes_reach_encoder_and_action_head(config):
    m = build_model(config)
    loss = m({"pixels": torch.rand(4, 4, 3, 56, 56), "actions": torch.randn(4, 3, 10)})
    assert {
        "forward",
        "backward",
        "inverse_flow",
        "policy_flow",
        "policy_latent",
        "sigreg",
    } <= loss.keys()
    loss["loss"].backward()
    assert m.encoder.patch.weight.grad.abs().sum() > 0
    assert m.action_head.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(v) for v in loss.values())
    expected = (
        2 * loss["forward"]
        + loss["backward"]
        + loss["inverse_flow"]
        + loss["policy_flow"]
        + loss["policy_latent"]
        + 0.36 * loss["sigreg"]
    )
    torch.testing.assert_close(loss["loss"], expected)


def test_euler_runs_from_noise_to_data(config):
    m = build_model(config).eval()

    def constant_velocity(self, z, actions, mode, sigma):
        return z[:, -1], torch.full_like(actions[:, -1], 2.0)

    m.predict_mode = MethodType(constant_velocity, m)
    noise = torch.randn(2, 10)
    result = m.sample_actions(torch.zeros(2, 3, 64), noise, steps=8)
    torch.testing.assert_close(result, noise - 2)


def test_sampling_deterministic_for_fixed_noise(config):
    m = build_model(config).eval()
    context, noise = torch.randn(2, 3, 64), torch.randn(2, 10)
    torch.testing.assert_close(m.sample_actions(context, noise), m.sample_actions(context, noise))


def test_frozen_encoder_keeps_batchnorm_frozen(config):
    m = build_model(config)
    m.freeze_encoder()
    m.train()
    assert not m.encoder.training
    assert not any(p.requires_grad for p in m.encoder.parameters())


def test_sigreg_penalizes_collapse_and_has_gradients():
    regularizer = SIGReg(projections=64)
    torch.manual_seed(7)
    gaussian = torch.randn(512, 4, 32, requires_grad=True)
    good = regularizer(gaussian)
    bad = regularizer(torch.zeros_like(gaussian))
    assert good < bad
    good.backward()
    assert gaussian.grad.abs().sum() > 0


def test_paper_parameter_budget():
    m = LeWAM(ModelConfig())
    assert 40_000_000 < sum(p.numel() for p in m.parameters()) < 42_000_000
    assert m.blocks[0].attention.qkv.out_features == 3 * 16 * 64


def test_lewm_is_causal_and_has_no_policy(config):
    config["model"]["kind"] = "lewm"
    m = build_model(config).eval()
    z, actions = torch.randn(2, 3, 64), torch.randn(2, 3, 10)
    before = m.lewm_predict(z, actions)
    z[:, -1] += 100
    after = m.lewm_predict(z, actions)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    with pytest.raises(ValueError, match="no policy"):
        m.sample_actions(z)


def test_policy_only_has_no_latent_objective(config):
    config["model"]["kind"] = "policy"
    config["train"]["mode_weights"] = {
        "forward": 0.0,
        "backward": 0.0,
        "inverse": 0.0,
        "policy": 1.0,
    }
    config["train"]["sigreg"]["weight"] = 0.0
    m = build_model(config)
    losses = m({"pixels": torch.rand(4, 4, 3, 56, 56), "actions": torch.randn(4, 3, 10)})
    assert "policy_latent" not in losses and "forward" not in losses
    with pytest.raises(ValueError, match="no trained dynamics"):
        m.predict_next(torch.rand(4, 3, 64), torch.rand(4, 2, 10), torch.rand(4, 10))


def test_bfloat16_autocast_keeps_regularizer_stable(config):
    m = build_model(config)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = m({"pixels": torch.rand(4, 4, 3, 56, 56), "actions": torch.randn(4, 3, 10)})
    assert losses["sigreg"].dtype == torch.float32
    losses["loss"].backward()
    assert torch.isfinite(m.encoder.patch.weight.grad).all()


def test_fixed_batch_can_be_overfit(config):
    config["train"]["random_context"] = False
    m = build_model(config)
    batch = {"pixels": torch.rand(4, 4, 3, 56, 56), "actions": torch.randn(4, 3, 10)}
    optimizer = torch.optim.AdamW(m.parameters(), lr=0.001)
    values = []
    for _ in range(30):
        torch.manual_seed(1729)  # Fix corruption/time/projections to isolate learnability.
        optimizer.zero_grad()
        loss = m(batch)["loss"]
        values.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert sum(values[-5:]) < sum(values[:5]) * 0.6
