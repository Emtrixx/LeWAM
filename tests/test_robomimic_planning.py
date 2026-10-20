"""Native action bounds must not promote planner actions under BF16 autocast."""

import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from lewam.data.windows import ActionNormalizer
from lewam.environments import make_environment
from lewam.models import LeWAM, ModelConfig
from lewam.planning import MPCPlanner


@pytest.mark.integration
@pytest.mark.parametrize(("task", "dimensions"), [("lift", 7), ("transport", 14)])
def test_native_factory_full_ds_cem_cpu_bf16(config, task, dimensions):
    raw_root = os.environ.get("LEWAM_ROBOMIMIC_DIR")
    if not raw_root:
        pytest.skip("Set LEWAM_ROBOMIMIC_DIR to pinned official PH data")
    pytest.importorskip("h5py")
    with initialize_config_module(config_module="lewam.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=" + task,
                "hardware=cpu",
                "data.root=" + raw_root,
                "output_dir=" + config["output_dir"],
            ],
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    torch.manual_seed(17)
    model = LeWAM(ModelConfig(**dict(config["model"], action_dim=dimensions))).eval()
    env = make_environment(cfg["environment"], cfg["distractor"], model.cfg.image_size)
    try:
        native_low, native_high = env.env.action_spec
        assert native_low.dtype == native_high.dtype == np.float64
        assert env.action_low.dtype == env.action_high.dtype == np.float32
        np.testing.assert_array_equal(env.action_low, native_low)
        np.testing.assert_array_equal(env.action_high, native_high)
        np.testing.assert_array_equal(env.action_low, -np.ones(dimensions))
        np.testing.assert_array_equal(env.action_high, np.ones(dimensions))
        chunk = model.chunk_dim
        normalizer = ActionNormalizer(
            np.linspace(-0.2, 0.2, chunk, dtype=np.float32),
            np.linspace(0.7, 1.3, chunk, dtype=np.float32),
        )
        planner = MPCPlanner(
            model, cfg["planner"], normalizer, env.action_low, env.action_high, seed=19
        )
        frame = torch.from_numpy(env.reset(19)).permute(2, 0, 1).float() / 255
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            context = model.encode(frame[None, None])
            assert context.dtype == torch.bfloat16
            bounded = planner.bound_actions(torch.full((1, chunk), 50.0, dtype=torch.bfloat16))
            assert bounded.dtype == torch.float32
            action, metrics = planner.plan(
                context, context.new_zeros(1, 0, chunk), context[:, -1].float()
            )
        assert cfg["planner"]["horizon"] == 5 and cfg["planner"]["elites"] == 8
        assert model.cfg.flow_steps == 8 and metrics["evaluations"] == 64 * 3
        assert np.isfinite(metrics["cost"])
        assert action.shape == (1, 5 * dimensions) and action.dtype == torch.float32
        assert torch.isfinite(action).all()
        primitive = normalizer.denormalize(action.numpy()).reshape(5, dimensions)
        assert primitive.min() >= -1 - 1e-6 and primitive.max() <= 1 + 1e-6
        _, done, info = env.step(np.clip(primitive[0], env.action_low, env.action_high))
        assert isinstance(done, bool) and isinstance(info["is_success"], bool)
    finally:
        env.close()
        assert env.env is None
