import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from lewam.data import make_datasets
from lewam.evaluation.calibration import recalibrate_batchnorm
from lewam.evaluation.rollout import _evaluate_loaded
from lewam.training.checkpoint import rng_state
from lewam.training.runtime import Runtime
from lewam.training.trainer import build_model


def assert_rng_equal(a, b):
    assert a["python"] == b["python"]
    assert a["numpy"][0] == b["numpy"][0]
    np.testing.assert_array_equal(a["numpy"][1], b["numpy"][1])
    assert a["numpy"][2:] == b["numpy"][2:]
    torch.testing.assert_close(a["torch"], b["torch"], rtol=0, atol=0)


def test_real_encoder_calibration_changes_only_bn_buffers_and_preserves_rng(config):
    train, val = make_datasets(config)
    model = build_model(config).eval()
    clone = deepcopy(model)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    random_before = rng_state()
    runtime = Runtime(torch.device("cpu"), 0, 1, "fp32")
    report = recalibrate_batchnorm(model, train, runtime, batch_size=4, batches=2, seed=19)
    assert_rng_equal(random_before, rng_state())
    assert report["episode_ids"] == train.episode_ids
    assert not set(report["episode_ids"]) & set(val.episode_ids)
    assert report["batches"] == 2 and report["windows"] == 8
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, original[k])}
    assert changed and all(
        k.endswith(("running_mean", "running_var", "num_batches_tracked")) for k in changed
    )
    assert all(p.grad is None for p in model.parameters())
    assert not any(m.training for m in model.modules())
    random.seed(333)
    np.random.seed(555)
    torch.manual_seed(777)
    repeat = recalibrate_batchnorm(clone, train, runtime, batch_size=4, batches=2, seed=19)
    assert repeat == report
    for k, v in model.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[k], v, rtol=0, atol=0)


class PoisonedTrain(Dataset):
    episode_ids = [1]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "pixels": torch.full((4, 3, 56, 56), float("nan")),
            "episode": torch.tensor(1),
            "start": torch.tensor(index),
        }


def test_failed_calibration_restores_complete_model_and_rng(config):
    model = build_model(config).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    runtime = Runtime(torch.device("cpu"), 0, 1, "fp32")
    random_before = rng_state()
    with pytest.raises(FloatingPointError, match="Non-finite"):
        recalibrate_batchnorm(model, PoisonedTrain(), runtime, 4, batches=1)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)
    assert_rng_equal(random_before, rng_state())
    assert not any(m.training for m in model.modules())
    assert model.encoder.projector[1].momentum == 0.1


def test_calibration_rejects_incomplete_or_excessive_batch_requests(config):
    model = build_model(config).eval()
    runtime = Runtime(torch.device("cpu"), 0, 1, "fp32")
    with pytest.raises(ValueError, match="complete train batches"):
        recalibrate_batchnorm(model, PoisonedTrain(), runtime, 10)
    with pytest.raises(ValueError, match="complete train batches"):
        recalibrate_batchnorm(model, PoisonedTrain(), runtime, 4, batches=3)
    with pytest.raises(ValueError, match="evaluation-mode"):
        recalibrate_batchnorm(model.train(), PoisonedTrain(), runtime, 4)


class FixedEnvironment:
    action_low = np.zeros(2, dtype="float32")
    action_high = np.full(2, 512, dtype="float32")

    def reset(self, seed):
        self.steps = 0
        self.last_native = np.zeros((56, 56, 3), dtype="uint8")
        return self.last_native

    def step(self, action):
        self.steps += 1
        coverage = 0.6 if self.steps == 1 else 0.2
        return (
            self.last_native,
            self.steps == 2,
            {"reward": coverage, "coverage": coverage, "is_success": False},
        )

    def close(self):
        pass


def test_production_rollout_separates_max_and_terminal_and_leaves_default_bn_untouched(
    config, monkeypatch
):
    from lewam.evaluation import rollout

    train, _ = make_datasets(config)
    model = build_model(config).eval()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    monkeypatch.setattr(rollout, "make_environment", lambda *args: FixedEnvironment())
    config["eval"].update(episodes=2, max_steps=2, save_video=False)
    runtime = Runtime(torch.device("cpu"), 0, 1, "fp32")
    with torch.no_grad():
        result = _evaluate_loaded(config, model, train.normalizer, config, {}, runtime)
    assert result["mean_max_coverage"] == 0.6
    assert result["mean_terminal_coverage"] == 0.2
    assert result["success_rate"] == 0
    assert result["bn_calibration"] is None
    assert [r["seed"] for r in result["episodes"]] == [config["seed"], config["seed"] + 1]
    assert all(r["steps"] == 2 for r in result["episodes"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, original[k], rtol=0, atol=0)
