from copy import deepcopy

import torch

from lewam.training.checkpoint import load_checkpoint
from lewam.training.trainer import train


def test_checkpoint_resume_matches_uninterrupted_training(config, tmp_path):
    config["train"]["max_steps"] = 4
    config["train"]["max_val_batches"] = 1
    config["train"]["log_every"] = 1
    full = train(deepcopy(config))
    interrupted = deepcopy(config)
    interrupted["output_dir"] = str(tmp_path / "interrupted")
    interrupted["train"]["max_steps"] = 2
    partial = train(interrupted)
    resumed = deepcopy(config)
    resumed["resume"] = partial["checkpoint"]
    resumed["output_dir"] = str(tmp_path / "resumed")
    result = train(resumed)
    a, b = load_checkpoint(full["checkpoint"]), load_checkpoint(result["checkpoint"])
    assert b["step"] == 4
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
