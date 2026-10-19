import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


@pytest.fixture
def config(tmp_path):
    with initialize_config_module(config_module="lewam.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["hardware=cpu", "data=synthetic", "logging.mode=disabled"],
        )
    cfg.output_dir = str(tmp_path / "run")
    cfg.data.root = str(tmp_path / "data")
    return OmegaConf.to_container(cfg, resolve=True)
