"""Each command accepts Hydra overrides, --cfg job, and multirun (-m)."""

import json

import hydra
from omegaconf import DictConfig, OmegaConf


def resolved(cfg):
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def train(cfg: DictConfig):
    from lewam.training.trainer import train as run

    print(json.dumps(run(resolved(cfg)), indent=2))


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def prepare(cfg: DictConfig):
    from lewam.data.prepare import prepare_dataset

    root, manifest = prepare_dataset(resolved(cfg)["data"])
    print(
        json.dumps(
            {
                "root": str(root),
                "revision": manifest["revision"],
                "episodes": len(manifest["episodes"]),
            },
            indent=2,
        )
    )


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def check(cfg: DictConfig):
    import torch

    from lewam.training.trainer import build_model, validate_config

    value = resolved(cfg)
    validate_config(value)
    torch.set_num_threads(value["hardware"]["threads"])
    model = build_model(value)
    report = {
        "model": value["model"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    print(json.dumps(report, indent=2))


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def evaluate(cfg: DictConfig):
    from lewam.evaluation.rollout import evaluate as run

    print(json.dumps(run(resolved(cfg)), indent=2))


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def probe(cfg: DictConfig):
    from lewam.evaluation.probes import probe as run

    print(json.dumps(run(resolved(cfg)), indent=2))


if __name__ == "__main__":
    train()
