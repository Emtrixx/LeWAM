# LeWAM

A faithful, extensible reproduction of **LeWAM: A JEPA World Action Model with
Diffusion-Steering-Based MPC** ([arXiv:2610.12407v1](https://arxiv.org/abs/2610.12407v1)).

The reference paper is included at [docs/LeWAM_2610.12407v1.pdf](docs/LeWAM_2610.12407v1.pdf).

## Status

Repository initialized. Model implementation and development setup are next.

## Planned implementation

- Plain PyTorch, Hydra configuration management, and Weights & Biases logging.
- Public LeRobot PushT data with versioned downloads, local caching, and episode-based splits.
- Clean PushT, Lissajous dot/bar distractors, and generated moving video backgrounds.
- LeWAM, a policy-only flow-matching model, and a forward-only LeWM baseline.
- Modular datasets, environments, models, losses, planners, and evaluation.
- Optional stable-worldmodel integration behind an adapter.

## Development and training targets

- **Development:** MacBook M1 Pro with 32 GB unified memory; small MPS/FP32 smoke runs.
- **Full training:** H100 with 94 GB GPU memory; a separate paper-scale configuration.

Downloaded datasets, credentials, checkpoints, generated assets, and run outputs
are excluded from version control. Dependency lockfiles and experiment configs
will be tracked as implementation proceeds.
