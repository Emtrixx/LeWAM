# LeWAM — unofficial implementation

An **unofficial, independently developed implementation** of
**LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC**
([arXiv:2610.12407v1](https://arxiv.org/abs/2610.12407v1)).

Credit for the LeWAM method and original research belongs to **Shashank Hegde,
Alexander Popov, Elie Aljalbout, and Nikolai Smolyanskiy**. This repository is
independently maintained; implementation choices, extensions and reproduction
results are documented separately from the original paper.

The reference paper is included at [docs/LeWAM_2610.12407v1.pdf](docs/LeWAM_2610.12407v1.pdf).

The completed five-seed PushT study reached **73.21% policy / 73.53% DS-CEM**
mean maximum raw coverage. See [research results and limitations](docs/research-results.md)
for the fixed evaluation contract, all inference arms and paper-comparison caveats.

## Start on the Mac

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. From the repository root:

```bash
uv sync --frozen
uv run lewam-prepare
uv run lewam-train output_dir=outputs/mac-smoke
uv run lewam-eval eval.checkpoint=outputs/mac-smoke/last.pt output_dir=outputs/mac-policy
uv run lewam-eval eval.checkpoint=outputs/mac-smoke/last.pt eval.method=planner \
  output_dir=outputs/mac-planner
```

The default `experiment=smoke` uses six genuine PushT episodes, a 643K-parameter
model, 56px images, batch 4, and 12 optimizer updates. `hardware=mac` uses native
MPS/FP32; select `hardware=cpu` explicitly if Metal is unavailable. Both run the same
four-mode objective, flow sampler, and planners as the full configuration.

W&B defaults to **offline** and also writes local `metrics.jsonl`. Use
`logging.mode=disabled` for no W&B process, or `logging.mode=online` with your own
`WANDB_API_KEY`. Training logs each loss, latent standard deviation, gradients,
learning rate, throughput, and CUDA peak memory. Evaluation saves metrics and MP4s.

For an entirely offline, procedural plumbing check (not a PushT benchmark):

```bash
uv run lewam-train data=synthetic hardware=cpu logging.mode=disabled
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## Full-size training

```bash
uv run lewam-check experiment=pusht hardware=cpu
uv run lewam-prepare experiment=pusht
uv run lewam-train experiment=pusht logging.name=pusht-lewam
```

`experiment=pusht` selects the 41.25M-parameter model and H100 hardware profile:
224px images, ViT-tiny/14, 384-D latents, an 8-layer bidirectional transformer,
all four modes, SIGReg, batch 128, BF16, AdamW, warmup/cosine decay, and 300 epochs.
It uses all 206 demonstrations with a fixed 90/10 **episode** split. The public
dataset revision is pinned; normalization is fitted only to training chunks.

The paper's fully enumerated optimization table is for robomimic. Applying its
settings to PushT and all other unresolved details is recorded in
[reproduction.md](docs/reproduction.md). This implementation has not been trained
for the paper's full schedule, and smoke results are not benchmark claims.

## Docker and the cluster

The same Dockerfile supports ARM64 CPU development and AMD64 CUDA training.
The GPU image uses locked PyTorch 2.8/CUDA 12.8 wheels; the host supplies the driver
through NVIDIA Container Toolkit. Build the **AMD64** image for an H100 server:

```bash
docker build --platform linux/amd64 --target runtime -t lewam:gpu .
mkdir -p data outputs
docker run --rm --gpus all --shm-size=8g \
  -e WANDB_API_KEY \
  -v "$PWD/data:/data" -v "$PWD/outputs:/runs" \
  lewam:gpu lewam-train experiment=pusht \
  output_dir=/runs/pusht hydra.run.dir=/runs/pusht
```

Use a fresh run directory per job. Before a long run, execute the same command
with `train.max_steps=2 train.max_val_batches=1` to verify device access, BF16,
the full batch, and your site's data/storage setup. Memory and throughput on an
actual H100 still require this server-side preflight.

For multiple GPUs on one node, replace the command after `lewam:gpu` with:

```bash
torchrun --standalone --nproc-per-node=4 -m lewam.cli \
  experiment=pusht output_dir=/runs/pusht-ddp hydra.run.dir=/runs/pusht-ddp
```

Batch size is **global**: batch 128 on four GPUs means 32 windows per rank.
CUDA DDP uses SyncBatchNorm and differentiable global-batch SIGReg. There is no
gradient accumulation masquerading as an equivalent SIGReg batch. Specify the
same `output_dir` and `hydra.run.dir` on every rank.

[cluster.md](docs/cluster.md) covers Slurm, Kubernetes, image transfer, and resume.
[cluster/slurm.sbatch](cluster/slurm.sbatch),
[cluster/kubernetes.yaml](cluster/kubernetes.yaml), and
[docker/compose.yaml](docker/compose.yaml) are editable deployment examples.

## Experiments and baselines

Hydra composes model, data, distractor, environment, planner, and hardware groups.

Lift, Can, Square, ToolHang and Transport are also available through
`experiment=lift`, `can`, `square`, `tool_hang` and `transport`. Use the separate
pinned simulator runtime and official PH200 datasets described in
[the Robomimic setup](docs/robomimic.md). The adapters have bounded real-data CPU
validation; their GPU experiments and paper-equivalent qualification remain open.
The configs ship **inside the Python package** at `src/lewam/configs`, so installed
wheels and Docker images use the same definitions.

```bash
# Same full-size architecture/schedule, three visual conditions and five seeds.
uv run lewam-train -m experiment=pusht distractor=clean,lissajous,video seed=0,1,2,3,4

# Policy-only flow matching (no dynamics loss or SIGReg).
uv run lewam-train experiment=policy hardware=h100

# Forward-only, causal, action-conditioned LeWM baseline.
uv run lewam-train experiment=lewm hardware=h100

# Optional frozen-LeWM-encoder policy control.
uv run lewam-train experiment=policy hardware=h100 \
  init_from=outputs/lewm/last.pt freeze_encoder=true

# Remove a mode or ablate the noise projection.
uv run lewam-train experiment=pusht train.mode_weights.backward=0
uv run lewam-eval eval.checkpoint=outputs/pusht/last.pt \
  eval.method=planner planner.project_noise=false
```

The `video` condition uses a reproducible procedural moving background by default.
Set `distractor.video_path=/path/to/clip.mp4` to use a supplied clip. Foreground
preservation currently assumes PushT's white table; another renderer needs a
foreground mask appropriate to that environment.

Available planner configs are `cem`, `mppi`, `cmaes` (action space) and `ds_cem`,
`ds_mppi`, `ds_cmaes` (noise space). Paper-sized evaluation budgets are 64 samples,
3 iterations, 5 chunks, 8 elites, and 8 Euler steps. The smoke experiment reduces
planning budgets and horizons as well as model size. For full evaluation use:

```bash
uv run lewam-eval experiment=pusht eval.checkpoint=outputs/pusht/last.pt \
  eval.method=planner planner=ds_cem logging.mode=offline
uv run lewam-probe experiment=pusht eval.checkpoint=outputs/pusht/last.pt \
  logging.mode=offline
```

Evaluation restores model/data/normalization from the checkpoint. Hardware,
logging, planner, output directory, and evaluation settings come from the command.
The training visual condition is retained unless explicitly changed through
`eval.visual_condition=clean|lissajous|video`. Goal retrieval uses designated
**training** demonstrations; simulator state never enters the model.

Optional train-only BatchNorm recalibration is disabled by default. To evaluate
it explicitly on a final checkpoint:

```bash
uv run lewam-eval experiment=pusht eval.checkpoint=outputs/pusht/last.pt \
  eval.method=planner planner=ds_cem eval.bn_calibration_batches=-1 \
  eval.bn_calibration_seed=0 output_dir=outputs/pusht-planner-calibrated
```

`-1` uses all complete training batches; a positive count limits a diagnostic run.
Calibration updates only in-memory BN running statistics, restores RNG state and
leaves the saved checkpoint unchanged. Evaluation records the calibration split
and order, maximum and terminal coverage, binary success and action-selection time.
The five-seed study found mixed calibration effects; it is not a recommended
universal performance improvement.

Ridge probes use episode-grouped cross-validation, held-out episodes, and a
shuffled-target control. The public LeRobot PushT data contains agent position,
not block pose, so full paper block-position/orientation probes are explicitly
unavailable until verified extra labels are supplied.

## Navigate and extend

- `models/lewam.py`: token masks, four losses, flow sampling, LeWM/policy controls.
- `models/encoder.py`, `models/transformer.py`: scratch ViT and explicit attention widths.
- `losses/sigreg.py`: FP32 anti-collapse statistics, including distributed gradients.
- `data/`: versioned LeRobot preparation, memory maps, windows, training-only statistics.
- `distractors/`: a shared deterministic renderer for offline frames and simulator observations.
- `environments/`: a small pixel-environment protocol and adapter registry.
- `planning/`: CEM, MPPI, CMA-ES and imagined-state noise decoding.
- `training/`: trainer, distributed runtime, provenance, logging, and full-state checkpoints.
- `evaluation/`: closed-loop rollouts, demonstration goals, videos, and ridge probes.
- `integrations/`: optional stable-worldmodel `Costable` contract.

All these directories are under `src/lewam`. See [architecture.md](docs/architecture.md)
for data shapes and extension points, and [validation.md](docs/validation.md) for
what has actually been exercised. OGBench, CarRacing-v3, ViZDoom, and MiniWorld
are future adapters, not installed dependencies or claimed supported benchmarks.


## Original paper and citation

If you use the LeWAM method in your research, please cite the original paper:

```bibtex
@misc{hegde2026lewam,
  title         = {LeWAM: A JEPA World Action Model with Diffusion-Steering-Based MPC},
  author        = {Shashank Hegde and Alexander Popov and Elie Aljalbout and Nikolai Smolyanskiy},
  year          = {2026},
  eprint        = {2610.12407},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2610.12407},
  url           = {https://arxiv.org/abs/2610.12407}
}
```

When referring to this code, identify it as this unofficial implementation and
link to this repository alongside the paper citation.
