# Cluster launch and recovery

## Image and host

Build `docker build --platform linux/amd64 --target runtime -t lewam:gpu .` on
the GPU server or a build host. The Dockerfile pins base-image and uv digests,
and `uv.lock` pins Python packages for Mac and Linux. PyTorch's Linux x86_64
wheels contain CUDA 12.8, cuDNN, and NCCL; no host CUDA toolkit is required.
An appropriate NVIDIA driver and Container Toolkit are required. CUDA 12.8's
Linux driver floor is 570.26; use your site's supported newer driver where possible.

For an air-gapped site, transfer the built image with `docker save`/`docker load`
and prepare the full dataset on a connected machine first. Transfer the entire
`data` directory and mount it at `/data`; `/data/pusht` must retain the
prepared manifest and episode files. Training uses the prepared cache without
contacting Hugging Face. Use `logging.mode=offline` when W&B cannot be reached.

Keep a unique writable run directory and enough shared memory for DataLoader
workers. Use `--shm-size=8g` or the cluster's equivalent. The container runs as
root by default; sites may pass `--user UID:GID` with writable data/run mounts
and a writable home directory.

## Preflight before full training

```bash
docker run --rm --gpus all lewam:gpu python -c \
  'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name())'

docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/data" -v "$PWD/outputs:/runs" \
  lewam:gpu lewam-train experiment=pusht train.max_steps=2 \
  train.max_val_batches=1 logging.mode=offline \
  output_dir=/runs/preflight hydra.run.dir=/runs/preflight
```

The latter uses the real full-size model and global batch 128. Record peak CUDA
memory and throughput before removing the step cap for a new training run. If
memory is constrained, use `model.activation_checkpointing=true`; changing batch
size changes SIGReg's statistics and is an experimental deviation.

## Multi-GPU and multiple nodes

One node: use `torchrun --standalone --nproc-per-node=N -m lewam.cli` as in the
README. The global batch must divide WORLD_SIZE. CUDA ranks use NCCL and
SyncBatchNorm; SIGReg gathers latents with autograd and broadcasts common random
projection directions. CPU tests use Gloo with local BN and do not establish
cross-device floating-point identity.

`cluster/slurm.sbatch` launches one Docker container per allocated node and one
torchrun worker per GPU inside it. Set site partition/account directives and
export IMAGE, DATA_DIR, RUN_DIR, and LEWAM_GPUS_PER_NODE. For example:

```bash
export IMAGE=your-registry/lewam:your-tag
export DATA_DIR=/shared/lewam/data
export RUN_DIR=/shared/lewam/runs/pusht-seed0
export LEWAM_GPUS_PER_NODE=4
sbatch --nodes=2 --gpus-per-node=4 cluster/slurm.sbatch
```

All nodes need the image, shared paths, driver access, and reachable rendezvous
address/port. This template assumes Docker is permitted on compute nodes; a site
using Apptainer/Enroot should translate the image invocation using its supported
container integration. That site-specific launcher is not provisioned by this repo.

`cluster/kubernetes.yaml` is a single-GPU Job example. Replace the image tag,
PVC names, and W&B Secret reference before applying it. It requires the NVIDIA
device plugin and sufficient GPU quota. Neither template submits a job by itself.

## Resume

```bash
docker run --rm --gpus all --shm-size=8g \
  -v "$PWD/data:/data" -v "$PWD/outputs:/runs" -e WANDB_API_KEY \
  lewam:gpu lewam-train experiment=pusht resume=/runs/pusht/last.pt \
  output_dir=/runs/pusht-resumed hydra.run.dir=/runs/pusht-resumed
```

Retain original seed, objective, data split, global batch, epochs, and world size.
Use the same torchrun topology for a DDP checkpoint. Last checkpoints are atomic
and saved every configured epoch (default every epoch), plus at a step cap or
normal completion. Abrupt termination can lose work since the last saved epoch;
the prior checkpoint remains usable. Do not load untrusted pickle checkpoints.
