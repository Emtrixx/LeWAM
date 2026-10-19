#!/usr/bin/env bash
#SBATCH --job-name=lewam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=lewam-%j.log

# Requires a Slurm site that permits Docker + NVIDIA Container Toolkit on compute nodes.
# Export IMAGE (an image available on every node), DATA_DIR, and RUN_DIR before sbatch.
set -euo pipefail
: "${IMAGE:?Set IMAGE to the built Docker image tag or registry digest}"
: "${DATA_DIR:?Set DATA_DIR to the shared dataset directory}"
: "${RUN_DIR:?Set RUN_DIR to a new shared run directory}"
mkdir -p "$DATA_DIR" "$RUN_DIR"
export LEWAM_MASTER_ADDR
LEWAM_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export LEWAM_MASTER_PORT=${LEWAM_MASTER_PORT:-29500}
export LEWAM_GPUS_PER_NODE=${LEWAM_GPUS_PER_NODE:-1}
export LEWAM_RDZV_ID=${SLURM_JOB_ID}

srun --kill-on-bad-exit=1 bash -c '
  set -euo pipefail
  gpu_request="\"device=${CUDA_VISIBLE_DEVICES:?Slurm must set CUDA_VISIBLE_DEVICES}\""
  exec docker run --rm --network host --ipc host --gpus "$gpu_request" \
    -e WANDB_API_KEY -e NCCL_DEBUG=WARN \
    -v "$DATA_DIR:/data" -v "$RUN_DIR:/runs" \
    "$IMAGE" torchrun \
      --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node="$LEWAM_GPUS_PER_NODE" \
      --node-rank="$SLURM_NODEID" --rdzv-id="$LEWAM_RDZV_ID" \
      --rdzv-backend=c10d --rdzv-endpoint="$LEWAM_MASTER_ADDR:$LEWAM_MASTER_PORT" \
      -m lewam.cli experiment=pusht hardware=h100 \
      output_dir=/runs hydra.run.dir=/runs
'
