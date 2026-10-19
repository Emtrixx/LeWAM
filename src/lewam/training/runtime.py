import contextlib
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Runtime:
    device: torch.device
    rank: int
    world_size: int
    precision: str

    @property
    def primary(self):
        return self.rank == 0

    def autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else contextlib.nullcontext()
        )

    def mean(self, value):
        value = value.detach().clone()
        if self.world_size > 1:
            dist.all_reduce(value)
            value /= self.world_size
        return value

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()


def initialize(hardware, seed):
    torch.set_num_threads(hardware["threads"])
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = hardware["device"]
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but unavailable; use hardware=cpu or hardware=mac"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but unavailable; choose hardware=cpu explicitly")
    if hardware["precision"] not in {"fp32", "bf16"}:
        raise ValueError("Supported precisions are fp32 and bf16")
    if hardware["precision"] == "bf16" and not device.startswith("cuda"):
        raise ValueError("The bf16 profile is for CUDA; use fp32 on CPU/MPS")
    if (
        device.startswith("cuda")
        and hardware["precision"] == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("CUDA device does not support bfloat16")
    if world > 1:
        if device == "mps":
            raise ValueError("MPS distributed training is unsupported")
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    seed_everything(seed)
    return Runtime(torch.device(device), rank, world, hardware["precision"])


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
