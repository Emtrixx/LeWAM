"""Trusted local checkpoints include optimizer, schedule, data identity, and per-rank RNG."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path, model, optimizer, scheduler, cfg, dataset, epoch, batch, step, runtime):
    states = [None] * runtime.world_size
    state = rng_state()
    if runtime.world_size > 1:
        dist.all_gather_object(states, state)
    else:
        states[0] = state
    if runtime.primary:
        value = {
            "format": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "manifest": dataset.manifest,
            "normalizer": dataset.normalizer.as_dict(),
            "epoch": epoch,
            "batch": batch,
            "step": step,
            "rng": states,
        }
        path = Path(path)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(value, temporary)
        temporary.replace(path)
    runtime.barrier()


def load_checkpoint(path):
    value = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if value.get("format") != 1:
        raise ValueError("Unsupported checkpoint format")
    return value
