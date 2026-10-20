"""Optional train-only BatchNorm recalibration; never edits a saved checkpoint."""

import hashlib
import struct

import torch
from torch import nn
from torch.utils.data import DataLoader

from lewam.training.checkpoint import restore_rng, rng_state


def _fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def recalibrate_batchnorm(model, train_dataset, runtime, batch_size, batches=-1, seed=0):
    """Update only BN running buffers from deterministic, shuffled train windows.

    ``batches=-1`` consumes every complete batch once. Positive values declare a
    bounded diagnostic subset; exceeding the available batches is an error.
    All model parameters remain frozen, random streams are restored, and failures
    restore the original BN buffers. The caller must supply the training split.
    """
    if model.training:
        raise ValueError("Recalibration requires an evaluation-mode model")
    if batch_size < 2 or batches == 0 or batches < -1:
        raise ValueError("Require batch_size >= 2 and batches = -1 or positive")
    norms = {
        name: module
        for name, module in model.encoder.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    if not norms or any(not module.track_running_stats for module in norms.values()):
        raise ValueError("Encoder must have BatchNorm with tracked running statistics")
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    count = len(loader) if batches == -1 else batches
    if not 0 < count <= len(loader):
        raise ValueError("Calibration must consume between one and all complete train batches")
    original = {
        name: {k: v.clone() for k, v in norm.state_dict().items()} for name, norm in norms.items()
    }
    momentum = {name: norm.momentum for name, norm in norms.items()}
    random = rng_state()
    completed = examples = 0
    sample_order = hashlib.sha256()
    try:
        for norm in norms.values():
            norm.reset_running_stats()
            norm.momentum = None  # Cumulative mean of equally sized batch statistics.
            norm.train()
        for batch in loader:
            with runtime.autocast():
                model.encode(batch["pixels"].to(runtime.device))
            for episode, start in zip(
                batch["episode"].tolist(), batch["start"].tolist(), strict=True
            ):
                sample_order.update(struct.pack("<qq", episode, start))
            completed += 1
            examples += len(batch["pixels"])
            if completed == count:
                break
        for norm in norms.values():
            if (
                not torch.isfinite(norm.running_mean).all()
                or not torch.isfinite(norm.running_var).all()
            ):
                raise FloatingPointError("Non-finite recalibrated BatchNorm statistics")
        result = {
            "method": "train_only_batchnorm_cumulative_v1",
            "batches": completed,
            "windows": examples,
            "batch_size": batch_size,
            "seed": seed,
            "episode_ids": list(train_dataset.episode_ids),
            "sample_order_sha256": sample_order.hexdigest(),
            "sample_order_encoding": "consumed (episode,start) pairs packed as little-endian signed int64",
            "checkpoint_modified": False,
            "norms": {
                name: {
                    "before_sha256": _fingerprint(original[name]),
                    "after_sha256": _fingerprint(norm.state_dict()),
                }
                for name, norm in norms.items()
            },
        }
    except BaseException:
        for name, norm in norms.items():
            norm.load_state_dict(original[name])
        raise
    finally:
        for name, norm in norms.items():
            norm.momentum = momentum[name]
            norm.eval()
        restore_rng(random)
    return result
