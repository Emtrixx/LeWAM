import math
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from lewam.data import make_datasets
from lewam.data.windows import ActionNormalizer
from lewam.models import LeWAM, ModelConfig

from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .logging import RunLogger
from .runtime import initialize, move_batch, seed_everything


def build_model(cfg):
    model = LeWAM(
        ModelConfig(**cfg["model"]),
        weights=cfg["train"]["mode_weights"],
        sigreg=cfg["train"]["sigreg"],
        random_context=cfg["train"]["random_context"],
    )
    if cfg.get("init_from"):
        source = load_checkpoint(cfg["init_from"])
        model.encoder.load_state_dict(
            {
                k.removeprefix("encoder."): v
                for k, v in source["model"].items()
                if k.startswith("encoder.")
            }
        )
    if cfg["freeze_encoder"]:
        model.freeze_encoder()
    return model


def validate_config(cfg):
    ModelConfig(**cfg["model"])
    train = cfg["train"]
    if train["batch_size"] < 2 or train["epochs"] < 1:
        raise ValueError("Training requires global batch >= 2 and epochs >= 1")
    if train["lr"] <= 0 or not 0 <= train["warmup_fraction"] < 1:
        raise ValueError("Positive learning rate and warmup_fraction in [0,1) required")
    if min(train["log_every"], train["save_every"], train["validate_every"]) < 1:
        raise ValueError("Logging, saving, and validation intervals must be positive")
    if any(v < 0 for v in train["mode_weights"].values()):
        raise ValueError("Mode weights cannot be negative")
    if not any(train["mode_weights"].values()):
        raise ValueError("At least one training mode must be active")
    if set(train["mode_weights"]) != {"forward", "backward", "inverse", "policy"}:
        raise ValueError("Mode weights must name forward, backward, inverse, and policy")
    if train["max_steps"] is not None and train["max_steps"] < 1:
        raise ValueError("max_steps must be positive or null")
    if train["max_val_batches"] is not None and train["max_val_batches"] < 1:
        raise ValueError("max_val_batches must be positive or null")
    kind = cfg["model"]["kind"]
    if kind == "lewm" and (
        not train["mode_weights"]["forward"]
        or any(train["mode_weights"][m] for m in ("inverse", "backward", "policy"))
    ):
        raise ValueError("LeWM uses only its forward objective; select experiment=lewm")
    if kind == "policy" and (
        not train["mode_weights"]["policy"]
        or any(train["mode_weights"][m] for m in ("inverse", "backward", "forward"))
    ):
        raise ValueError("Policy-only uses only policy flow matching; select experiment=policy")
    if cfg["resume"] and cfg["init_from"]:
        raise ValueError("Choose resume or encoder initialization, not both")


def train(cfg):
    validate_config(cfg)
    runtime = initialize(cfg["hardware"], cfg["seed"])
    logger = None
    try:
        output = Path(cfg["output_dir"]).resolve()
        if runtime.primary:
            output.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(cfg), output / "config.yaml")
        runtime.barrier()
        resume = load_checkpoint(cfg["resume"]) if cfg["resume"] else None
        if resume and resume["config"]["model"] != cfg["model"]:
            raise ValueError("Resume model architecture must match the checkpoint")
        normalizer = ActionNormalizer.from_dict(resume["normalizer"]) if resume else None
        # Preparation has an interprocess file lock; rank zero prepares before other ranks read.
        if not runtime.primary:
            runtime.barrier()
        train_set, val_set = make_datasets(
            cfg,
            output if runtime.primary else None,
            normalizer,
            resume["manifest"] if resume else None,
        )
        if runtime.primary:
            runtime.barrier()
        batch = cfg["train"]["batch_size"]
        if batch % runtime.world_size:
            raise ValueError("Global batch size must be divisible by WORLD_SIZE")
        local_batch = batch // runtime.world_size
        if len(train_set) < batch:
            raise ValueError("Training split is smaller than the global batch")
        sampler = DistributedSampler(
            train_set,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
            seed=cfg["seed"],
            drop_last=True,
        )
        loader_options = dict(
            num_workers=cfg["hardware"]["workers"], pin_memory=cfg["hardware"]["pin_memory"]
        )
        # A private generator keeps worker seeding from consuming the model's RNG stream.
        train_loader = DataLoader(
            train_set,
            batch_size=local_batch,
            sampler=sampler,
            drop_last=True,
            generator=torch.Generator().manual_seed(cfg["seed"]),
            **loader_options,
        )
        val_sampler = (
            DistributedSampler(
                val_set, num_replicas=runtime.world_size, rank=runtime.rank, shuffle=False
            )
            if runtime.world_size > 1
            else None
        )
        val_loader = DataLoader(
            val_set,
            batch_size=local_batch,
            sampler=val_sampler,
            generator=torch.Generator().manual_seed(0),
            **loader_options,
        )
        model = build_model(cfg).to(runtime.device)
        if resume:
            model.load_state_dict(resume["model"])
        if runtime.world_size > 1 and cfg["hardware"]["sync_batchnorm"]:
            if runtime.device.type != "cuda":
                raise ValueError("SyncBatchNorm requires CUDA; disable for CPU DDP tests")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        wrapped = (
            DistributedDataParallel(
                model,
                device_ids=[runtime.device.index] if runtime.device.type == "cuda" else None,
                find_unused_parameters=True,
            )
            if runtime.world_size > 1
            else model
        )
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        total = cfg["train"]["epochs"] * len(train_loader)
        warmup = max(1, int(total * cfg["train"]["warmup_fraction"]))

        def schedule(step):
            if step < warmup:
                return (step + 1) / warmup
            fraction = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.5 * (1 + math.cos(math.pi * fraction))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        epoch_start = batch_start = step = 0
        seed_everything(cfg["seed"] + runtime.rank)
        if resume:
            if len(resume["rng"]) != runtime.world_size:
                raise ValueError("Exact resume requires the original WORLD_SIZE")
            for key in (
                "batch_size",
                "epochs",
                "lr",
                "weight_decay",
                "warmup_fraction",
                "mode_weights",
                "sigreg",
                "random_context",
            ):
                if resume["config"]["train"][key] != cfg["train"][key]:
                    raise ValueError(
                        f"Resume training setting {key!r} changed; start a new run instead"
                    )
            for key in ("split_seed", "train_fraction"):
                if resume["config"]["data"][key] != cfg["data"][key]:
                    raise ValueError(f"Resume data setting {key!r} changed")
            if resume["config"]["distractor"] != cfg["distractor"]:
                raise ValueError("Resume distractor settings changed")
            for key in ("seed", "freeze_encoder"):
                if resume["config"][key] != cfg[key]:
                    raise ValueError(f"Resume setting {key!r} changed")
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            epoch_start, batch_start, step = resume["epoch"], resume["batch"], resume["step"]
            restore_rng(resume["rng"][runtime.rank])
        logger = RunLogger(output, cfg, runtime.primary)
        # Logger initialization must not perturb the resumed training random stream.
        if resume:
            restore_rng(resume["rng"][runtime.rank])
        else:
            seed_everything(cfg["seed"] + runtime.rank)
        if runtime.primary:
            logger.log(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_windows": len(train_set),
                    "val_windows": len(val_set),
                },
                step,
            )
        last_epoch, last_batch = epoch_start, batch_start
        limit = cfg["train"]["max_steps"]
        for epoch in range(epoch_start, cfg["train"]["epochs"]):
            if limit is not None and step >= limit:
                break
            sampler.set_epoch(epoch)
            model.train()
            for batch_index, raw in enumerate(train_loader):
                if epoch == epoch_start and batch_index < batch_start:
                    continue
                start = time.perf_counter()
                data = move_batch(raw, runtime.device)
                optimizer.zero_grad(set_to_none=True)
                with runtime.autocast():
                    losses = wrapped(data)
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                losses["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["train"]["grad_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                step += 1
                last_epoch, last_batch = epoch, batch_index + 1
                if step % cfg["train"]["log_every"] == 0:
                    metrics = {f"train/{k}": float(runtime.mean(v)) for k, v in losses.items()}
                    metrics.update(
                        {
                            "train/grad_norm": float(runtime.mean(grad_norm)),
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch,
                            "train/samples_per_second": batch / (time.perf_counter() - start),
                        }
                    )
                    if runtime.device.type == "cuda":
                        metrics["train/peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    logger.log(metrics, step)
                if limit is not None and step >= limit:
                    break
            complete_epoch = last_batch == len(train_loader)
            if (epoch + 1) % cfg["train"]["validate_every"] == 0:
                # Validation must not perturb training noise/context/SIGReg random streams.
                state = rng_state()
                seed_everything(100000 + cfg["seed"] + runtime.rank)
                model.eval()
                sums, count = {}, 0
                with torch.no_grad():
                    for val_index, raw in enumerate(val_loader):
                        if (
                            cfg["train"]["max_val_batches"] is not None
                            and val_index >= cfg["train"]["max_val_batches"]
                        ):
                            break
                        with runtime.autocast():
                            metrics = model(move_batch(raw, runtime.device))
                        n = len(raw["pixels"])
                        for k, v in metrics.items():
                            sums[k] = sums.get(k, 0.0) + float(runtime.mean(v)) * n
                        count += n
                logger.log({f"val/{k}": v / max(1, count) for k, v in sums.items()}, step)
                restore_rng(state)
            save_epoch, save_batch = (epoch + 1, 0) if complete_epoch else (last_epoch, last_batch)
            if (epoch + 1) % cfg["train"]["save_every"] == 0 or (
                limit is not None and step >= limit
            ):
                save_checkpoint(
                    output / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    cfg,
                    train_set,
                    save_epoch,
                    save_batch,
                    step,
                    runtime,
                )
            if limit is not None and step >= limit:
                break
        # Always leave a resumable checkpoint, including runs where save_every > epochs.
        save_epoch, save_batch = (
            (last_epoch + 1, 0) if last_batch == len(train_loader) else (last_epoch, last_batch)
        )
        save_checkpoint(
            output / "last.pt",
            model,
            optimizer,
            scheduler,
            cfg,
            train_set,
            save_epoch,
            save_batch,
            step,
            runtime,
        )
        return {"step": step, "checkpoint": str(output / "last.pt")}
    finally:
        if logger:
            logger.close()
        runtime.close()
