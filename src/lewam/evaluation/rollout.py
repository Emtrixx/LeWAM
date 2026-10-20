import time
from pathlib import Path

import numpy as np
import torch

from lewam.data import make_datasets
from lewam.data.prepare import write_json
from lewam.environments import make_environment
from lewam.planning import MPCPlanner
from lewam.training.logging import RunLogger

from .calibration import recalibrate_batchnorm
from .common import load_evaluation
from .video import write_video


class GoalBank:
    """Nearest current latent -> frame 25 primitive steps later, in designated train demos."""

    @torch.no_grad()
    def __init__(self, model, dataset, device, count=1, lookahead=25):
        current, goals = [], []
        for episode in dataset.episode_ids[:count]:
            length = dataset.records[episode]["length"]
            encoded = []
            for start in range(0, length, 64):
                images = torch.stack(
                    [dataset.image(episode, i) for i in range(start, min(start + 64, length))]
                )
                encoded.append(model.encode(images[:, None].to(device))[:, 0])
            latent = torch.cat(encoded)
            current.append(latent)
            indices = (torch.arange(length, device=device) + lookahead).clamp_max(length - 1)
            goals.append(latent[indices])
        if not current:
            raise ValueError("At least one goal demonstration is required")
        self.current, self.goals = torch.cat(current), torch.cat(goals)

    def retrieve(self, latent):
        index = (self.current - latent[:, -1]).square().sum(-1).argmin()
        return self.goals[index : index + 1]


def pixel_tensor(image, device):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().to(device) / 255


@torch.no_grad()
def evaluate(cfg):
    loaded = load_evaluation(cfg)
    with loaded[-1].autocast():
        return _evaluate_loaded(cfg, *loaded)


def _evaluate_loaded(cfg, model, normalizer, saved, checkpoint, runtime):
    env = logger = None
    try:
        output = Path(saved["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        logger = RunLogger(output, saved)
        calibration = None
        calibration_batches = saved["eval"].get("bn_calibration_batches", 0)
        if calibration_batches:
            train, _ = make_datasets(
                saved, normalizer=normalizer, expected_manifest=checkpoint["manifest"]
            )
            calibration = recalibrate_batchnorm(
                model,
                train,
                runtime,
                saved["train"]["batch_size"],
                calibration_batches,
                saved["eval"].get("bn_calibration_seed", 0),
            )
            write_json(output / "bn-calibration.json", calibration)
        env = make_environment(saved["environment"], saved["distractor"], model.cfg.image_size)
        method = saved["eval"]["method"]
        if method not in {"policy", "planner"}:
            raise ValueError("eval.method must be policy or planner")
        if method == "policy" and not model.has_policy:
            raise ValueError("LeWM has no policy; evaluate with eval.method=planner planner=cem")
        planner = bank = None
        if method == "planner":
            # Planner is an inference choice, not a training checkpoint property.
            planner = MPCPlanner(
                model,
                cfg["planner"],
                normalizer,
                env.action_low,
                env.action_high,
                seed=saved["seed"],
            )
            train, _ = make_datasets(
                saved, normalizer=normalizer, expected_manifest=checkpoint["manifest"]
            )
            bank = GoalBank(
                model,
                train,
                runtime.device,
                saved["eval"]["goal_episodes"],
                saved["eval"]["goal_lookahead"],
            )
        results = []
        generator = torch.Generator().manual_seed(saved["seed"])
        for episode in range(saved["eval"]["episodes"]):
            seed = saved["seed"] + episode
            image = env.reset(seed)
            context = model.encode(pixel_tensor(image, runtime.device)[None, None])
            history = context.new_zeros(1, 0, model.chunk_dim)
            frames = [env.last_native.copy()]
            success = False
            coverage = terminal_coverage = total_reward = seconds = action_selection_seconds = 0.0
            steps = 0
            while steps < saved["eval"]["max_steps"]:
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                decision_start = time.perf_counter()
                if planner:
                    normalized, metrics = planner.plan(context, history, bank.retrieve(context))
                    seconds += metrics["seconds"]
                else:
                    noise = torch.randn(1, model.chunk_dim, generator=generator).to(runtime.device)
                    normalized = model.sample_actions(context, noise)
                action = normalizer.denormalize(normalized.float().cpu().numpy()[0]).reshape(
                    model.cfg.chunk_size, model.cfg.action_dim
                )
                action = np.clip(action, env.action_low, env.action_high).astype("float32")
                if runtime.device.type == "cuda":
                    torch.cuda.synchronize(runtime.device)
                action_selection_seconds += time.perf_counter() - decision_start
                done = False
                for primitive in action:
                    image, done, info = env.step(primitive)
                    steps += 1
                    success |= bool(info.get("is_success", False))
                    terminal_coverage = float(info.get("coverage", 0))
                    coverage = max(coverage, terminal_coverage)
                    total_reward += info["reward"]
                    if saved["eval"]["save_video"]:
                        frames.append(env.last_native.copy())
                    if done or steps >= saved["eval"]["max_steps"]:
                        break
                if done or steps >= saved["eval"]["max_steps"]:
                    break
                newest = model.encode(pixel_tensor(image, runtime.device)[None, None])
                context = torch.cat((context, newest), 1)[:, -model.cfg.context_frames :]
                executed = torch.from_numpy(normalizer.normalize(action.reshape(-1))).to(
                    runtime.device
                )
                history = torch.cat((history, executed[None, None]), 1)[
                    :, -(context.shape[1] - 1) :
                ]
                if context.shape[1] == 1:
                    history = history[:, :0]
            result = {
                "seed": seed,
                "success": success,
                "max_coverage": coverage,
                "terminal_coverage": terminal_coverage,
                "return": total_reward,
                "steps": steps,
                "planning_seconds": seconds,
                "action_selection_seconds": action_selection_seconds,
            }
            results.append(result)
            logger.log({f"eval/{k}": v for k, v in result.items()}, episode)
            if saved["eval"]["save_video"]:
                video = output / f"episode_{seed:04d}.mp4"
                write_video(video, frames, saved["eval"]["video_fps"])
                if episode == 0:
                    logger.video(video)
        summary = {
            "episodes": results,
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_max_coverage": float(np.mean([r["max_coverage"] for r in results])),
            "mean_terminal_coverage": float(np.mean([r["terminal_coverage"] for r in results])),
            "bn_calibration": calibration,
            "method": method,
            "visual_condition": saved["distractor"]["kind"],
            "planner": cfg["planner"] if planner else None,
        }
        write_json(output / "evaluation.json", summary)
        return summary
    finally:
        if env:
            env.close()
        if logger:
            logger.close()
        runtime.close()
