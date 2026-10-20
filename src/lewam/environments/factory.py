"""Environment adapters are independent of model architecture and training datasets."""

import os
from typing import Protocol

import numpy as np
from PIL import Image

from lewam.distractors import Distractor


class PixelEnvironment(Protocol):
    action_low: np.ndarray
    action_high: np.ndarray

    def reset(self, seed: int) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict]: ...
    def close(self) -> None: ...


class GymAdapter:
    def __init__(self, cfg, distractor_cfg, image_size):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import gymnasium as gym

        kwargs = dict(cfg["kwargs"])
        if cfg["backend"] == "pusht":
            import gym_pusht  # noqa: F401  (registers the environment)

            kwargs.update(obs_type="pixels", observation_width=96, observation_height=96)
        self.env = gym.make(cfg["id"], render_mode="rgb_array", **kwargs)
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.distractor, self.size = Distractor(distractor_cfg), image_size
        self.episode, self.step_index = 0, 0
        self.last_native = None

    def pixels(self, observation):
        if isinstance(observation, dict):
            observation = observation.get("pixels")
        if not isinstance(observation, np.ndarray) or observation.ndim != 3:
            observation = self.env.render()
        native = self.distractor(observation, self.episode, self.step_index)
        self.last_native = native
        return np.asarray(
            Image.fromarray(native).resize((self.size, self.size), Image.Resampling.BILINEAR)
        ).copy()

    def reset(self, seed):
        self.episode, self.step_index = seed, 0
        obs, _ = self.env.reset(seed=seed)
        return self.pixels(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_index += 1
        info = dict(info, reward=float(reward))
        return self.pixels(obs), bool(terminated or truncated), info

    def close(self):
        self.env.close()


_FACTORIES = {"pusht": GymAdapter, "gymnasium": GymAdapter}


def _robomimic(cfg, distractor_cfg, image_size):
    from .robomimic import RobomimicAdapter

    return RobomimicAdapter(cfg, distractor_cfg, image_size)


_FACTORIES["robomimic"] = _robomimic


def register_environment(name, factory):
    """Register an adapter implementing PixelEnvironment; no trainer changes required."""
    if name in _FACTORIES:
        raise ValueError(f"Environment backend {name!r} is already registered")
    _FACTORIES[name] = factory


def make_environment(cfg, distractor_cfg, image_size):
    if cfg["backend"] not in _FACTORIES:
        raise ValueError(f"Unknown environment backend {cfg['backend']!r}; register its adapter")
    return _FACTORIES[cfg["backend"]](cfg, distractor_cfg, image_size)
