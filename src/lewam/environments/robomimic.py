"""Actual robosuite 1.5 raw-state replay and pixels-only online environment."""

import json
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version

import numpy as np
from PIL import Image

from lewam.data.robomimic import verified_source
from lewam.distractors.boxes import SimBoxes


@contextmanager
def simulator_rng(rng):
    # Robosuite 1.5 uses NumPy's legacy global RNG. Keep it outside model RNG order.
    previous = np.random.get_state()
    np.random.set_state(rng.get_state())
    try:
        yield
    finally:
        rng.set_state(np.random.get_state())
        np.random.set_state(previous)


class RobomimicAdapter:
    def __init__(self, cfg, distractor_cfg, image_size):
        import robosuite

        source = cfg["source"]
        if cfg.get("task", source["task"]) != source["task"]:
            raise ValueError("Environment and dataset tasks differ")
        _, self.spec, meta = verified_source(source)
        self.versions = {p: version(p) for p in ("robosuite", "mujoco", "robomimic", "h5py")}
        if self.versions["robosuite"] != meta["env_version"]:
            raise ValueError("Simulator version differs from raw metadata")
        if source["render"] != distractor_cfg:
            raise ValueError("Offline and online physical distractor recipes must match")
        self.boxes, self.size = SimBoxes(distractor_cfg), image_size
        if distractor_cfg["fps"] != meta["env_kwargs"]["control_freq"]:
            raise ValueError("Physical nuisance and simulator clocks differ")
        self.rng = np.random.RandomState(0)
        self.episode = self.step_index = 0
        self.last_native = None
        kwargs = deepcopy(meta["env_kwargs"])
        kwargs.update(
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            hard_reset=False,
            ignore_done=True,
        )
        with simulator_rng(self.rng):
            self.env = robosuite.make(meta["env_name"], **kwargs)
        try:
            self.env.has_offscreen_renderer = True
            self.restore_model(self.env.sim.model.get_xml())
            self.action_low, self.action_high = (v.copy() for v in self.env.action_spec)
            if self.action_low.shape != (self.spec["action_dim"],):
                raise ValueError("Native action dimension differs from source")
        except BaseException:
            self.close()
            raise

    def restore_model(self, xml, ep_meta=None):
        # The same model/XML restoration order as robomimic's reset_to.
        with simulator_rng(self.rng):
            self.env.set_ep_meta(json.loads(ep_meta) if ep_meta else {})
            self.env.reset()
            xml = self.env.edit_model_xml(xml)
            self.env.reset_from_xml_string(self.boxes.add_to_xml(xml))
            self.env.sim.reset()
        for camera in self.spec["paper_camera_semantics"]:
            self.env.sim.model.camera_name2id(camera)  # no fallback or guessed aliases

    def native_frame(self):
        self.boxes.move(self.env.sim, self.episode, self.step_index)
        n = self.spec["native_resolution"]
        frames = [
            self.env.sim.render(height=n, width=n, camera_name=camera)[::-1].copy()
            for camera in self.spec["paper_camera_semantics"]
        ]
        self.last_native = np.concatenate(frames, axis=0)
        return self.last_native.copy()

    def pixels(self):
        native = self.native_frame()
        if self.size is None:
            return native
        return np.asarray(
            Image.fromarray(native).resize((self.size, self.size), Image.Resampling.BILINEAR)
        ).copy()

    def replay_frame(self, state, episode, step):
        self.episode, self.step_index = int(episode), int(step)
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        return self.native_frame()

    def reset(self, seed):
        self.rng = np.random.RandomState(seed)
        self.episode, self.step_index = seed, 0
        with simulator_rng(self.rng):
            self.env.unset_ep_meta()
            self.env.reset()
        return self.pixels()

    def step(self, action):
        action = np.asarray(action, dtype="float32")
        if action.shape != self.action_low.shape or not np.isfinite(action).all():
            raise ValueError("Expected a finite primitive action of the native dimension")
        if np.any(action < self.action_low) or np.any(action > self.action_high):
            raise ValueError("Primitive action exceeds native bounds")
        with simulator_rng(self.rng):
            _, reward, _, _ = self.env.step(action)
        self.step_index += 1
        success = bool(self.env._check_success())
        done = success or self.step_index >= self.spec["paper_eval_max_steps"]
        return self.pixels(), done, {"reward": float(reward), "is_success": success}

    def close(self):
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None
