"""Pinned official PH200 raw HDF5 -> native camera episode cache."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

CATALOG = json.loads(Path(__file__).with_name("robomimic-catalog.json").read_text())


def render_identity():
    base = Path(__file__).parents[1]
    files = (
        "data/robomimic.py",
        "data/robomimic-catalog.json",
        "environments/robomimic.py",
        "distractors/boxes.py",
    )
    return {
        "source": {p: hashlib.sha256((base / p).read_bytes()).hexdigest() for p in files},
        "runtime": {
            p: version(p)
            for p in ("robosuite", "mujoco", "robomimic", "h5py", "numpy", "pillow", "PyOpenGL")
        },
    }


def verified_source(cfg):
    if (cfg["repo_id"], cfg["revision"]) != (CATALOG["repo_id"], CATALOG["revision"]):
        raise ValueError("Use the pinned official PH dataset source")
    entry = next(e for e in CATALOG["files"] if e["task"] == cfg["task"])
    path = Path(cfg["root"]).expanduser().resolve() / entry["path"]
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Raw HDF5 identity mismatch: {path}")
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        meta = json.loads(data.attrs["env_args"])
        if len(data) != entry["episodes_expected"] or meta["env_name"] != entry["simulator_task"]:
            raise ValueError("Unexpected official episode count or task metadata")
        if meta["env_version"] != "1.5.0" or meta["env_kwargs"]["control_freq"] != 20:
            raise ValueError("Expected robosuite 1.5.0 / 20 Hz data")
    return path, entry, meta


def prepare_robomimic(cfg, destination):
    from lewam.environments.robomimic import RobomimicAdapter

    from .prepare import save_episode, write_json

    path, entry, meta = verified_source(cfg)
    count = cfg["max_episodes"] or entry["episodes_expected"]
    if not 2 <= count <= entry["episodes_expected"]:
        raise ValueError("Select between 2 and 200 complete source episodes")
    limit = cfg.get("max_frames")
    if limit is not None and limit < 16:
        raise ValueError("Bounded proof needs at least 16 frames per episode")
    env = None
    records = []
    try:
        with h5py.File(path, "r") as handle:
            for episode in range(count):
                demo = handle[f"data/demo_{episode}"]
                states, actions = demo["states"], demo["actions"]
                state_dim = states.shape[1]
                if len(states) != len(actions) or actions.shape[1] != entry["action_dim"]:
                    raise ValueError("Unaligned raw state/action arrays")
                length = len(actions) if limit is None else min(len(actions), limit)
                completed = destination / f"episode_{episode:06d}.json"
                if completed.exists():
                    record = json.loads(completed.read_text())
                    for key in ("pixels", "actions", "state"):
                        if len(np.load(destination / record[key], mmap_mode="r")) != length:
                            raise ValueError("Incomplete rendered episode receipt")
                    records.append(record)
                    continue
                if env is None:
                    env = RobomimicAdapter({"source": cfg}, cfg["render"], None)
                env.restore_model(demo.attrs["model_file"], demo.attrs.get("ep_meta"))
                # Render s[t] directly, paired with a[t]; never integrate a drifted replay.
                frames = [env.replay_frame(states[t], episode, t) for t in range(length)]
                record = save_episode(
                    destination, episode, np.stack(frames), actions[:length], states[:length]
                )
                record["source_length"] = len(actions)
                record["model_sha256"] = hashlib.sha256(
                    demo.attrs["model_file"].encode()
                ).hexdigest()
                # A timed-out preparation retains complete episodes; unfinished ones
                # have no receipt and are rendered again on the next bounded call.
                write_json(completed, record)
                records.append(record)
    finally:
        if env is not None:
            env.close()
    return {
        "format": 1,
        "kind": "robomimic",
        "repo_id": cfg["repo_id"],
        "revision": cfg["revision"],
        "task": cfg["task"],
        "raw_sha256": entry["sha256"],
        "fps": 20,
        "camera": entry["paper_camera_semantics"],
        "native_resolution": entry["native_resolution"],
        "composition": "vertical",
        "action_dim": entry["action_dim"],
        "env_args": meta,
        "render": cfg["render"],
        "render_identity": render_identity(),
        "qualification": "adapted-physical-boxes; paper-equivalence-unqualified",
        "alignment": "pixels[t] rendered from states[t], paired with actions[t]",
        "partial": count != 200 or limit is not None,
        "state_names": [f"sim_state_{i}" for i in range(state_dim)],
        "episodes": records,
    }
