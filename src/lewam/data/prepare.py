"""Prepare immutable, memory-mapped episode caches from LeRobot v3 Parquet/MP4.

LeRobot is an interchange format here, not a dependency on a policy-training framework.
Each cache records the Hub SHA, camera, FPS, selected episodes, and source metadata.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cache_path(cfg):
    identity = {
        k: cfg[k]
        for k in (
            "kind",
            "repo_id",
            "revision",
            "camera",
            "max_episodes",
            "state_key",
            "action_key",
            "synthetic_episodes",
            "synthetic_length",
            "fps",
        )
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return Path(cfg["root"]).expanduser().resolve() / "prepared" / digest


def prepare_dataset(cfg):
    from filelock import FileLock

    destination = cache_path(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination / ".prepare.lock")):
        manifest_file = destination / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            for ep in manifest["episodes"]:
                for key in ("pixels", "actions", "state"):
                    if not (destination / ep[key]).exists():
                        raise FileNotFoundError(
                            f"Incomplete prepared cache: {destination / ep[key]}"
                        )
            return destination, manifest
        if cfg["kind"] not in _PREPARERS:
            raise ValueError(f"Unknown dataset kind {cfg['kind']!r}; register its preparer")
        manifest = _PREPARERS[cfg["kind"]](cfg, destination)
        write_json(manifest_file, manifest)
        return destination, manifest


def save_episode(destination, episode_id, pixels, actions, state):
    if len(pixels) != len(actions) or len(actions) != len(state):
        raise ValueError("Episode images, actions, and state must align exactly")
    record = {"id": int(episode_id), "length": len(actions)}
    for key, array in (("pixels", pixels), ("actions", actions), ("state", state)):
        filename = f"episode_{int(episode_id):06d}_{key}.npy"
        np.save(destination / filename, array, allow_pickle=False)
        record[key] = filename
    return record


def prepare_synthetic(cfg, destination):
    from PIL import Image, ImageDraw

    count = cfg["max_episodes"] or cfg["synthetic_episodes"]
    records = []
    for episode in range(count):
        rng = np.random.default_rng(episode)
        length = cfg["synthetic_length"]
        angles = np.arange(length) * 0.08 + rng.uniform(0, 6)
        agent = np.stack((48 + 25 * np.sin(angles), 48 + 25 * np.cos(angles * 0.7)), -1).astype(
            "float32"
        )
        frames = []
        for pos in agent:
            image = Image.new("RGB", (96, 96), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((40, 30, 65, 40), fill=(119, 136, 153))
            draw.rectangle((48, 30, 57, 65), fill=(119, 136, 153))
            x, y = pos
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(65, 105, 225))
            frames.append(np.asarray(image))
        action = np.concatenate((agent[1:], agent[-1:])) * (512 / 96)
        records.append(save_episode(destination, episode, np.stack(frames), action, agent))
    return {
        "format": 1,
        "kind": "synthetic",
        "revision": "procedural-v1",
        "fps": cfg["fps"],
        "camera": cfg["camera"],
        "action_dim": 2,
        "state_names": ["agent_x", "agent_y"],
        "episodes": records,
    }


def prepare_lerobot(cfg, destination):
    import av
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download

    root = Path(cfg["root"]).expanduser().resolve()
    local = root / "meta" / "info.json"
    if local.exists():
        source, revision = root, "local"
    else:
        token = None if cfg.get("authenticated", False) else False
        api = HfApi(token=token)
        revision = api.dataset_info(cfg["repo_id"], revision=cfg["revision"]).sha
        source = Path(
            snapshot_download(
                cfg["repo_id"],
                repo_type="dataset",
                revision=revision,
                cache_dir=root / "hub",
                allow_patterns=["meta/*", "meta/**", "data/**"],
                token=token,
            )
        )
    info = json.loads((source / "meta/info.json").read_text())
    if not info["codebase_version"].startswith("v3"):
        raise ValueError("This reader supports LeRobot v3; convert older formats first")
    camera = cfg["camera"]
    if camera not in info["features"] or info["features"][camera]["dtype"] != "video":
        raise ValueError(f"Expected video camera {camera!r} in LeRobot metadata")
    metadata_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    episodes = [r for path in metadata_files for r in pq.read_table(path).to_pylist()]
    episodes.sort(key=lambda r: r["episode_index"])
    if cfg["max_episodes"]:
        episodes = episodes[: cfg["max_episodes"]]
    if len(episodes) < 2:
        raise ValueError("At least two episodes are required for train/validation splitting")
    video_paths = {}
    for ep in episodes:
        prefix = f"videos/{camera}"
        video_paths[ep["episode_index"]] = info["video_path"].format(
            video_key=camera,
            chunk_index=ep[f"{prefix}/chunk_index"],
            file_index=ep[f"{prefix}/file_index"],
        )
    if revision != "local":
        snapshot_download(
            cfg["repo_id"],
            repo_type="dataset",
            revision=revision,
            cache_dir=root / "hub",
            allow_patterns=sorted(set(video_paths.values())),
            token=token,
        )
    table_cache, records = {}, []
    fps = float(info["fps"])
    for ep in episodes:
        episode_id = int(ep["episode_index"])
        data_path = source / info["data_path"].format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )
        if data_path not in table_cache:
            table_cache[data_path] = pq.read_table(data_path).to_pylist()
        rows = [r for r in table_cache[data_path] if int(r["episode_index"]) == episode_id]
        rows.sort(key=lambda r: r["frame_index"])
        if len(rows) != int(ep["length"]):
            raise ValueError(f"Episode {episode_id}: metadata length does not match data rows")
        actions = np.asarray([r[cfg["action_key"]] for r in rows], dtype="float32")
        state = np.asarray([r.get(cfg["state_key"], []) for r in rows], dtype="float32")
        start = float(ep[f"videos/{camera}/from_timestamp"])
        desired = start + np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
        decoded, times = [], []
        with av.open(str(source / video_paths[episode_id])) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int((start - 1) / float(stream.time_base))), stream=stream)
            for frame in container.decode(stream):
                timestamp = float(frame.pts * stream.time_base)
                if timestamp < desired[0] - 1 / fps:
                    continue
                decoded.append(frame.to_ndarray(format="rgb24"))
                times.append(timestamp)
                if timestamp >= desired[-1] + 0.5 / fps:
                    break
        if not times:
            raise ValueError(f"No decoded frames for episode {episode_id}")
        distances = np.abs(np.asarray(times)[:, None] - desired[None])
        nearest = distances.argmin(0)
        if np.any(distances[nearest, np.arange(len(rows))] > 0.51 / fps):
            raise ValueError(f"Video timestamps fail alignment tolerance in episode {episode_id}")
        pixels = np.stack(decoded)[nearest]
        records.append(save_episode(destination, episode_id, pixels, actions, state))
        print(f"Prepared episode {episode_id}: {len(rows)} frames", flush=True)
    state_dim = np.load(destination / records[0]["state"]).shape[-1]
    names = (
        ["agent_x", "agent_y"]
        if cfg["repo_id"] == "lerobot/pusht" and state_dim == 2
        else [f"state_{i}" for i in range(state_dim)]
    )
    return {
        "format": 1,
        "kind": "lerobot",
        "repo_id": cfg["repo_id"],
        "revision": revision,
        "requested_revision": cfg["revision"],
        "fps": fps,
        "camera": camera,
        "source_info": info,
        "action_dim": actions.shape[-1],
        "state_names": names,
        "episodes": records,
    }


_PREPARERS = {"lerobot": prepare_lerobot, "synthetic": prepare_synthetic}


def register_preparer(name, preparer):
    """Add an offline source adapter that writes the standard episode-cache manifest."""
    if name in _PREPARERS:
        raise ValueError(f"Dataset kind {name!r} is already registered")
    _PREPARERS[name] = preparer
