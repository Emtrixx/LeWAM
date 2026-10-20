"""Episode-safe temporal windows; normalization is fitted on training chunks only."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lewam.distractors import Distractor

from .prepare import prepare_dataset, write_json


@dataclass
class ActionNormalizer:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, actions):
        return (actions - self.mean) / self.std

    def denormalize(self, actions):
        return actions * self.std + self.mean

    def as_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value):
        return cls(
            np.asarray(value["mean"], dtype="float32"), np.asarray(value["std"], dtype="float32")
        )


class WindowDataset(Dataset):
    def __init__(self, root, manifest, episode_ids, model_cfg, distractor_cfg, normalizer=None):
        self.root, self.manifest = root, manifest
        self.episode_ids = list(episode_ids)
        self.image_size = model_cfg["image_size"]
        self.context = model_cfg["context_frames"]
        self.chunk = model_cfg["chunk_size"]
        if manifest["kind"] == "robomimic":
            from lewam.distractors.boxes import SimBoxes

            if manifest["render"] != distractor_cfg:
                raise ValueError("Prepared and requested physical distractor recipes differ")
            self.distractor = SimBoxes(distractor_cfg)
        else:
            self.distractor = Distractor(distractor_cfg)
        self.normalizer = normalizer
        self.records = {ep["id"]: ep for ep in manifest["episodes"]}
        self.arrays = {}
        self.index = [
            (episode, start)
            for episode in episode_ids
            for start in range(self.records[episode]["length"] - self.context * self.chunk)
        ]
        if not self.index:
            raise ValueError("No complete training windows; use longer/more episodes")

    def episode(self, episode_id):
        if episode_id not in self.arrays:
            self.arrays[episode_id] = {
                key: np.load(self.root / self.records[episode_id][key], mmap_mode="r")
                for key in ("pixels", "actions", "state")
            }
        return self.arrays[episode_id]

    def __len__(self):
        return len(self.index)

    def image(self, episode_id, step):
        frame = self.distractor(self.episode(episode_id)["pixels"][step], episode_id, step)
        image = Image.fromarray(frame).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)

    def __getitem__(self, index):
        episode, start = self.index[index]
        steps = start + np.arange(self.context + 1) * self.chunk
        arrays = self.episode(episode)
        action = np.array(arrays["actions"][start : steps[-1]], copy=True).reshape(self.context, -1)
        if self.normalizer is not None:
            action = self.normalizer.normalize(action)
        state = np.array(arrays["state"][steps], copy=True)
        distractor_state = np.stack([self.distractor.state(episode, int(s)) for s in steps])
        return {
            "pixels": torch.stack([self.image(episode, int(step)) for step in steps]),
            "actions": torch.from_numpy(action.astype("float32")),
            "state": torch.from_numpy(state.astype("float32")),
            "distractor_state": torch.from_numpy(distractor_state),
            "episode": torch.tensor(episode),
            "start": torch.tensor(start),
        }

    def fit_normalizer(self):
        # Each chunk offset has its own statistics, as specified in the paper.
        chunks = []
        for episode in self.episode_ids:
            action = self.episode(episode)["actions"]
            chunks.extend(
                np.asarray(action[i : i + self.chunk]).reshape(-1)
                for i in range(len(action) - self.chunk)
            )
        values = np.stack(chunks).astype("float64")
        return ActionNormalizer(
            values.mean(0).astype("float32"), np.maximum(values.std(0), 1e-6).astype("float32")
        )


def make_datasets(cfg, output_dir=None, normalizer=None, expected_manifest=None):
    root, manifest = prepare_dataset(cfg["data"])
    if cfg["data"]["frameskip"] != cfg["model"]["chunk_size"]:
        raise ValueError("Frame spacing must equal the number of primitive actions per chunk")
    if manifest["action_dim"] != cfg["model"]["action_dim"]:
        raise ValueError("Dataset and model action dimensions disagree")
    if float(manifest["fps"]) != float(cfg["environment"]["fps"]):
        raise ValueError("Dataset and environment frame rates disagree")
    if expected_manifest is not None:
        if manifest["kind"] == "robomimic":
            for key in (
                "raw_sha256",
                "task",
                "render",
                "render_identity",
                "env_args",
                "native_resolution",
                "composition",
                "partial",
            ):
                if manifest[key] != expected_manifest[key]:
                    raise ValueError(f"Checkpoint dataset {key} differs from prepared data")
        for key in ("kind", "revision", "camera"):
            if manifest[key] != expected_manifest[key]:
                raise ValueError(f"Checkpoint dataset {key} differs from prepared data")
        if manifest["episodes"] != expected_manifest["episodes"]:
            raise ValueError("Checkpoint and current dataset episode selections differ")
    ids = np.asarray([r["id"] for r in manifest["episodes"]])
    np.random.default_rng(cfg["data"]["split_seed"]).shuffle(ids)
    split = int(len(ids) * cfg["data"]["train_fraction"])
    if not 0 < split < len(ids):
        raise ValueError(
            "The episode split must leave at least one train and one validation episode"
        )
    kwargs = dict(
        root=root, manifest=manifest, model_cfg=cfg["model"], distractor_cfg=cfg["distractor"]
    )
    train = WindowDataset(episode_ids=ids[:split].tolist(), **kwargs)
    normalizer = normalizer or train.fit_normalizer()
    train.normalizer = normalizer
    val = WindowDataset(episode_ids=ids[split:].tolist(), normalizer=normalizer, **kwargs)
    if output_dir:
        write_json(output_dir / "dataset.json", manifest)
        write_json(output_dir / "split.json", {"train": train.episode_ids, "val": val.episode_ids})
        write_json(output_dir / "normalization.json", normalizer.as_dict())
    return train, val
