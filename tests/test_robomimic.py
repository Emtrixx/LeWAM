"""Bounded real PH data / real simulator checks; opt in with LEWAM_ROBOMIMIC_DIR.

Each task renders only two 20-frame prefixes plus a few diagnostic frames.
No generated simulator, HDF5 fixture or mocked registration stands in for PH data.
"""

import hashlib
import json
import os
import resource
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from PIL import Image

from lewam.data import make_datasets
from lewam.data.prepare import prepare_dataset
from lewam.environments import make_environment

h5py = pytest.importorskip("h5py")
robomimic_data = pytest.importorskip("lewam.data.robomimic")
CATALOG, verified_source = robomimic_data.CATALOG, robomimic_data.verified_source
TASKS = [entry["task"] for entry in CATALOG["files"]]


def task_config(task):
    root = os.environ.get("LEWAM_ROBOMIMIC_DIR")
    if not root:
        pytest.skip("Set LEWAM_ROBOMIMIC_DIR to the pinned raw PH200 root")
    with initialize_config_module(config_module="lewam.configs", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"experiment={task}",
                "hardware=cpu",
                "logging.mode=disabled",
                f"data.root={root}",
                "data.max_episodes=2",
                "data.max_frames=20",
                "output_dir=" + str(Path(root).parent / "tmp/proof-run"),
            ],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def write_proof(name, proof):
    if os.environ.get("LEWAM_PROOF_DIR"):
        path = Path(os.environ["LEWAM_PROOF_DIR"]) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(proof, indent=2) + "\n")


@pytest.mark.integration
@pytest.mark.parametrize("task", TASKS)
def test_real_ph_replay_and_native_environment(task):
    import robosuite
    from OpenGL import GL

    start = time.monotonic()
    cfg = task_config(task)
    path, spec, meta = verified_source(cfg["data"])
    from lewam.data.prepare import cache_path

    cached_before = (cache_path(cfg["data"]) / "manifest.json").exists()
    preparation_start = time.monotonic()
    root, manifest = prepare_dataset(cfg["data"])
    preparation_seconds = time.monotonic() - preparation_start
    if task == "lift":
        # Simulate loss of the aggregate manifest after per-episode publication.
        # Actual arrays must be reused, without another rendering pass.
        mtimes = [(root / r["pixels"]).stat().st_mtime_ns for r in manifest["episodes"]]
        (root / "manifest.json").unlink()
        _, recovered = prepare_dataset(cfg["data"])
        assert recovered == manifest
        assert [(root / r["pixels"]).stat().st_mtime_ns for r in recovered["episodes"]] == mtimes
    assert manifest["partial"] and manifest["action_dim"] == cfg["model"]["action_dim"]
    assert cfg["eval"]["max_steps"] == spec["paper_eval_max_steps"]
    train, val = make_datasets(cfg)
    assert len(train.episode_ids) == len(val.episode_ids) == 1
    assert set(train.episode_ids).isdisjoint(val.episode_ids)
    sample = train[0]
    assert sample["pixels"].shape == (4, 3, 224, 224)
    assert sample["actions"].shape == (3, 5 * spec["action_dim"])
    chunks = np.stack(
        [train.episode(train.episode_ids[0])["actions"][i : i + 5].reshape(-1) for i in range(15)]
    )
    np.testing.assert_allclose(train.normalizer.mean, chunks.mean(0), atol=1e-7)
    env = make_environment(cfg["environment"], cfg["distractor"], 224)
    base = None
    try:
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        if not torch.cuda.is_available():
            assert "llvmpipe" in renderer.lower() or "softpipe" in renderer.lower()
        rng_before = np.random.get_state()
        first = env.reset(37)
        online_cameras = {
            c: (
                env.env.sim.model.cam_pos[env.env.sim.model.camera_name2id(c)].copy(),
                env.env.sim.model.cam_quat[env.env.sim.model.camera_name2id(c)].copy(),
            )
            for c in spec["paper_camera_semantics"]
        }
        np.testing.assert_array_equal(first, env.reset(37))
        rng_after = np.random.get_state()
        assert rng_before[0] == rng_after[0]
        np.testing.assert_array_equal(rng_before[1], rng_after[1])
        assert rng_before[2:] == rng_after[2:]
        image, done, info = env.step(np.zeros(spec["action_dim"], dtype="float32"))
        assert image.shape == (224, 224, 3) and image.dtype == np.uint8
        assert isinstance(done, bool) and isinstance(info["is_success"], bool)
        with h5py.File(path) as raw:
            demo = raw["data/demo_0"]
            xml = demo.attrs["model_file"]
            state, action = demo["states"][0], demo["actions"][0]
            env.restore_model(xml, demo.attrs.get("ep_meta"))
            native = env.replay_frame(state, 0, 0)
            n = spec["native_resolution"]
            assert native.shape == (n * len(spec["paper_camera_semantics"]), n, 3)
            independent = np.concatenate(
                [
                    env.env.sim.render(height=n, width=n, camera_name=c)[::-1]
                    for c in spec["paper_camera_semantics"]
                ],
                axis=0,
            )
            np.testing.assert_array_equal(native, independent)
            np.testing.assert_array_equal(
                native, np.load(root / manifest["episodes"][0]["pixels"])[0]
            )
            for record in manifest["episodes"]:
                d = raw[f"data/demo_{record['id']}"]
                np.testing.assert_array_equal(np.load(root / record["actions"]), d["actions"][:20])
                np.testing.assert_array_equal(np.load(root / record["state"]), d["states"][:20])
            episode = train.episode_ids[0]
            expected = np.asarray(train.episode(episode)["actions"][:15]).reshape(3, -1)
            np.testing.assert_allclose(
                train.normalizer.denormalize(sample["actions"].numpy()), expected, atol=1e-7
            )
            resized = np.asarray(
                Image.fromarray(train.episode(episode)["pixels"][0]).resize(
                    (224, 224), Image.Resampling.BILINEAR
                )
            )
            np.testing.assert_array_equal(
                (sample["pixels"][0].permute(1, 2, 0).numpy() * 255).round(), resized
            )
            sim = env.env.sim
            cameras = []
            for c in spec["paper_camera_semantics"]:
                idx = sim.model.camera_name2id(c)
                cameras.append(
                    {
                        "name": c,
                        "pos": sim.model.cam_pos[idx].tolist(),
                        "quat": sim.model.cam_quat[idx].tolist(),
                    }
                )
                np.testing.assert_allclose(sim.model.cam_pos[idx], online_cameras[c][0], atol=1e-6)
                np.testing.assert_allclose(sim.model.cam_quat[idx], online_cameras[c][1], atol=1e-6)
            visible = []
            for i in range(2):
                gid = sim.model.geom_name2id(f"lewam_box_geom_{i}")
                assert sim.model.geom_contype[gid] == sim.model.geom_conaffinity[gid] == 0
                sim.model.geom_rgba[gid, 3] = 0
                hidden = env.native_frame()
                visible.append(int(np.count_nonzero(np.any(hidden != native, axis=-1))))
                sim.model.geom_rgba[gid, 3] = 1
            assert all(v > 0 for v in visible), visible
            moved = env.replay_frame(state, 0, 50)
            assert np.any(moved != native)
            env.replay_frame(demo["states"][-1], 0, len(demo["states"]) - 1)
            terminal_success = bool(env.env._check_success())
            assert terminal_success, "Recorded successful demonstration must satisfy native success"

            env.restore_model(xml, demo.attrs.get("ep_meta"))
            env.replay_frame(state, 0, 0)
            env.step(action)
            augmented_state = sim_state = env.env.sim.get_state().flatten()
            kwargs = deepcopy(meta["env_kwargs"])
            kwargs.update(hard_reset=False, has_offscreen_renderer=False)
            from lewam.environments.robomimic import simulator_rng

            with simulator_rng(np.random.RandomState(0)):
                base = robosuite.make(meta["env_name"], **kwargs)
                base.reset()
                base.reset_from_xml_string(base.edit_model_xml(xml))
                base.sim.reset()
                base.sim.set_state_from_flattened(state)
                base.sim.forward()
                base.step(action.astype("float32"))
            clean_state = base.sim.get_state().flatten()
            np.testing.assert_allclose(augmented_state, clean_state, rtol=0, atol=1e-10)
            raw_step_difference = float(np.max(np.abs(sim_state - demo["states"][1])))
        wrong = deepcopy(manifest)
        wrong["render"]["seed"] += 1
        with pytest.raises(ValueError, match="render"):
            make_datasets(cfg, expected_manifest=wrong)
        proof = dict(
            task=task,
            raw_sha256=spec["sha256"],
            raw_episodes=200,
            source_path=str(path),
            cache_root=str(root),
            manifest=manifest,
            rendered_prefix_frames=40,
            renderer=renderer,
            cameras=cameras,
            cached_before=cached_before,
            preparation_seconds=preparation_seconds,
            native_shape=native.shape,
            model_shape=image.shape,
            reset_reproducible=True,
            global_numpy_rng_preserved=True,
            native_step_info=info,
            recorded_final_state_success=terminal_success,
            each_box_visible_changed_pixels=visible,
            boxes_move=True,
            noncollision_one_step_max_abs=float(np.max(np.abs(augmented_state - clean_state))),
            raw_next_state_max_abs=raw_step_difference,
            first_native_sha256=digest(native),
            action0_sha256=digest(action),
            elapsed_seconds=time.monotonic() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write_proof(task, proof)
        if os.environ.get("LEWAM_PROOF_DIR"):
            Image.fromarray(native).save(Path(os.environ["LEWAM_PROOF_DIR"]) / f"{task}.png")
    finally:
        if base is not None:
            base.close()
        env.close()
        env.close()
        assert env.env is None


@pytest.mark.integration
@pytest.mark.parametrize("task", ["lift", "transport"])
def test_real_checkpoint_consumers_and_model_privacy(task, tmp_path):
    from lewam.evaluation.common import load_evaluation
    from lewam.evaluation.rollout import evaluate
    from lewam.training.checkpoint import load_checkpoint, restore_rng, rng_state
    from lewam.training.trainer import build_model
    from lewam.training.trainer import train as run_train

    cfg = task_config(task)
    with initialize_config_module(config_module="lewam.configs", version_base="1.3"):
        tiny = compose(config_name="config", overrides=["hardware=cpu"])
    tiny = OmegaConf.to_container(tiny, resolve=False)
    cfg["model"] = dict(tiny["model"], action_dim=cfg["model"]["action_dim"])
    cfg["train"].update(
        batch_size=2,
        epochs=2,
        max_steps=2,
        max_val_batches=1,
        log_every=1,
        sigreg={"weight": 0.36, "knots": 17, "projections": 8},
    )
    cfg["output_dir"] = str(tmp_path / "full")
    full = run_train(deepcopy(cfg))
    interrupted = deepcopy(cfg)
    interrupted["output_dir"] = str(tmp_path / "partial")
    interrupted["train"]["max_steps"] = 1
    partial = run_train(interrupted)
    resumed = deepcopy(cfg)
    resumed.update(resume=partial["checkpoint"], output_dir=str(tmp_path / "resumed"))
    resumed = run_train(resumed)
    a, b = load_checkpoint(full["checkpoint"]), load_checkpoint(resumed["checkpoint"])

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        elif isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                equal(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for u, v in zip(x, y, strict=True):
                equal(u, v)
        else:
            assert x == y

    for key in (
        "model",
        "optimizer",
        "scheduler",
        "rng",
        "manifest",
        "normalizer",
        "epoch",
        "batch",
        "step",
        "format",
    ):
        equal(a[key], b[key])
    data, _ = make_datasets(cfg)
    sample = data[0]
    batch = {k: torch.stack([v, v]) for k, v in sample.items()}
    model = build_model(cfg).eval()
    state = rng_state()
    loss1 = model(batch)
    private = dict(
        batch, state=batch["state"] + 100, distractor_state=batch["distractor_state"] - 100
    )
    restore_rng(state)
    loss2 = model(private)
    equal(loss1, loss2)
    cfg["eval"].update(checkpoint=full["checkpoint"], max_steps=2, episodes=1, save_video=False)
    cfg["output_dir"] = str(tmp_path / "eval")
    loaded = load_evaluation(cfg)
    assert loaded[2]["environment"]["source"] == loaded[2]["data"]
    loaded[-1].close()
    result = evaluate(cfg)
    assert result["episodes"][0]["steps"] == 2
    write_proof(
        task + "-checkpoint",
        dict(
            task=task,
            checkpoint_keys=sorted(a),
            exact_cpu_continuation=True,
            state_and_distractor_privacy=True,
            evaluation=result,
            tiny_model=True,
            updates=2,
        ),
    )
