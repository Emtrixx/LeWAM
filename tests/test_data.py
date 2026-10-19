import numpy as np
import pytest
import torch

from lewam.data import make_datasets
from lewam.distractors import Distractor


def test_windows_are_episode_safe_and_action_aligned(config):
    train, val = make_datasets(config)
    assert set(train.episode_ids).isdisjoint(val.episode_ids)
    episode, start = train.index[5]
    item = train[5]
    expected = np.asarray(train.episode(episode)["actions"][start : start + 15]).reshape(3, 10)
    np.testing.assert_allclose(
        train.normalizer.denormalize(item["actions"].numpy()), expected, rtol=1e-5
    )
    assert item["pixels"].shape == (4, 3, 56, 56)
    assert item["pixels"].min() >= 0 and item["pixels"].max() <= 1
    for episode, start in train.index:
        assert start + 15 < train.records[episode]["length"]


def test_normalization_uses_only_training_episodes(config):
    train, val = make_datasets(config)
    chunks = []
    for ep in train.episode_ids:
        action = train.episode(ep)["actions"]
        chunks.extend(action[i : i + 5].reshape(-1) for i in range(len(action) - 5))
    np.testing.assert_allclose(train.normalizer.mean, np.stack(chunks).mean(0), rtol=1e-5)
    assert val.normalizer is train.normalizer


@pytest.mark.parametrize("kind", ["clean", "lissajous", "video"])
def test_distractors_are_stateless_and_do_not_mutate_input(config, kind):
    config["distractor"]["kind"] = kind
    renderer = Distractor(config["distractor"])
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    original = image.copy()
    first = renderer(image, 3, 14)
    renderer(image, 9, 500)
    np.testing.assert_array_equal(first, renderer(image, 3, 14))
    np.testing.assert_array_equal(image, original)


def test_video_preserves_foreground(config):
    config["distractor"]["kind"] = "video"
    image = np.full((96, 96, 3), 255, dtype=np.uint8)
    image[20:40, 30:50] = (65, 105, 225)
    result = Distractor(config["distractor"])(image, 1, 2)
    np.testing.assert_array_equal(result[20:40, 30:50], image[20:40, 30:50])
    assert not np.array_equal(result[0, 0], image[0, 0])


def test_manifest_mismatch_rejected(config):
    train, _ = make_datasets(config)
    wrong = dict(train.manifest, revision="another-revision")
    with pytest.raises(ValueError, match="revision"):
        make_datasets(config, expected_manifest=wrong)


def test_cache_reloads_identically(config):
    first, _ = make_datasets(config)
    second, _ = make_datasets(config)
    torch.testing.assert_close(first[0]["pixels"], second[0]["pixels"])
