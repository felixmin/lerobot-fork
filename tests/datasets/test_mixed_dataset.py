from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import torch
import yaml

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mixed_dataset import (
    MixedLeRobotDataset,
    build_explicit_mixed_stats,
    load_dataset_mix_config,
)
from lerobot.datasets.sampler import WeightedSourceSampler
from lerobot.datasets.utils import flatten_dict


@pytest.fixture(autouse=True)
def _local_hf_datasets_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_dir))
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", str(cache_dir))
    monkeypatch.setattr(
        datasets.builder, "HF_DATASETS_CACHE", str(cache_dir), raising=False
    )


def _make_local_dataset(root: Path, repo_id: str, episode_lengths: list[int]) -> Path:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        root=root,
        use_videos=False,
        features={
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
        },
    )

    for episode_index, length in enumerate(episode_lengths):
        for frame_index in range(length):
            dataset.add_frame(
                {
                    "action": np.asarray(
                        [episode_index, frame_index], dtype=np.float32
                    ),
                    "observation.state": np.asarray(
                        [episode_index, frame_index, episode_index + frame_index],
                        dtype=np.float32,
                    ),
                    "task": f"task_{episode_index}",
                }
            )
        dataset.save_episode()

    dataset.finalize()
    return root


def _write_mix_config(path: Path, dataset_root: Path, *, overlap: bool = False) -> Path:
    second_source = {
        "name": "multitask_source",
        "repo_id": "local/stage3",
        "root": str(dataset_root),
        "weight": 1.0,
        "supervision": "multitask",
        "tolerance_s": 0.0001,
        "video_backend": "pyav",
    }
    if overlap:
        second_source["episodes"] = [1, 2]
    else:
        second_source["exclude_episodes"] = [0, 1]

    path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "latent_source",
                        "repo_id": "local/stage3",
                        "root": str(dataset_root),
                        "weight": 3.0,
                        "episodes": [0, 1],
                        "supervision": "latent_only",
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                    second_source,
                ]
            }
        )
    )
    return path


def _make_cfg(mix_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=DatasetConfig(repo_id="logical/stage3_mix", mix_path=str(mix_path)),
        policy=SimpleNamespace(
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=None,
        ),
        tolerance_s=1e-4,
    )


def test_load_dataset_mix_config_supports_mix_path(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)

    cfg = DatasetConfig(repo_id="logical/stage3_mix", mix_path=str(mix_path))
    mix_cfg = load_dataset_mix_config(cfg.mix_path)

    assert cfg.mix_path == str(mix_path)
    assert mix_cfg.path == mix_path.resolve()
    assert mix_cfg.sources[0].name == "latent_source"
    assert mix_cfg.sources[0].episodes == (0, 1)
    assert mix_cfg.sources[1].exclude_episodes == (0, 1)
    assert mix_cfg.sources[1].supervision == "multitask"


def test_load_dataset_mix_config_treats_null_episode_fields_as_absent(tmp_path):
    mix_path = tmp_path / "mix.yaml"
    mix_path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "latent_source",
                        "repo_id": "local/stage3",
                        "weight": 1.0,
                        "episodes": [0],
                        "exclude_episodes": None,
                        "supervision": "latent_only",
                    }
                ]
            }
        )
    )

    mix_cfg = load_dataset_mix_config(mix_path)

    assert mix_cfg.sources[0].episodes == (0,)
    assert mix_cfg.sources[0].exclude_episodes is None


def test_make_dataset_routes_mix_path_and_stamps_supervision(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)

    dataset = make_dataset(_make_cfg(mix_path))

    assert isinstance(dataset, MixedLeRobotDataset)
    assert dataset.repo_id == "logical/stage3_mix"
    assert dataset.meta.info["logical_repo_id"] == "logical/stage3_mix"

    latent_item = dataset[0]
    multitask_item = dataset[6]

    assert bool(latent_item["hlrp_action_supervised"]) is False
    assert bool(latent_item["hlrp_latent_supervised"]) is True
    assert latent_item["hlrp_source_name"] == "latent_source"
    assert latent_item["dataset_source_name"] == "latent_source"
    assert latent_item["dataset_source_repo_id"] == "local/stage3"

    assert bool(multitask_item["hlrp_action_supervised"]) is True
    assert bool(multitask_item["hlrp_latent_supervised"]) is True
    assert multitask_item["hlrp_source_name"] == "multitask_source"
    assert multitask_item["dataset_source_name"] == "multitask_source"
    assert multitask_item["dataset_source_root"] == str(dataset_root)


def test_mixed_dataset_collates_supervision_and_source_metadata(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)
    dataset = make_dataset(_make_cfg(mix_path))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        num_workers=0,
        sampler=dataset.build_sampler(seed=11),
    )
    batch = next(iter(dataloader))

    assert batch["hlrp_action_supervised"].dtype == torch.bool
    assert batch["hlrp_latent_supervised"].dtype == torch.bool
    assert isinstance(batch["hlrp_source_name"], list)
    assert batch["dataset_source_index"].dtype == torch.int64
    assert isinstance(batch["dataset_source_name"], list)
    assert len(batch["dataset_source_name"]) == 3


def test_mixed_dataset_shares_physical_dataset_instance_across_sources(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)
    dataset = make_dataset(_make_cfg(mix_path))

    shared_a = dataset.sources[0]._get_dataset()
    shared_b = dataset.sources[1]._get_dataset()

    assert shared_a is shared_b
    assert shared_a.episodes is None
    assert int(dataset.sources[0].get_item(0)["index"]) == 0
    assert int(dataset.sources[1].get_item(6)["index"]) == 6


def test_mixed_dataset_meta_stats_follow_explicit_source_weights(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)
    dataset = make_dataset(_make_cfg(mix_path))

    weighted_stats = build_explicit_mixed_stats(
        [
            SimpleNamespace(
                stats={"action": {"mean": np.asarray([0.5, 1.0]), "std": np.asarray([1.0, 1.0]), "min": np.asarray([0.0, 0.0]), "max": np.asarray([1.0, 2.0]), "count": np.asarray([6])}},
                weight=3.0,
            ),
            SimpleNamespace(
                stats={"action": {"mean": np.asarray([2.5, 1.5]), "std": np.asarray([1.0, 1.0]), "min": np.asarray([2.0, 0.0]), "max": np.asarray([3.0, 3.0]), "count": np.asarray([8])}},
                weight=1.0,
            ),
        ]
    )
    action_mean = flatten_dict(weighted_stats)["action/mean"]

    assert action_mean.shape == (2,)
    assert action_mean[0] == pytest.approx(1.0, abs=1e-6)
    assert flatten_dict(dataset.meta.stats) == flatten_dict(
        build_explicit_mixed_stats(dataset.sources)
    )


def test_mixed_dataset_rejects_source_overlap(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(
        tmp_path / "mix_overlap.yaml", dataset_root, overlap=True
    )

    with pytest.raises(ValueError, match="cannot overlap"):
        make_dataset(_make_cfg(mix_path))


def test_weighted_source_sampler_biases_towards_heavier_sources(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)
    dataset = make_dataset(_make_cfg(mix_path))

    sampler = WeightedSourceSampler(
        sources=dataset.sources,
        source_weights=dataset.source_weights,
        num_samples=2000,
        seed=7,
        drop_n_last_frames=1,
    )
    source_ids = [source_id for source_id, _ in sampler]

    assert len(dataset.build_sampler(seed=7, drop_n_last_frames=1)) == 10
    assert sum(source_id == 0 for source_id in source_ids) / len(source_ids) > 0.65
