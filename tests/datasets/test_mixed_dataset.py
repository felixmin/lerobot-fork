from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
import torch
import yaml

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.compact_mixed_dataset import CompactMixedDataset, CompactSourceAdapter
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mixed_dataset import (
    MixedLeRobotDataset,
    _aggregate_selected_stats,
    build_explicit_mixed_stats,
    load_dataset_mix_config,
)
from lerobot.datasets.sampler import WeightedSourceSampler
from lerobot.datasets.utils import flatten_dict


def _assert_flat_stats_equal(
    left: dict[str, np.ndarray], right: dict[str, np.ndarray]
) -> None:
    assert set(left) == set(right)
    for key in left:
        np.testing.assert_allclose(np.asarray(left[key]), np.asarray(right[key]))


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
        "action_supervision": True,
        "latent_supervision": True,
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
                        "action_supervision": False,
                        "latent_supervision": True,
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                    second_source,
                ]
            }
        )
    )
    return path


def _write_three_source_mix_config(path: Path, dataset_root: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "source_a",
                        "repo_id": "local/stage3",
                        "root": str(dataset_root),
                        "weight": 1.0,
                        "episodes": [0],
                        "action_supervision": True,
                        "latent_supervision": True,
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                    {
                        "name": "source_b",
                        "repo_id": "local/stage3",
                        "root": str(dataset_root),
                        "weight": 1.0,
                        "episodes": [1],
                        "action_supervision": True,
                        "latent_supervision": True,
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                    {
                        "name": "source_c",
                        "repo_id": "local/stage3",
                        "root": str(dataset_root),
                        "weight": 1.0,
                        "exclude_episodes": [0, 1],
                        "action_supervision": True,
                        "latent_supervision": True,
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                ]
            }
        )
    )
    return path


def _write_labeled_disjoint_mix_config(path: Path, dataset_root: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "latent_only",
                        "repo_id": "local/stage3",
                        "root": str(dataset_root),
                        "weight": 4.0,
                        "episodes": [0, 2],
                        "action_supervision": False,
                        "latent_supervision": True,
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                    {
                        "name": "action_only",
                        "repo_id": "local/stage3",
                        "root": str(dataset_root),
                        "weight": 1.0,
                        "exclude_episodes": [0, 2],
                        "action_supervision": True,
                        "latent_supervision": False,
                        "video_backend": "pyav",
                        "tolerance_s": 0.0001,
                    },
                ]
            }
        )
    )
    return path


def _make_cfg(
    mix_path: Path,
    *,
    mix_implementation: str = "current",
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="logical/stage3_mix",
            mix_path=str(mix_path),
            mix_implementation=mix_implementation,
        ),
        policy=SimpleNamespace(
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=None,
        ),
        tolerance_s=1e-4,
    )


def _make_cfg_with_observation_delta(mix_path: Path, observation_delta_indices: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="logical/stage3_mix",
            mix_path=str(mix_path),
            mix_implementation="current",
        ),
        policy=SimpleNamespace(
            reward_delta_indices=None,
            action_delta_indices=None,
            observation_delta_indices=observation_delta_indices,
        ),
        tolerance_s=1e-4,
    )


def _item_episode_index(item: dict[str, torch.Tensor | str]) -> int:
    return int(torch.as_tensor(item["action"])[0].item())


def test_load_dataset_mix_config_supports_mix_path(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)

    cfg = DatasetConfig(
        repo_id="logical/stage3_mix",
        mix_path=str(mix_path),
        mix_implementation="current",
    )
    mix_cfg = load_dataset_mix_config(cfg.mix_path)

    assert cfg.mix_path == str(mix_path)
    assert mix_cfg.path == mix_path.resolve()
    assert mix_cfg.sources[0].name == "latent_source"
    assert mix_cfg.sources[0].episodes == (0, 1)
    assert mix_cfg.sources[1].exclude_episodes == (0, 1)
    assert mix_cfg.sources[1].action_supervision is True
    assert mix_cfg.sources[1].latent_supervision is True


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
                        "action_supervision": False,
                        "latent_supervision": True,
                    }
                ]
            }
        )
    )

    mix_cfg = load_dataset_mix_config(mix_path)

    assert mix_cfg.sources[0].episodes == (0,)
    assert mix_cfg.sources[0].exclude_episodes is None


def test_load_dataset_mix_config_rejects_removed_supervision_field(tmp_path):
    mix_path = tmp_path / "mix_removed_supervision.yaml"
    mix_path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "latent_source",
                        "repo_id": "local/stage3",
                        "weight": 1.0,
                        "supervision": "latent_only",
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="removed field 'supervision'"):
        load_dataset_mix_config(mix_path)


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

    assert bool(latent_item["action_supervision"]) is False
    assert bool(latent_item["latent_supervision"]) is True
    assert latent_item["dataset_source_name"] == "latent_source"
    assert latent_item["dataset_source_repo_id"] == "local/stage3"

    assert bool(multitask_item["action_supervision"]) is True
    assert bool(multitask_item["latent_supervision"]) is True
    assert multitask_item["dataset_source_name"] == "multitask_source"
    assert multitask_item["dataset_source_root"] == str(dataset_root)


def test_make_dataset_routes_compact_manifest_as_selectable_alternate(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)

    dataset = make_dataset(
        _make_cfg(mix_path, mix_implementation="compact_manifest")
    )

    assert isinstance(dataset, CompactMixedDataset)
    assert dataset.loader_hints()["mixed_impl"] == "compact_manifest"
    assert len(dataset) == dataset.num_frames == int(dataset.meta.info["total_frames"])
    assert dataset.num_episodes == int(dataset.meta.info["total_episodes"])

    batch_items = dataset.__getitems__([0, 1, 2])
    assert len(batch_items) == 3
    assert [item["dataset_source_name"] for item in batch_items] == [
        "latent_source",
        "latent_source",
        "latent_source",
    ]


def test_compact_manifest_dataset_collates_mixed_batches(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)
    dataset = make_dataset(
        _make_cfg(
            mix_path,
            mix_implementation="compact_manifest",
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        num_workers=0,
        sampler=dataset.build_sampler(seed=11, batch_size=3),
    )
    batch = next(iter(dataloader))

    assert batch["action_supervision"].dtype == torch.bool
    assert batch["latent_supervision"].dtype == torch.bool
    assert batch["observation.state_is_pad"].dtype == torch.bool
    assert batch["action_is_pad"].dtype == torch.bool
    assert isinstance(batch["dataset_source_name"], list)


def test_compact_manifest_default_runtime_cache_scales_with_active_sources(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_three_source_mix_config(tmp_path / "mix_three.yaml", dataset_root)
    dataset = make_dataset(
        _make_cfg(
            mix_path,
            mix_implementation="compact_manifest",
        )
    )

    assert isinstance(dataset, CompactMixedDataset)
    assert len(dataset.sources) == 3
    assert dataset.max_open_datasets_per_worker == 3


def test_compact_source_opens_full_physical_dataset_for_logical_splits(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)
    dataset = make_dataset(
        _make_cfg(
            mix_path,
            mix_implementation="compact_manifest",
        )
    )

    assert isinstance(dataset, CompactMixedDataset)

    source = dataset.sources[0]
    assert isinstance(source, CompactSourceAdapter)

    physical_dataset = source.make_dataset()

    assert physical_dataset.episodes is None
    assert physical_dataset._absolute_to_relative_idx is None

    anchor_abs_index = source.flat_index_to_anchor(0)
    item = source.fetch_one(physical_dataset, anchor_abs_index=anchor_abs_index)

    assert int(item["index"]) == anchor_abs_index
    assert item["dataset_source_name"] == source.name


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

    assert batch["action_supervision"].dtype == torch.bool
    assert batch["latent_supervision"].dtype == torch.bool
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


def test_legacy_mixed_dataset_only_returns_selected_episodes_and_source_labels(tmp_path):
    episode_lengths = [2, 3, 4, 5]
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", episode_lengths
    )
    mix_path = _write_labeled_disjoint_mix_config(
        tmp_path / "legacy_mix.yaml", dataset_root
    )
    dataset = make_dataset(_make_cfg(mix_path, mix_implementation="legacy"))

    assert len(dataset) == sum(episode_lengths)

    episodes_by_source: dict[str, list[int]] = {"latent_only": [], "action_only": []}
    for item_index in range(len(dataset)):
        item = dataset[item_index]
        source_name = item["dataset_source_name"]
        episodes_by_source[source_name].append(_item_episode_index(item))

        if source_name == "latent_only":
            assert bool(item["action_supervision"]) is False
            assert bool(item["latent_supervision"]) is True
        elif source_name == "action_only":
            assert bool(item["action_supervision"]) is True
            assert bool(item["latent_supervision"]) is False
        else:
            raise AssertionError(f"Unexpected source name {source_name!r}")

    assert Counter(episodes_by_source["latent_only"]) == Counter({0: 2, 2: 4})
    assert Counter(episodes_by_source["action_only"]) == Counter({1: 3, 3: 5})


def test_legacy_weighted_sampler_respects_source_weights_and_sample_labels(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [3, 3, 4, 4]
    )
    mix_path = _write_labeled_disjoint_mix_config(
        tmp_path / "legacy_mix.yaml", dataset_root
    )
    dataset = make_dataset(_make_cfg(mix_path, mix_implementation="legacy"))

    sampler = WeightedSourceSampler(
        sources=dataset.sources,
        source_weights=dataset.source_weights,
        num_samples=3000,
        seed=17,
        drop_n_last_frames=1,
    )
    sampled_indices = list(sampler)
    source_counts = Counter(source_id for source_id, _ in sampled_indices)
    latent_fraction = source_counts[0] / len(sampled_indices)

    assert len(dataset.build_sampler(seed=17, drop_n_last_frames=1)) == 10
    assert latent_fraction > 0.72
    assert source_counts[1] > 0

    checked_by_source = Counter()
    for source_id, anchor_abs_index in sampled_indices:
        if checked_by_source[source_id] >= 20:
            continue

        item = dataset[(source_id, anchor_abs_index)]
        source = dataset.sources[source_id]
        checked_by_source[source_id] += 1

        assert item["dataset_source_name"] == source.name
        assert bool(item["action_supervision"]) is source.action_supervision
        assert bool(item["latent_supervision"]) is source.latent_supervision
        assert _item_episode_index(item) in source.selected_episodes

        if all(checked_by_source[idx] >= 20 for idx in range(len(dataset.sources))):
            break

    assert checked_by_source == Counter({0: 20, 1: 20})


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
    _assert_flat_stats_equal(
        flatten_dict(dataset.meta.stats),
        flatten_dict(build_explicit_mixed_stats(dataset.sources)),
    )


def test_aggregate_selected_stats_falls_back_to_dataset_level_stats_for_missing_keys():
    meta = SimpleNamespace(
        episodes=datasets.Dataset.from_dict(
            {
                "episode_index": [0, 1],
                "stats/action/mean": [
                    np.asarray([1.0, 2.0], dtype=np.float32),
                    np.asarray([3.0, 4.0], dtype=np.float32),
                ],
                "stats/action/std": [
                    np.asarray([0.5, 0.25], dtype=np.float32),
                    np.asarray([0.25, 0.5], dtype=np.float32),
                ],
                "stats/action/min": [
                    np.asarray([0.0, 1.0], dtype=np.float32),
                    np.asarray([2.0, 3.0], dtype=np.float32),
                ],
                "stats/action/max": [
                    np.asarray([2.0, 3.0], dtype=np.float32),
                    np.asarray([4.0, 5.0], dtype=np.float32),
                ],
                "stats/action/count": [
                    np.asarray([10], dtype=np.int64),
                    np.asarray([20], dtype=np.int64),
                ],
            }
        ),
        stats={
            "action": {
                "mean": np.asarray([99.0, 99.0], dtype=np.float32),
                "std": np.asarray([1.0, 1.0], dtype=np.float32),
                "min": np.asarray([0.0, 0.0], dtype=np.float32),
                "max": np.asarray([100.0, 100.0], dtype=np.float32),
                "count": np.asarray([30], dtype=np.int64),
            },
            "latent_labels.continuous_vector_latents": {
                "mean": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                "std": np.asarray([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
                "min": np.asarray([[-1.0, -1.0], [-1.0, -1.0]], dtype=np.float32),
                "max": np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
                "count": np.asarray([[100, 100], [100, 100]], dtype=np.int64),
            },
        },
    )

    stats = _aggregate_selected_stats(meta, (0, 1))

    assert "action" in stats
    assert "latent_labels.continuous_vector_latents" in stats
    np.testing.assert_allclose(
        stats["action"]["mean"],
        np.asarray([2.3333333, 3.3333333], dtype=np.float32),
    )
    np.testing.assert_allclose(
        stats["latent_labels.continuous_vector_latents"]["mean"],
        np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
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




def test_mixed_dataset_sampler_drops_source_lookahead_frames(tmp_path):
    dataset_root = _make_local_dataset(
        tmp_path / "dataset", "local/stage3", [5, 4, 4, 4]
    )
    mix_path = _write_mix_config(tmp_path / "mix.yaml", dataset_root)

    dataset = make_dataset(_make_cfg_with_observation_delta(mix_path, [0, 2]))

    latent_source = dataset.sources[0]
    multitask_source = dataset.sources[1]

    assert latent_source.required_lookahead_frames == 2
    assert multitask_source.required_lookahead_frames == 2
    assert latent_source.get_effective_lengths().tolist() == [3, 2]
    assert multitask_source.get_effective_lengths().tolist() == [2, 2]


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
