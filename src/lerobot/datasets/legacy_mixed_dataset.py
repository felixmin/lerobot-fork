from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mixed_dataset import (
    DatasetMixConfig,
    DatasetMixSourceConfig,
    MixedLeRobotDatasetMetadata,
    MixedSourceMetadata,
    SourceIndex,
    _aggregate_selected_stats,
    _build_source_index,
    _debug_mixed_dataset,
    _selected_episodes,
    _stats_signature,
    build_explicit_mixed_stats,
    load_dataset_mix_config,
)
from lerobot.datasets.sampler import WeightedSourceSampler


def validate_legacy_mix_config(mix_cfg: DatasetMixConfig) -> None:
    if mix_cfg.retained_features is not None:
        raise ValueError(
            "Legacy mixed dataset does not support compatibility.retained_features."
        )
    if not mix_cfg.enforce_matching_fps:
        raise ValueError(
            "Legacy mixed dataset requires compatibility.enforce_matching_fps=true."
        )
    if not mix_cfg.enforce_matching_delta_timestamps:
        raise ValueError(
            "Legacy mixed dataset requires compatibility.enforce_matching_delta_timestamps=true."
        )
    if mix_cfg.allow_visual_shape_mismatch:
        raise ValueError(
            "Legacy mixed dataset does not support compatibility.allow_visual_shape_mismatch=true."
        )
    for source in mix_cfg.sources:
        if source.feature_key_mapping:
            raise ValueError(
                f"Legacy mixed dataset does not support feature_key_mapping for source {source.name!r}."
            )


@dataclass
class LegacyLogicalSource:
    source_index: int
    config: DatasetMixSourceConfig
    meta: LeRobotDatasetMetadata
    delta_timestamps: dict[str, list[float]] | None
    image_transforms: Any = None
    default_tolerance_s: float = 1e-4
    shared_dataset_cache: dict[tuple[Any, ...], LeRobotDataset] | None = None

    def __post_init__(self) -> None:
        self.delta_timestamps = deepcopy(self.delta_timestamps)
        self.selected_episodes = _selected_episodes(self.config, self.meta.total_episodes)
        self.index: SourceIndex = _build_source_index(self.meta, self.selected_episodes)
        self.features = deepcopy(self.meta.features)
        self.stats = _aggregate_selected_stats(self.meta, self.selected_episodes)
        self.tolerance_s = float(
            self.default_tolerance_s
            if self.config.tolerance_s is None
            else self.config.tolerance_s
        )
        self._shared_dataset_cache = (
            {} if self.shared_dataset_cache is None else self.shared_dataset_cache
        )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def repo_id(self) -> str:
        return self.config.repo_id

    @property
    def root(self) -> str | None:
        return self.config.root

    @property
    def revision(self) -> str | None:
        return self.config.revision

    @property
    def weight(self) -> float:
        return self.config.weight

    @property
    def action_supervision(self) -> bool:
        return bool(self.config.action_supervision)

    @property
    def latent_supervision(self) -> bool:
        return bool(self.config.latent_supervision)

    @property
    def num_frames(self) -> int:
        return int(self.index.lengths.sum())

    @property
    def num_episodes(self) -> int:
        return int(len(self.selected_episodes))

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in {"image", "video"}
        ]

    @property
    def dataset_identity(self) -> tuple[str, str | None, str | None]:
        return (self.repo_id, str(Path(self.meta.root).resolve()), self.revision)

    def get_effective_lengths(self, drop_n_last_frames: int = 0) -> np.ndarray:
        return np.maximum(self.index.lengths - int(drop_n_last_frames), 0)

    def flat_index_to_anchor(self, index: int, *, drop_n_last_frames: int = 0) -> int:
        effective_lengths = self.get_effective_lengths(drop_n_last_frames)
        total_effective = int(effective_lengths.sum())
        if index < 0 or index >= total_effective:
            raise IndexError(f"Index {index} out of bounds for source {self.name!r}.")

        cumulative_lengths = np.cumsum(effective_lengths, dtype=np.int64)
        episode_pos = int(np.searchsorted(cumulative_lengths, index, side="right"))
        prev_total = 0 if episode_pos == 0 else int(cumulative_lengths[episode_pos - 1])
        return int(self.index.dataset_from_index[episode_pos] + (index - prev_total))

    def metadata(self) -> MixedSourceMetadata:
        return MixedSourceMetadata(
            name=self.name,
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            weight=self.weight,
            action_supervision=self.action_supervision,
            latent_supervision=self.latent_supervision,
            episodes=self.selected_episodes,
            video_backend=self.config.video_backend,
            tolerance_s=self.tolerance_s,
            num_frames=self.num_frames,
            feature_key_mapping={},
            retained_features=None,
        )

    def _get_dataset(self) -> LeRobotDataset:
        cache_key = (
            self.repo_id,
            str(Path(self.meta.root).resolve()),
            self.revision,
            self.config.video_backend,
            float(self.tolerance_s),
            repr(self.delta_timestamps),
            id(self.image_transforms),
        )
        dataset = self._shared_dataset_cache.get(cache_key)
        if dataset is None:
            dataset = LeRobotDataset(
                self.repo_id,
                root=self.meta.root,
                episodes=None,
                image_transforms=self.image_transforms,
                delta_timestamps=self.delta_timestamps,
                revision=self.revision,
                video_backend=self.config.video_backend,
                tolerance_s=self.tolerance_s,
            )
            self._shared_dataset_cache[cache_key] = dataset
        return dataset

    def get_item(self, anchor_abs_index: int) -> dict[str, Any]:
        dataset = self._get_dataset()
        if dataset._absolute_to_relative_idx is None:
            relative_index = int(anchor_abs_index)
        else:
            relative_index = int(dataset._absolute_to_relative_idx[int(anchor_abs_index)])
        if _debug_mixed_dataset():
            import logging

            logging.getLogger(__name__).info(
                "[legacy_mixed] source=%s action_supervision=%s latent_supervision=%s anchor_abs=%s relative=%s",
                self.name,
                self.action_supervision,
                self.latent_supervision,
                int(anchor_abs_index),
                int(relative_index),
            )

        item = dataset[relative_index]
        item["action_supervision"] = torch.tensor(self.action_supervision, dtype=torch.bool)
        item["latent_supervision"] = torch.tensor(self.latent_supervision, dtype=torch.bool)
        item["dataset_source_index"] = torch.tensor(self.source_index, dtype=torch.int64)
        item["dataset_source_name"] = self.name
        item["dataset_source_repo_id"] = self.repo_id
        item["dataset_source_root"] = "" if self.root is None else self.root
        item["dataset_source_revision"] = "" if self.revision is None else self.revision
        return item


def _build_virtual_episodes(sources: list[LegacyLogicalSource]) -> pd.DataFrame:
    rows = []
    cursor = 0
    logical_episode_index = 0
    for source in sources:
        for source_episode_index, length in zip(
            source.selected_episodes, source.index.lengths, strict=True
        ):
            rows.append(
                {
                    "episode_index": logical_episode_index,
                    "dataset_from_index": cursor,
                    "dataset_to_index": cursor + int(length),
                    "source_name": source.name,
                    "source_repo_id": source.repo_id,
                    "source_episode_index": int(source_episode_index),
                }
            )
            cursor += int(length)
            logical_episode_index += 1
    return pd.DataFrame(rows)


def _build_mixed_info(
    logical_repo_id: str,
    mix_path: Path,
    sources: list[LegacyLogicalSource],
) -> dict[str, Any]:
    info = deepcopy(sources[0].meta.info)
    info["total_episodes"] = int(sum(source.num_episodes for source in sources))
    info["total_frames"] = int(sum(source.num_frames for source in sources))
    info["mixed_sources"] = [
        {
            "name": metadata.name,
            "repo_id": metadata.repo_id,
            "root": metadata.root,
            "revision": metadata.revision,
            "weight": metadata.weight,
            "action_supervision": metadata.action_supervision,
            "latent_supervision": metadata.latent_supervision,
            "episodes": list(metadata.episodes),
            "video_backend": metadata.video_backend,
            "tolerance_s": metadata.tolerance_s,
            "num_frames": metadata.num_frames,
        }
        for metadata in (source.metadata() for source in sources)
    ]
    info["mix_path"] = str(mix_path)
    info["logical_repo_id"] = logical_repo_id
    return info


def validate_legacy_mixed_sources(sources: list[LegacyLogicalSource]) -> None:
    if len(sources) == 0:
        raise ValueError("Expected at least one logical source.")

    base_source = sources[0]
    base_features = base_source.features
    base_fps = base_source.meta.fps
    base_delta_timestamps = base_source.delta_timestamps
    base_camera_keys = base_source.camera_keys
    base_stats_signature = _stats_signature(base_source.stats)

    seen_episodes: dict[tuple[str, str | None, str | None], set[int]] = {}
    for source in sources:
        if source.num_frames <= 0:
            raise ValueError(f"Mix source {source.name!r} resolved to zero frames.")
        if source.meta.fps != base_fps:
            raise ValueError("All mix sources must have matching fps.")
        if source.features != base_features:
            raise ValueError("All mix sources must have identical feature schemas.")
        if source.camera_keys != base_camera_keys:
            raise ValueError("All mix sources must expose the same camera keys.")
        if source.delta_timestamps != base_delta_timestamps:
            raise ValueError("All mix sources must resolve to identical delta timestamps.")
        if _stats_signature(source.stats) != base_stats_signature:
            raise ValueError(
                "All mix sources must expose compatible normalization-stat schemas."
            )

        seen = seen_episodes.setdefault(source.dataset_identity, set())
        overlap = seen.intersection(source.selected_episodes)
        if overlap:
            overlap_text = ", ".join(str(episode) for episode in sorted(overlap))
            raise ValueError(
                "Mix sources cannot overlap on the same physical dataset. "
                f"Source {source.name!r} overlaps on episodes [{overlap_text}]."
            )
        seen.update(source.selected_episodes)


class LegacyMixedLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        logical_repo_id: str,
        mix_path: str | Path,
        sources: list[LegacyLogicalSource],
    ) -> None:
        super().__init__()
        validate_legacy_mixed_sources(sources)
        self.repo_id = logical_repo_id
        self.mix_path = str(Path(mix_path).expanduser().resolve())
        self.sources = list(sources)
        self.source_weights = np.asarray([source.weight for source in self.sources], dtype=np.float64)
        self.episodes = None
        self._cumulative_frames = np.cumsum([source.num_frames for source in self.sources], dtype=np.int64)

        self.meta = MixedLeRobotDatasetMetadata(
            repo_id=self.repo_id,
            mix_path=self.mix_path,
            fps=int(self.sources[0].meta.fps),
            features=deepcopy(self.sources[0].features),
            stats=build_explicit_mixed_stats(self.sources),
            episodes=_build_virtual_episodes(self.sources),
            source_metadata=tuple(source.metadata() for source in self.sources),
            info=_build_mixed_info(self.repo_id, Path(self.mix_path), self.sources),
        )

    @property
    def num_frames(self) -> int:
        return 0 if len(self._cumulative_frames) == 0 else int(self._cumulative_frames[-1])

    @property
    def num_episodes(self) -> int:
        return int(sum(source.num_episodes for source in self.sources))

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.meta.features

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        if isinstance(index, tuple):
            source_index, anchor_abs_index = index
            return self.sources[int(source_index)].get_item(int(anchor_abs_index))

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds.")

        source_index = int(np.searchsorted(self._cumulative_frames, int(index), side="right"))
        previous_total = 0 if source_index == 0 else int(self._cumulative_frames[source_index - 1])
        logical_index = int(index) - previous_total
        anchor_abs_index = self.sources[source_index].flat_index_to_anchor(logical_index)
        return self.sources[source_index].get_item(anchor_abs_index)

    def build_sampler(
        self,
        *,
        seed: int | None = None,
        drop_n_last_frames: int = 0,
    ) -> WeightedSourceSampler:
        num_samples = int(
            sum(source.get_effective_lengths(drop_n_last_frames).sum() for source in self.sources)
        )
        if num_samples <= 0:
            raise ValueError("Mix dataset has no effective samples after drop_n_last_frames.")
        return WeightedSourceSampler(
            sources=self.sources,
            source_weights=self.source_weights,
            num_samples=num_samples,
            seed=0 if seed is None else int(seed),
            drop_n_last_frames=drop_n_last_frames,
        )


def make_legacy_mixed_dataset(
    *,
    logical_repo_id: str,
    mix_path: str | Path,
    default_tolerance_s: float,
    image_transforms: Any,
    sources: tuple[DatasetMixSourceConfig, ...],
    metadata_by_source: list[LeRobotDatasetMetadata],
    delta_timestamps_by_source: list[dict[str, list[float]] | None],
) -> LegacyMixedLeRobotDataset:
    shared_dataset_cache: dict[tuple[Any, ...], LeRobotDataset] = {}
    logical_sources = [
        LegacyLogicalSource(
            source_index=source_index,
            config=source_cfg,
            meta=source_meta,
            delta_timestamps=source_delta_timestamps,
            image_transforms=image_transforms,
            default_tolerance_s=default_tolerance_s,
            shared_dataset_cache=shared_dataset_cache,
        )
        for source_index, (source_cfg, source_meta, source_delta_timestamps) in enumerate(
            zip(sources, metadata_by_source, delta_timestamps_by_source, strict=True)
        )
    ]
    return LegacyMixedLeRobotDataset(
        logical_repo_id=logical_repo_id,
        mix_path=mix_path,
        sources=logical_sources,
    )


__all__ = [
    "LegacyLogicalSource",
    "LegacyMixedLeRobotDataset",
    "load_dataset_mix_config",
    "make_legacy_mixed_dataset",
    "validate_legacy_mix_config",
]
