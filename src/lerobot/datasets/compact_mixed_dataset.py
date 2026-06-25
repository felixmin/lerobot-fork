from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mixed_dataset import (
    DatasetMixSourceConfig,
    MixedLeRobotDatasetMetadata,
    MixedSourceMetadata,
    _aggregate_selected_stats,
    _build_source_index,
    _feature_signature,
    _filter_top_level_dict,
    _infer_required_lookahead_frames,
    _normalize_visual_target_size,
    _normalize_weights,
    _remap_output_key,
    _remap_top_level_dict,
    _resize_visual_tensor,
    _selected_episodes,
    _stats_signature,
    build_explicit_mixed_stats,
)

logger = logging.getLogger(__name__)


def _default_max_open_datasets_per_worker() -> int:
    raw = os.environ.get("HLRP_MIXED_MAX_OPEN_DATASETS_PER_WORKER", "").strip()
    if not raw:
        return 2
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            "HLRP_MIXED_MAX_OPEN_DATASETS_PER_WORKER must be an integer when set."
        ) from exc


def _resolve_max_open_datasets_per_worker(
    *,
    num_sources: int,
    max_sources_per_batch: int | None,
    configured: int | None,
) -> int:
    if configured is not None:
        return max(1, int(configured))

    env_default = _default_max_open_datasets_per_worker()
    batch_cap = None if max_sources_per_batch in {None, 0} else int(max_sources_per_batch)
    heuristic_default = (
        min(int(num_sources), 4)
        if batch_cap is None
        else min(int(num_sources), max(2, batch_cap * 2))
    )
    return max(env_default, heuristic_default)


def _compact_profile_enabled() -> bool:
    return os.environ.get("HLRP_COMPACT_PROFILE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _compact_profile_every() -> int:
    raw = os.environ.get("HLRP_COMPACT_PROFILE_EVERY", "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ValueError(
            "HLRP_COMPACT_PROFILE_EVERY must be an integer when set."
        ) from exc


def _compact_profile_prefix() -> str:
    worker = torch.utils.data.get_worker_info()
    worker_id = -1 if worker is None else int(worker.id)
    return f"[compact-profile pid={os.getpid()} worker={worker_id}]"


def _compact_profile_log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class CompactManifest:
    source_sample_offsets: np.ndarray
    source_effective_lengths: np.ndarray
    source_cumulative_effective_lengths: tuple[np.ndarray, ...]
    total_samples: int


@dataclass(frozen=True)
class CompiledSourceIndex:
    episode_index: np.ndarray
    dataset_from_index: np.ndarray
    dataset_to_index: np.ndarray
    valid_anchor_start: np.ndarray
    valid_anchor_end: np.ndarray
    valid_anchor_count: np.ndarray
    sampleable_episode_ids: np.ndarray
    sampleable_episode_weights: np.ndarray


@dataclass(frozen=True)
class SampleToken:
    source_id: int
    episode_id: int
    anchor_abs_index: int


@dataclass
class CompactSourceRuntime:
    dataset: LeRobotDataset
    compiled_index: CompiledSourceIndex
    open_count: int = 1


class WeightedSampleIndexSampler(torch.utils.data.Sampler[SampleToken]):
    def __init__(
        self,
        *,
        dataset: CompactMixedDataset,
        source_weights: np.ndarray,
        num_samples: int,
        seed: int,
        drop_n_last_frames: int = 0,
        batch_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.source_weights = _normalize_weights(source_weights)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.drop_n_last_frames = int(drop_n_last_frames)
        self.batch_size = max(1, int(batch_size))
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _build_source_states(
        self,
        rng: np.random.Generator,
        compiled_indices: Sequence[CompiledSourceIndex],
    ) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        for source, compiled in zip(self.dataset.sources, compiled_indices, strict=True):
            sampleable = np.asarray(compiled.sampleable_episode_ids, dtype=np.int32)
            if sampleable.size == 0:
                raise ValueError(f"Source {source.repo_id!r} has no sampleable episodes")
            order = np.asarray(sampleable[rng.permutation(sampleable.size)], dtype=np.int32)
            row_by_episode = {
                int(ep_id): idx
                for idx, ep_id in enumerate(compiled.episode_index.tolist())
            }
            states.append(
                {
                    "order": order,
                    "pointer": 0,
                    "row_by_episode": row_by_episode,
                }
            )
        return states

    def _next_episode_id(
        self,
        *,
        source_id: int,
        source_state: dict[str, Any],
        rng: np.random.Generator,
    ) -> int:
        del source_id
        if int(source_state["pointer"]) >= int(source_state["order"].shape[0]):
            source_state["order"] = np.asarray(
                source_state["order"][rng.permutation(source_state["order"].shape[0])],
                dtype=np.int32,
            )
            source_state["pointer"] = 0
        episode_id = int(source_state["order"][source_state["pointer"]])
        source_state["pointer"] = int(source_state["pointer"]) + 1
        return episode_id

    def _sample_source_id(
        self,
        *,
        rng: np.random.Generator,
        used_sources: set[int],
        source_weights: np.ndarray,
    ) -> int:
        if (
            self.dataset.max_sources_per_batch is not None
            and len(used_sources) >= int(self.dataset.max_sources_per_batch)
        ):
            source_ids = np.asarray(sorted(used_sources), dtype=np.int64)
            local_weights = source_weights[source_ids]
            local_weights = _normalize_weights(local_weights)
            return int(rng.choice(source_ids, p=local_weights))
        return int(rng.choice(len(self.dataset.sources), p=source_weights))

    def _sample_token_from_source(
        self,
        *,
        source_id: int,
        source_state: dict[str, Any],
        rng: np.random.Generator,
        compiled: CompiledSourceIndex,
    ) -> SampleToken:
        source = self.dataset.sources[source_id]
        episode_id = self._next_episode_id(
            source_id=source_id,
            source_state=source_state,
            rng=rng,
        )
        row_idx = source_state["row_by_episode"][episode_id]
        start = int(compiled.valid_anchor_start[row_idx])
        end = int(compiled.valid_anchor_end[row_idx])
        if end <= start:
            raise ValueError(
                f"Episode {episode_id} of source {source.repo_id!r} has no valid anchors: "
                f"[{start}, {end})"
            )
        anchor = int(rng.integers(start, end))
        return SampleToken(
            source_id=int(source_id),
            episode_id=int(episode_id),
            anchor_abs_index=anchor,
        )

    def __iter__(self) -> Iterator[SampleToken]:
        if self.num_samples <= 0:
            return

        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        compiled_indices = [
            source.compiled_index_for_drop(self.drop_n_last_frames)
            for source in self.dataset.sources
        ]
        source_weights = self.source_weights.copy()
        for source_id, compiled in enumerate(compiled_indices):
            if compiled.sampleable_episode_ids.size == 0:
                source_weights[source_id] = 0.0
        source_weights = _normalize_weights(source_weights)

        source_states = self._build_source_states(rng, compiled_indices)
        produced = 0
        while produced < self.num_samples:
            used_sources: set[int] = set()
            batch_count = min(self.batch_size, self.num_samples - produced)
            for _ in range(batch_count):
                source_id = self._sample_source_id(
                    rng=rng,
                    used_sources=used_sources,
                    source_weights=source_weights,
                )
                token = self._sample_token_from_source(
                    source_id=source_id,
                    source_state=source_states[source_id],
                    rng=rng,
                    compiled=compiled_indices[source_id],
                )
                used_sources.add(source_id)
                yield token
                produced += 1


class CompactSourceAdapter:
    def __init__(
        self,
        *,
        source_index: int,
        config: DatasetMixSourceConfig,
        meta: LeRobotDatasetMetadata,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Any = None,
        default_tolerance_s: float = 1e-4,
        retained_features: tuple[str, ...] | None = None,
        visual_target_size: tuple[int, int] | list[int] | None = None,
    ) -> None:
        self.source_index = int(source_index)
        self.config = config
        self.meta = meta
        self.feature_key_mapping = deepcopy(config.feature_key_mapping or {})
        self.retained_features = retained_features
        self.raw_delta_timestamps = deepcopy(delta_timestamps)
        self.delta_timestamps = _remap_top_level_dict(
            delta_timestamps or {},
            self.feature_key_mapping,
            source_name=config.name,
            field_name="delta_timestamps",
        )
        if delta_timestamps is None:
            self.delta_timestamps = None
        self.image_transforms = image_transforms
        self.visual_target_size = _normalize_visual_target_size(visual_target_size)
        self.selected_episodes = _selected_episodes(config, meta.total_episodes)
        self.index = _build_source_index(meta, self.selected_episodes)
        unknown_feature_keys = sorted(
            key for key in self.feature_key_mapping if key not in meta.features
        )
        if unknown_feature_keys:
            raise ValueError(
                f"Feature remapping for source '{config.name}' references unknown feature keys: "
                f"{unknown_feature_keys}"
            )

        remapped_features = _remap_top_level_dict(
            meta.features,
            self.feature_key_mapping,
            source_name=config.name,
            field_name="features",
        )
        remapped_stats = _remap_top_level_dict(
            _aggregate_selected_stats(meta, self.selected_episodes),
            self.feature_key_mapping,
            source_name=config.name,
            field_name="stats",
        )
        if self.retained_features is not None:
            missing_retained = sorted(
                key for key in self.retained_features if key not in remapped_features
            )
            if missing_retained:
                raise ValueError(
                    f"Source '{config.name}' is missing retained features "
                    f"{missing_retained} after remapping."
                )
        self._all_feature_keys = set(remapped_features)
        self.features = _filter_top_level_dict(remapped_features, self.retained_features)
        self.stats = _filter_top_level_dict(remapped_stats, self.retained_features)
        self.delta_timestamps = _filter_top_level_dict(
            self.delta_timestamps or {},
            self.retained_features,
        )
        if self.retained_features is not None:
            inverse_mapping = {
                target_key: source_key
                for source_key, target_key in self.feature_key_mapping.items()
            }
            raw_retained_features = tuple(
                inverse_mapping.get(feature_key, feature_key)
                for feature_key in self.retained_features
            )
        else:
            raw_retained_features = None
        self.raw_delta_timestamps = _filter_top_level_dict(
            self.raw_delta_timestamps or {},
            raw_retained_features,
        )
        if delta_timestamps is None or len(self.delta_timestamps) == 0:
            self.delta_timestamps = None
        if delta_timestamps is None or len(self.raw_delta_timestamps) == 0:
            self.raw_delta_timestamps = None
        self.required_lookahead_frames = _infer_required_lookahead_frames(
            self.raw_delta_timestamps,
            fps=self.meta.fps,
        )
        self.tolerance_s = float(
            default_tolerance_s if config.tolerance_s is None else config.tolerance_s
        )
        self.compiled_index = self._compile_index(drop_n_last_frames=0)
        self._profile_fetch_many_calls = 0

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

    def metadata(self, *, effective_num_frames: int | None = None) -> MixedSourceMetadata:
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
            num_frames=(
                self.num_frames
                if effective_num_frames is None
                else int(effective_num_frames)
            ),
            feature_key_mapping=deepcopy(self.feature_key_mapping),
            retained_features=self.retained_features,
        )

    def _compile_index(self, *, drop_n_last_frames: int) -> CompiledSourceIndex:
        total_drop_n_last_frames = int(drop_n_last_frames) + int(
            self.required_lookahead_frames
        )
        valid_anchor_start = self.index.dataset_from_index.copy()
        valid_anchor_end = np.maximum(
            self.index.dataset_to_index - total_drop_n_last_frames,
            valid_anchor_start,
        )
        valid_anchor_count = np.maximum(valid_anchor_end - valid_anchor_start, 0)
        sampleable_mask = valid_anchor_count > 0
        return CompiledSourceIndex(
            episode_index=self.index.episode_indices.copy(),
            dataset_from_index=self.index.dataset_from_index.copy(),
            dataset_to_index=self.index.dataset_to_index.copy(),
            valid_anchor_start=valid_anchor_start,
            valid_anchor_end=valid_anchor_end,
            valid_anchor_count=valid_anchor_count,
            sampleable_episode_ids=self.index.episode_indices[sampleable_mask].copy(),
            sampleable_episode_weights=valid_anchor_count[sampleable_mask].astype(
                np.float64
            ),
        )

    def get_effective_lengths(self, drop_n_last_frames: int = 0) -> np.ndarray:
        return self.compiled_index_for_drop(drop_n_last_frames).valid_anchor_count.copy()

    def compiled_index_for_drop(
        self, drop_n_last_frames: int = 0
    ) -> CompiledSourceIndex:
        if int(drop_n_last_frames) == 0:
            return self.compiled_index
        return self._compile_index(drop_n_last_frames=int(drop_n_last_frames))

    def flat_index_to_anchor(self, index: int, *, drop_n_last_frames: int = 0) -> int:
        compiled = self.compiled_index_for_drop(drop_n_last_frames)
        total_effective = int(compiled.valid_anchor_count.sum())
        if index < 0 or index >= total_effective:
            raise IndexError(f"Index {index} out of bounds for source '{self.name}'.")

        cumulative_lengths = np.cumsum(compiled.valid_anchor_count, dtype=np.int64)
        episode_pos = int(np.searchsorted(cumulative_lengths, index, side="right"))
        prev_total = 0 if episode_pos == 0 else int(cumulative_lengths[episode_pos - 1])
        return int(compiled.valid_anchor_start[episode_pos] + (index - prev_total))

    def make_dataset(self) -> LeRobotDataset:
        # Keep compact sources pinned to absolute anchors over the full physical
        # dataset. Opening a per-source filtered LeRobotDataset is dramatically
        # slower on large parquet-backed caches and defeats the compact path's
        # intent of cheap logical splits over shared storage.
        return LeRobotDataset(
            self.repo_id,
            root=self.meta.root,
            episodes=None,
            image_transforms=self.image_transforms,
            delta_timestamps=self.raw_delta_timestamps,
            revision=self.revision,
            video_backend=self.config.video_backend,
            tolerance_s=self.tolerance_s,
        )

    def _adapt_item(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.feature_key_mapping:
            remapped_item: dict[str, Any] = {}
            for key, value in item.items():
                target_key = _remap_output_key(key, self.feature_key_mapping)
                if target_key in remapped_item:
                    raise ValueError(
                        f"Feature remapping for source '{self.name}' causes duplicate item key "
                        f"{target_key!r}."
                    )
                remapped_item[target_key] = value
            item = remapped_item

        if self.retained_features is not None:
            retained = set(self.retained_features)
            filtered_item: dict[str, Any] = {}
            for key, value in item.items():
                base_key = key[: -len("_is_pad")] if key.endswith("_is_pad") else key
                if base_key in self._all_feature_keys and base_key not in retained:
                    continue
                filtered_item[key] = value
            item = filtered_item

        if self.visual_target_size is not None:
            for key, value in list(item.items()):
                feature = self.features.get(key)
                if feature is None or feature.get("dtype") not in {"image", "video"}:
                    continue
                if not isinstance(value, torch.Tensor):
                    continue
                item[key] = _resize_visual_tensor(value, self.visual_target_size)

        for key, feature in self.features.items():
            if key not in item:
                continue
            pad_key = f"{key}_is_pad"
            if pad_key in item:
                continue
            value = item[key]
            if not isinstance(value, torch.Tensor):
                continue
            if feature.get("dtype") in {"image", "video"}:
                time_steps = 1 if value.ndim == 3 else int(value.shape[0])
            else:
                time_steps = 1 if value.ndim <= 1 else int(value.shape[0])
            item[pad_key] = torch.zeros((time_steps,), dtype=torch.bool)

        item["action_supervision"] = torch.tensor(
            self.action_supervision, dtype=torch.bool
        )
        item["latent_supervision"] = torch.tensor(
            self.latent_supervision, dtype=torch.bool
        )
        item["dataset_source_index"] = torch.tensor(
            self.source_index, dtype=torch.int64
        )
        item["dataset_source_name"] = self.name
        item["dataset_source_repo_id"] = self.repo_id
        item["dataset_source_root"] = "" if self.root is None else self.root
        item["dataset_source_revision"] = "" if self.revision is None else self.revision
        return item

    def fetch_one(
        self,
        dataset: LeRobotDataset,
        *,
        anchor_abs_index: int,
    ) -> dict[str, Any]:
        if dataset.absolute_to_relative_idx is None:
            relative_index = int(anchor_abs_index)
        else:
            relative_index = int(dataset.absolute_to_relative_idx[int(anchor_abs_index)])
        return self._adapt_item(dataset[relative_index])

    def fetch_many(
        self,
        dataset: LeRobotDataset,
        *,
        anchor_abs_indices: Sequence[int],
    ) -> list[dict[str, Any]]:
        profile = _compact_profile_enabled()
        profile_every = _compact_profile_every() if profile else 0
        self._profile_fetch_many_calls += 1
        profile_this_call = profile and (
            self._profile_fetch_many_calls % profile_every == 0
        )
        started = time.perf_counter() if profile_this_call else 0.0
        if len(anchor_abs_indices) == 0:
            return []
        if len(anchor_abs_indices) == 1:
            return [self.fetch_one(dataset, anchor_abs_index=int(anchor_abs_indices[0]))]

        t_before_ensure = time.perf_counter() if profile_this_call else 0.0
        if dataset.reader.hf_dataset is None:
            dataset.reader.load_and_activate()
        t_after_ensure = time.perf_counter() if profile_this_call else 0.0
        records: list[dict[str, Any]] = []
        relative_indices: list[int] = []
        for anchor_abs_index in anchor_abs_indices:
            if dataset.absolute_to_relative_idx is None:
                relative_index = int(anchor_abs_index)
            else:
                relative_index = int(dataset.absolute_to_relative_idx[int(anchor_abs_index)])
            relative_indices.append(relative_index)

            item = dict(dataset.hf_dataset[relative_index])
            ep_idx = int(item["episode_index"].item())
            abs_idx = int(item["index"].item())
            query_indices = None
            padding: dict[str, torch.Tensor] = {}
            if dataset.reader.delta_indices is not None:
                query_indices, padding = dataset.reader._get_query_indices(abs_idx, ep_idx)
            records.append(
                {
                    "item": item,
                    "ep_idx": ep_idx,
                    "abs_idx": abs_idx,
                    "query_indices": query_indices,
                    "padding": padding,
                    "query_timestamps": dataset.reader._get_query_timestamps(
                        float(item["timestamp"].item()),
                        query_indices,
                    ),
                }
            )
        t_after_plan = time.perf_counter() if profile_this_call else 0.0

        if dataset.reader.delta_indices is not None:
            non_video_query_indices: dict[str, list[int]] = {}
            for record in records:
                query_indices = record["query_indices"]
                if query_indices is None:
                    continue
                for key, values in query_indices.items():
                    if key in dataset.meta.video_keys:
                        continue
                    non_video_query_indices.setdefault(key, []).extend(values)

            if non_video_query_indices:
                non_video_results = dataset.reader._query_hf_dataset(non_video_query_indices)
                key_offsets = {key: 0 for key in non_video_query_indices}
                for record in records:
                    query_indices = record["query_indices"]
                    if query_indices is None:
                        continue
                    for key, values in query_indices.items():
                        if key in dataset.meta.video_keys:
                            continue
                        start = key_offsets[key]
                        stop = start + len(values)
                        record["item"][key] = non_video_results[key][start:stop]
                        key_offsets[key] = stop
                    record["item"].update(record["padding"])
        t_after_nonvisual = time.perf_counter() if profile_this_call else 0.0

        if dataset.meta.video_keys:
            video_requests: dict[int, dict[str, list[tuple[int, int]]]] = {}
            video_timestamps: dict[int, dict[str, list[float]]] = {}
            for record_index, record in enumerate(records):
                ep_group = video_requests.setdefault(record["ep_idx"], {})
                ts_group = video_timestamps.setdefault(record["ep_idx"], {})
                for key, timestamps in record["query_timestamps"].items():
                    ep_group.setdefault(key, []).append((record_index, len(timestamps)))
                    ts_group.setdefault(key, []).extend(timestamps)

            for ep_idx, timestamps_by_key in video_timestamps.items():
                decoded_by_key = dataset.reader._query_videos(timestamps_by_key, ep_idx)
                for key, requests in video_requests[ep_idx].items():
                    start = 0
                    decoded = decoded_by_key[key]
                    for record_index, count in requests:
                        stop = start + count
                        records[record_index]["item"][key] = decoded[start:stop].squeeze(0)
                        start = stop
        t_after_video = time.perf_counter() if profile_this_call else 0.0

        items: list[dict[str, Any]] = []
        for record in records:
            item = record["item"]
            if dataset.image_transforms is not None:
                for cam in dataset.meta.camera_keys:
                    item[cam] = dataset.image_transforms(item[cam])

            task_idx = item["task_index"].item()
            item["task"] = dataset.meta.tasks.iloc[task_idx].name
            if "subtask_index" in dataset.features and dataset.meta.subtasks is not None:
                subtask_idx = item["subtask_index"].item()
                item["subtask"] = dataset.meta.subtasks.iloc[subtask_idx].name
            items.append(self._adapt_item(item))
        t_after_assemble = time.perf_counter() if profile_this_call else 0.0

        if profile_this_call:
            unique_eps = len({int(record["ep_idx"]) for record in records})
            query_sizes = [
                sum(len(ts) for ts in record["query_timestamps"].values())
                for record in records
            ]
            _compact_profile_log(
                (
                    f"{_compact_profile_prefix()} source={self.name} anchors={len(anchor_abs_indices)} "
                    f"unique_eps={unique_eps} rel_idx_span="
                    f"{f'{min(relative_indices)}..{max(relative_indices)}' if relative_indices else 'NA'} "
                    f"ensure_s={t_after_ensure - t_before_ensure:.3f} "
                    f"plan_s={t_after_plan - t_after_ensure:.3f} "
                    f"nonvisual_s={t_after_nonvisual - t_after_plan:.3f} "
                    f"video_s={t_after_video - t_after_nonvisual:.3f} "
                    f"assemble_s={t_after_assemble - t_after_video:.3f} "
                    f"total_s={t_after_assemble - started:.3f} "
                    f"avg_queries_per_item={0.0 if len(query_sizes) == 0 else float(np.mean(query_sizes)):.1f}"
                )
            )

        return items


def validate_compact_sources(
    sources: list[CompactSourceAdapter],
    *,
    enforce_matching_fps: bool = True,
    enforce_matching_delta_timestamps: bool = True,
    allow_visual_shape_mismatch: bool = False,
) -> None:
    if len(sources) == 0:
        raise ValueError("Expected at least one logical source.")

    base_source = sources[0]
    base_feature_signature = _feature_signature(
        base_source.features,
        allow_visual_shape_mismatch=allow_visual_shape_mismatch,
    )
    base_fps = base_source.meta.fps
    base_delta_timestamps = base_source.delta_timestamps
    base_camera_keys = base_source.camera_keys
    base_stats_signature = _stats_signature(
        base_source.stats,
        features=base_source.features,
        allow_visual_shape_mismatch=allow_visual_shape_mismatch,
    )

    seen_episodes: dict[tuple[str, str | None, str | None], set[int]] = {}
    for source in sources:
        if source.num_frames <= 0:
            raise ValueError(f"Mix source '{source.name}' resolved to zero frames.")
        if enforce_matching_fps and source.meta.fps != base_fps:
            raise ValueError("All mix sources must have matching fps.")
        if (
            _feature_signature(
                source.features,
                allow_visual_shape_mismatch=allow_visual_shape_mismatch,
            )
            != base_feature_signature
        ):
            raise ValueError("All mix sources must have identical feature schemas.")
        if source.camera_keys != base_camera_keys:
            raise ValueError("All mix sources must expose the same camera keys.")
        if (
            enforce_matching_delta_timestamps
            and source.delta_timestamps != base_delta_timestamps
        ):
            raise ValueError(
                "All mix sources must resolve to identical delta timestamps."
            )
        if (
            _stats_signature(
                source.stats,
                features=source.features,
                allow_visual_shape_mismatch=allow_visual_shape_mismatch,
            )
            != base_stats_signature
        ):
            raise ValueError(
                "All mix sources must expose compatible normalization-stat schemas."
            )

        seen = seen_episodes.setdefault(source.dataset_identity, set())
        overlap = seen.intersection(source.selected_episodes)
        if overlap:
            overlap_text = ", ".join(str(episode) for episode in sorted(overlap))
            raise ValueError(
                f"Mix sources cannot overlap on the same physical dataset. "
                f"Source '{source.name}' overlaps on episodes [{overlap_text}]."
            )
        seen.update(source.selected_episodes)


def _build_manifest(
    sources: Sequence[CompactSourceAdapter],
) -> CompactManifest:
    source_lengths = np.asarray(
        [int(source.get_effective_lengths(0).sum()) for source in sources],
        dtype=np.int64,
    )
    source_sample_offsets = np.zeros((len(sources),), dtype=np.int64)
    if len(sources) > 1:
        source_sample_offsets[1:] = np.cumsum(source_lengths[:-1], dtype=np.int64)
    source_episode_offsets = tuple(
        np.cumsum(source.get_effective_lengths(0), dtype=np.int64) for source in sources
    )
    total_samples = int(source_lengths.sum())
    return CompactManifest(
        source_sample_offsets=source_sample_offsets,
        source_effective_lengths=source_lengths,
        source_cumulative_effective_lengths=source_episode_offsets,
        total_samples=total_samples,
    )


def _build_compact_virtual_episodes(
    sources: Sequence[CompactSourceAdapter],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = 0
    logical_episode_index = 0
    for source in sources:
        effective_lengths = source.get_effective_lengths(0)
        for source_episode_index, effective_length in zip(
            source.selected_episodes,
            effective_lengths,
            strict=True,
        ):
            if int(effective_length) <= 0:
                continue
            rows.append(
                {
                    "episode_index": logical_episode_index,
                    "dataset_from_index": cursor,
                    "dataset_to_index": cursor + int(effective_length),
                    "source_name": source.name,
                    "source_repo_id": source.repo_id,
                    "source_episode_index": int(source_episode_index),
                }
            )
            cursor += int(effective_length)
            logical_episode_index += 1
    return pd.DataFrame(rows)


def _build_compact_mixed_info(
    logical_repo_id: str,
    mix_path: Path,
    sources: Sequence[CompactSourceAdapter],
    manifest: CompactManifest,
) -> dict[str, Any]:
    # Upstream's meta.info is a DatasetInfo dataclass that rejects unknown keys; work on a plain dict
    # so we can attach mix-specific metadata (mixed_sources, mix_path, ...).
    base_info = sources[0].meta.info
    info = deepcopy(base_info.to_dict() if hasattr(base_info, "to_dict") else dict(base_info))
    source_metadata: list[dict[str, Any]] = []
    total_effective_episodes = 0
    for source in sources:
        effective_num_frames = int(source.get_effective_lengths(0).sum())
        effective_num_episodes = int(np.count_nonzero(source.get_effective_lengths(0)))
        total_effective_episodes += effective_num_episodes
        source_metadata.append(
            {
                "name": source.name,
                "repo_id": source.repo_id,
                "root": source.root,
                "revision": source.revision,
                "weight": source.weight,
                "action_supervision": source.action_supervision,
                "latent_supervision": source.latent_supervision,
                "episodes": list(source.selected_episodes),
                "video_backend": source.config.video_backend,
                "tolerance_s": source.tolerance_s,
                "num_frames": effective_num_frames,
                "raw_num_frames": source.num_frames,
                "num_episodes": effective_num_episodes,
                "raw_num_episodes": source.num_episodes,
                "feature_key_mapping": deepcopy(source.feature_key_mapping),
                "retained_features": (
                    None
                    if source.retained_features is None
                    else list(source.retained_features)
                ),
            }
        )
    info["total_episodes"] = total_effective_episodes
    info["total_frames"] = int(manifest.total_samples)
    info["mixed_sources"] = source_metadata
    info["mix_path"] = str(mix_path)
    info["logical_repo_id"] = logical_repo_id
    return info


class CompactMixedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        logical_repo_id: str,
        mix_path: str | Path,
        sources: list[CompactSourceAdapter],
        enforce_matching_fps: bool = True,
        enforce_matching_delta_timestamps: bool = True,
        allow_visual_shape_mismatch: bool = False,
        max_open_datasets_per_worker: int | None = None,
        max_sources_per_batch: int | None = None,
    ) -> None:
        super().__init__()
        validate_compact_sources(
            sources,
            enforce_matching_fps=enforce_matching_fps,
            enforce_matching_delta_timestamps=enforce_matching_delta_timestamps,
            allow_visual_shape_mismatch=allow_visual_shape_mismatch,
        )
        self.repo_id = logical_repo_id
        self.mix_path = str(Path(mix_path).expanduser().resolve())
        self.sources = list(sources)
        self.source_weights = np.asarray(
            [source.weight for source in self.sources], dtype=np.float64
        )
        self.episodes = None
        self.manifest = _build_manifest(self.sources)
        self.max_sources_per_batch = (
            None
            if max_sources_per_batch in {None, 0}
            else max(1, int(max_sources_per_batch))
        )
        self.max_open_datasets_per_worker = _resolve_max_open_datasets_per_worker(
            num_sources=len(self.sources),
            max_sources_per_batch=self.max_sources_per_batch,
            configured=max_open_datasets_per_worker,
        )
        self._source_runtimes: OrderedDict[int, CompactSourceRuntime] = OrderedDict()
        self._profile_batch_calls = 0
        virtual_episodes = _build_compact_virtual_episodes(self.sources)

        self.meta = MixedLeRobotDatasetMetadata(
            repo_id=self.repo_id,
            mix_path=self.mix_path,
            fps=int(self.sources[0].meta.fps),
            features=deepcopy(self.sources[0].features),
            stats=build_explicit_mixed_stats(self.sources),
            episodes=virtual_episodes,
            source_metadata=tuple(
                source.metadata(
                    effective_num_frames=int(source.get_effective_lengths(0).sum())
                )
                for source in self.sources
            ),
            info=_build_compact_mixed_info(
                self.repo_id,
                Path(self.mix_path),
                self.sources,
                self.manifest,
            ),
        )

    @property
    def num_frames(self) -> int:
        return self.manifest.total_samples

    @property
    def num_episodes(self) -> int:
        return int(len(self.meta.episodes))

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return self.meta.features

    def loader_hints(self) -> dict[str, Any]:
        return {
            "is_mixed": True,
            "prefetch_factor": 1,
            "sampler_mode": "sample_level",
            "mixed_impl": "compact_manifest",
            "pass_batch_size_to_sampler": True,
        }

    def __len__(self) -> int:
        return self.num_frames

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_source_runtimes"] = OrderedDict()
        return state

    def get_effective_source_lengths(
        self, drop_n_last_frames: int = 0
    ) -> list[np.ndarray]:
        return [
            source.get_effective_lengths(drop_n_last_frames) for source in self.sources
        ]

    def _resolve_sample_index(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds.")
        source_index = int(
            np.searchsorted(
                self.manifest.source_sample_offsets
                + self.manifest.source_effective_lengths,
                int(index),
                side="right",
            )
        )
        source_offset = int(self.manifest.source_sample_offsets[source_index])
        source_local_index = int(index) - source_offset
        return source_index, source_local_index

    def _get_source_runtime(self, source_index: int) -> CompactSourceRuntime:
        profile = _compact_profile_enabled()
        runtime = self._source_runtimes.pop(source_index, None)
        if runtime is None:
            source = self.sources[source_index]
            runtime = CompactSourceRuntime(
                dataset=source.make_dataset(),
                compiled_index=source.compiled_index,
            )
            if profile:
                _compact_profile_log(
                    (
                        f"{_compact_profile_prefix()} runtime_open source={source.name} "
                        f"cache_size_before={len(self._source_runtimes)} "
                        f"max_open={self.max_open_datasets_per_worker}"
                    )
                )
        self._source_runtimes[source_index] = runtime
        while len(self._source_runtimes) > self.max_open_datasets_per_worker:
            evicted_index, _ = self._source_runtimes.popitem(last=False)
            if profile:
                _compact_profile_log(
                    (
                        f"{_compact_profile_prefix()} runtime_evict "
                        f"source={self.sources[int(evicted_index)].name} "
                        f"cache_size_after={len(self._source_runtimes)}"
                    )
                )
        return runtime

    def _fetch_token(self, token: SampleToken) -> dict[str, Any]:
        source_index = int(token.source_id)
        source = self.sources[source_index]
        runtime = self._get_source_runtime(source_index)
        return source.fetch_one(
            runtime.dataset,
            anchor_abs_index=int(token.anchor_abs_index),
        )

    def _fetch_one(self, index: int | SampleToken) -> dict[str, Any]:
        if isinstance(index, SampleToken):
            return self._fetch_token(index)
        source_index, source_local_index = self._resolve_sample_index(int(index))
        source = self.sources[source_index]
        runtime = self._get_source_runtime(source_index)
        anchor_abs_index = source.flat_index_to_anchor(source_local_index)
        return source.fetch_one(runtime.dataset, anchor_abs_index=anchor_abs_index)

    def __getitem__(self, index: int | SampleToken) -> dict[str, Any]:
        return self._fetch_one(index)

    def __getitems__(self, indices: Sequence[int | SampleToken]) -> list[dict[str, Any]]:
        profile = _compact_profile_enabled()
        profile_every = _compact_profile_every() if profile else 0
        self._profile_batch_calls += 1
        profile_this_call = profile and (
            self._profile_batch_calls % profile_every == 0
        )
        started = time.perf_counter() if profile_this_call else 0.0
        grouped_requests: dict[int, list[tuple[int, int | SampleToken]]] = {}
        ordered: list[dict[str, Any] | None] = [None] * len(indices)
        for batch_pos, sample_index in enumerate(indices):
            if isinstance(sample_index, SampleToken):
                grouped_requests.setdefault(int(sample_index.source_id), []).append(
                    (batch_pos, sample_index)
                )
            else:
                source_index, source_local_index = self._resolve_sample_index(
                    int(sample_index)
                )
                grouped_requests.setdefault(source_index, []).append(
                    (batch_pos, source_local_index)
                )
        grouped_at = time.perf_counter() if profile_this_call else 0.0

        per_source_stats: list[str] = []
        for source_index, requests in grouped_requests.items():
            source = self.sources[source_index]
            source_started = time.perf_counter() if profile_this_call else 0.0
            runtime = self._get_source_runtime(source_index)
            anchors: list[int] = []
            for _, request_payload in requests:
                if isinstance(request_payload, SampleToken):
                    anchors.append(int(request_payload.anchor_abs_index))
                else:
                    anchors.append(source.flat_index_to_anchor(int(request_payload)))
            anchors_ready = time.perf_counter() if profile_this_call else 0.0
            items = source.fetch_many(runtime.dataset, anchor_abs_indices=anchors)
            fetched_at = time.perf_counter() if profile_this_call else 0.0
            for (batch_pos, _), item in zip(requests, items, strict=True):
                ordered[batch_pos] = item
            if profile_this_call:
                per_source_stats.append(
                    (
                        f"{source.name}:n={len(requests)} prep={anchors_ready - source_started:.3f}s "
                        f"fetch={fetched_at - anchors_ready:.3f}s"
                    )
                )

        if any(item is None for item in ordered):
            raise RuntimeError(
                "CompactMixedDataset.__getitems__ failed to populate every requested index."
            )
        if profile_this_call:
            finished = time.perf_counter()
            _compact_profile_log(
                (
                    f"{_compact_profile_prefix()} batch={self._profile_batch_calls} "
                    f"size={len(indices)} groups={len(grouped_requests)} "
                    f"group_s={grouped_at - started:.3f} total_s={finished - started:.3f} "
                    f"cache={[self.sources[idx].name for idx in self._source_runtimes.keys()]} "
                    f"per_source=[{'; '.join(per_source_stats)}]"
                )
            )
        return [item for item in ordered if item is not None]

    def build_sampler(
        self,
        *,
        seed: int | None = None,
        drop_n_last_frames: int = 0,
        source_block_size: int = 1,
        batch_size: int = 1,
    ) -> WeightedSampleIndexSampler:
        if int(source_block_size) != 1:
            raise ValueError(
                "CompactMixedDataset keeps sample-level mixing and does not support source_block_size."
            )
        num_samples = int(
            sum(
                source.get_effective_lengths(drop_n_last_frames).sum()
                for source in self.sources
            )
        )
        if num_samples <= 0:
            raise ValueError(
                "Mix dataset has no effective samples after drop_n_last_frames."
            )
        return WeightedSampleIndexSampler(
            dataset=self,
            source_weights=self.source_weights,
            num_samples=num_samples,
            seed=0 if seed is None else int(seed),
            drop_n_last_frames=drop_n_last_frames,
            batch_size=batch_size,
        )


CompactMixedLeRobotDataset = CompactMixedDataset
