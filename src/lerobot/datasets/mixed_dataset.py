from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import WeightedSourceSampler
from lerobot.datasets.utils import flatten_dict, unflatten_dict
from lerobot.utils.constants import ACTION

VALID_SUPERVISION_MODES = {"latent_only", "multitask"}
logger = logging.getLogger(__name__)


def _debug_mixed_dataset() -> bool:
    value = os.environ.get("HLRP_STAGE3_DEBUG_DATASET", "")
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DatasetMixSourceConfig:
    name: str
    repo_id: str
    root: str | None = None
    revision: str | None = None
    weight: float = 1.0
    episodes: tuple[int, ...] | None = None
    exclude_episodes: tuple[int, ...] | None = None
    supervision: str = "multitask"
    video_backend: str | None = None
    tolerance_s: float | None = None
    camera_keys: tuple[str, ...] | None = None
    camera_map: dict[str, str] | None = None
    action_key: str | None = None
    filtering: dict[str, Any] | None = None
    feature_key_mapping: dict[str, str] | None = None


@dataclass(frozen=True)
class DatasetMixConfig:
    path: Path
    sources: tuple[DatasetMixSourceConfig, ...]
    retained_features: tuple[str, ...] | None = None
    enforce_matching_fps: bool = True
    enforce_matching_delta_timestamps: bool = True
    allow_visual_shape_mismatch: bool = False


@dataclass(frozen=True)
class SourceIndex:
    episode_indices: np.ndarray
    dataset_from_index: np.ndarray
    dataset_to_index: np.ndarray
    lengths: np.ndarray
    cumulative_lengths: np.ndarray


@dataclass(frozen=True)
class MixedSourceMetadata:
    name: str
    repo_id: str
    root: str | None
    revision: str | None
    weight: float
    supervision: str
    episodes: tuple[int, ...]
    video_backend: str | None
    tolerance_s: float
    num_frames: int
    filter_cache_path: str | None = None
    filter_summary: dict[str, Any] | None = None
    feature_key_mapping: dict[str, str]
    retained_features: tuple[str, ...] | None


@dataclass
class MixedLeRobotDatasetMetadata:
    repo_id: str
    mix_path: str
    fps: int
    features: dict[str, dict[str, Any]]
    stats: dict[str, dict[str, np.ndarray]]
    episodes: pd.DataFrame
    source_metadata: tuple[MixedSourceMetadata, ...]
    info: dict[str, Any]

    @property
    def camera_keys(self) -> list[str]:
        return [
            key
            for key, feature in self.features.items()
            if feature["dtype"] in {"image", "video"}
        ]

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])


def _parse_episode_list(
    *,
    field_name: str,
    values: list[int] | None,
    total_episodes: int | None = None,
) -> tuple[int, ...] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise TypeError(f"Expected '{field_name}' to be a list of integers.")
    if any(not isinstance(value, int) for value in values):
        raise TypeError(f"Expected '{field_name}' to contain only integers.")
    if len(set(values)) != len(values):
        raise ValueError(f"Expected '{field_name}' to contain unique episode indices.")
    if total_episodes is not None and any(
        value < 0 or value >= total_episodes for value in values
    ):
        raise ValueError(
            f"Episode indices in '{field_name}' are out of range for dataset with {total_episodes} episodes."
        )
    return tuple(sorted(values))


def _parse_string_list(
    *,
    field_name: str,
    values: list[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if not isinstance(values, list):
        raise TypeError(f"Expected '{field_name}' to be a list of strings.")
    parsed: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"Expected '{field_name}' to contain only strings.")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"Expected '{field_name}' to contain non-empty strings.")
        parsed.append(cleaned)
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Expected '{field_name}' to contain unique strings.")
    return tuple(parsed)


def _parse_bool_field(*, field_name: str, value: bool | None, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"Expected '{field_name}' to be a boolean.")
    return value


def _parse_feature_key_mapping(
    *,
    field_name: str,
    value: dict[str, str] | None,
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"Expected '{field_name}' to be a mapping of strings.")

    mapping: dict[str, str] = {}
    for raw_source_key, raw_target_key in value.items():
        if not isinstance(raw_source_key, str) or not isinstance(raw_target_key, str):
            raise TypeError(
                f"Expected '{field_name}' to map string keys to string values."
            )
        source_key = raw_source_key.strip()
        target_key = raw_target_key.strip()
        if not source_key or not target_key:
            raise ValueError(
                f"Expected '{field_name}' to map non-empty strings, got "
                f"{raw_source_key!r}->{raw_target_key!r}."
            )
        mapping[source_key] = target_key

    inverse: dict[str, str] = {}
    for source_key, target_key in mapping.items():
        previous_source = inverse.get(target_key)
        if previous_source is not None and previous_source != source_key:
            raise ValueError(
                f"Expected '{field_name}' to be one-to-one, but target key "
                f"{target_key!r} is mapped from both {previous_source!r} and "
                f"{source_key!r}."
            )
        inverse[target_key] = source_key
    return mapping


def _remap_output_key(key: str, feature_key_mapping: dict[str, str]) -> str:
    if key in feature_key_mapping:
        return feature_key_mapping[key]
    suffix = "_is_pad"
    if key.endswith(suffix):
        base_key = key[: -len(suffix)]
        if base_key in feature_key_mapping:
            return f"{feature_key_mapping[base_key]}{suffix}"
    return key


def _remap_top_level_dict(
    data: dict[str, Any],
    feature_key_mapping: dict[str, str] | None,
    *,
    source_name: str,
    field_name: str,
) -> dict[str, Any]:
    mapping = feature_key_mapping or {}
    remapped: dict[str, Any] = {}
    for key, value in data.items():
        target_key = mapping.get(key, key)
        if target_key in remapped:
            raise ValueError(
                f"Feature remapping for source '{source_name}' causes duplicate "
                f"{field_name} key {target_key!r}."
            )
        remapped[target_key] = deepcopy(value)
    return remapped


def _filter_top_level_dict(
    data: dict[str, Any],
    retained_keys: tuple[str, ...] | None,
) -> dict[str, Any]:
    if retained_keys is None:
        return deepcopy(data)
    retained = set(retained_keys)
    return {key: deepcopy(value) for key, value in data.items() if key in retained}


def _feature_signature(
    features: dict[str, dict[str, Any]],
    *,
    allow_visual_shape_mismatch: bool,
) -> dict[str, dict[str, Any]]:
    signature: dict[str, dict[str, Any]] = {}
    for key, feature in features.items():
        entry = deepcopy(feature)
        if allow_visual_shape_mismatch and entry.get("dtype") in {"image", "video"}:
            entry = {"dtype": "visual"}
        signature[key] = entry
    return signature


def _normalize_visual_target_size(
    value: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("Expected visual target size to have exactly two dimensions.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("Expected visual target size to be strictly positive.")
    return (height, width)


def _resize_visual_tensor(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    if tensor.ndim < 3:
        return tensor

    current_size = tuple(int(dim) for dim in tensor.shape[-2:])
    if current_size == target_size:
        return tensor

    if tensor.shape[-3] not in {1, 3}:
        return tensor

    original_dtype = tensor.dtype
    leading_shape = tuple(int(dim) for dim in tensor.shape[:-3])
    channels = int(tensor.shape[-3])
    work = tensor.reshape(-1, channels, *current_size)
    if not work.is_floating_point():
        work = work.to(torch.float32)
    resized = F.interpolate(
        work,
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    resized = resized.reshape(*leading_shape, channels, *target_size)
    if resized.dtype != original_dtype:
        resized = resized.to(original_dtype)
    return resized


def load_dataset_mix_config(path: str | Path) -> DatasetMixConfig:
    mix_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(mix_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mix config '{mix_path}' to contain a mapping.")

    raw_compatibility = payload.get("compatibility")
    if raw_compatibility is None:
        raw_compatibility = {}
    if not isinstance(raw_compatibility, dict):
        raise TypeError(
            f"Expected mix config '{mix_path}' compatibility field to be a mapping."
        )

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) == 0:
        raise ValueError(
            f"Expected mix config '{mix_path}' to contain a non-empty 'sources' list."
        )

    sources = []
    seen_names = set()
    for raw_source in raw_sources:
        if not isinstance(raw_source, dict):
            raise TypeError("Each mix source must be a mapping.")
        if (
            raw_source.get("episodes") is not None
            and raw_source.get("exclude_episodes") is not None
        ):
            raise ValueError(
                "A mix source cannot define both 'episodes' and 'exclude_episodes'."
            )

        raw_camera_keys = raw_source.get("camera_keys")
        raw_camera_map = raw_source.get("camera_map")
        if raw_camera_keys is not None and raw_camera_map is not None:
            raise ValueError(
                "A mix source cannot define both 'camera_keys' and 'camera_map'."
            )

        camera_keys: tuple[str, ...] | None = None
        camera_map: dict[str, str] | None = None

        if raw_camera_keys is not None:
            if not isinstance(raw_camera_keys, list):
                raise TypeError("Expected 'camera_keys' to be a list of camera feature keys.")
            if any(not isinstance(value, str) for value in raw_camera_keys):
                raise TypeError("Expected 'camera_keys' to contain only strings.")
            if len(raw_camera_keys) == 0:
                raise ValueError("Expected 'camera_keys' to contain at least one entry.")
            camera_keys = tuple(raw_camera_keys)

        if raw_camera_map is not None:
            if not isinstance(raw_camera_map, dict):
                raise TypeError("Expected 'camera_map' to be a mapping of role to camera key.")
            if len(raw_camera_map) == 0:
                raise ValueError("Expected 'camera_map' to contain at least one mapping.")
            parsed_map = {
                str(role): str(camera_key)
                for role, camera_key in raw_camera_map.items()
            }
            camera_map = parsed_map

        raw_filtering = raw_source.get("filtering")
        if raw_filtering is not None and not isinstance(raw_filtering, dict):
            raise TypeError("Expected 'filtering' to be a mapping when provided.")

        source = DatasetMixSourceConfig(
            name=str(raw_source["name"]),
            repo_id=str(raw_source["repo_id"]),
            root=None if raw_source.get("root") is None else str(raw_source["root"]),
            revision=(
                None
                if raw_source.get("revision") is None
                else str(raw_source["revision"])
            ),
            weight=float(raw_source["weight"]),
            episodes=_parse_episode_list(
                field_name="episodes", values=raw_source.get("episodes")
            ),
            exclude_episodes=_parse_episode_list(
                field_name="exclude_episodes", values=raw_source.get("exclude_episodes")
            ),
            supervision=str(raw_source["supervision"]),
            video_backend=(
                None
                if raw_source.get("video_backend") is None
                else str(raw_source["video_backend"])
            ),
            tolerance_s=(
                None
                if raw_source.get("tolerance_s") is None
                else float(raw_source["tolerance_s"])
            ),
            feature_key_mapping=_parse_feature_key_mapping(
                field_name="feature_key_mapping",
                value=raw_source.get("feature_key_mapping"),
            ),
            camera_keys=camera_keys,
            camera_map=camera_map,
            action_key=(
                None if raw_source.get("action_key") is None else str(raw_source["action_key"])
            ),
            filtering=(None if raw_filtering is None else deepcopy(raw_filtering)),
        )
        if source.name in seen_names:
            raise ValueError(f"Duplicate mix source name '{source.name}'.")
        if source.weight <= 0:
            raise ValueError(f"Mix source '{source.name}' must have a positive weight.")
        if source.supervision not in VALID_SUPERVISION_MODES:
            raise ValueError(
                f"Mix source '{source.name}' has invalid supervision '{source.supervision}'. "
                f"Expected one of {sorted(VALID_SUPERVISION_MODES)}."
            )

        seen_names.add(source.name)
        sources.append(source)

    return DatasetMixConfig(
        path=mix_path,
        sources=tuple(sources),
        retained_features=_parse_string_list(
            field_name="compatibility.retained_features",
            values=raw_compatibility.get("retained_features"),
        ),
        enforce_matching_fps=_parse_bool_field(
            field_name="compatibility.enforce_matching_fps",
            value=raw_compatibility.get("enforce_matching_fps"),
            default=True,
        ),
        enforce_matching_delta_timestamps=_parse_bool_field(
            field_name="compatibility.enforce_matching_delta_timestamps",
            value=raw_compatibility.get("enforce_matching_delta_timestamps"),
            default=True,
        ),
        allow_visual_shape_mismatch=_parse_bool_field(
            field_name="compatibility.allow_visual_shape_mismatch",
            value=raw_compatibility.get("allow_visual_shape_mismatch"),
            default=False,
        ),
    )


def _selected_episodes(
    source: DatasetMixSourceConfig, total_episodes: int
) -> tuple[int, ...]:
    episodes = _parse_episode_list(
        field_name=f"{source.name}.episodes",
        values=list(source.episodes) if source.episodes is not None else None,
        total_episodes=total_episodes,
    )
    exclude_episodes = _parse_episode_list(
        field_name=f"{source.name}.exclude_episodes",
        values=(
            list(source.exclude_episodes)
            if source.exclude_episodes is not None
            else None
        ),
        total_episodes=total_episodes,
    )
    if episodes is not None and exclude_episodes is not None:
        raise ValueError(
            f"Mix source '{source.name}' cannot define both 'episodes' and 'exclude_episodes'."
        )
    if episodes is not None:
        return episodes
    excluded = set(exclude_episodes or ())
    selected = tuple(
        episode for episode in range(total_episodes) if episode not in excluded
    )
    if len(selected) == 0:
        raise ValueError(f"Mix source '{source.name}' does not select any episodes.")
    return selected


def _build_source_index(
    meta: LeRobotDatasetMetadata, selected_episodes: tuple[int, ...]
) -> SourceIndex:
    starts = []
    stops = []
    for episode_index in selected_episodes:
        episode = meta.episodes[int(episode_index)]
        starts.append(int(episode["dataset_from_index"]))
        stops.append(int(episode["dataset_to_index"]))

    starts_np = np.asarray(starts, dtype=np.int64)
    stops_np = np.asarray(stops, dtype=np.int64)
    lengths = stops_np - starts_np
    if len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError("Mix sources must resolve to non-empty episodes.")

    return SourceIndex(
        episode_indices=np.asarray(selected_episodes, dtype=np.int64),
        dataset_from_index=starts_np,
        dataset_to_index=stops_np,
        lengths=lengths,
        cumulative_lengths=np.cumsum(lengths, dtype=np.int64),
    )


def _episode_stats(
    meta: LeRobotDatasetMetadata, episode_index: int
) -> dict[str, dict[str, np.ndarray]]:
    episode = meta.episodes[int(episode_index)]
    flat_stats = {}
    for key, value in episode.items():
        if key.startswith("stats/"):
            flat_stats[key[len("stats/") :]] = np.asarray(value)
    return unflatten_dict(flat_stats)


def _aggregate_selected_stats(
    meta: LeRobotDatasetMetadata, selected_episodes: tuple[int, ...]
) -> dict[str, dict[str, np.ndarray]]:
    return aggregate_stats(
        [_episode_stats(meta, episode_index) for episode_index in selected_episodes]
    )


def _stats_signature(
    stats: dict[str, dict[str, np.ndarray]],
) -> dict[str, tuple[int, ...]]:
    return {
        key: tuple(np.asarray(value).shape)
        for key, value in flatten_dict(stats).items()
    }


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError(f"Expected 1D source weights, got shape {tuple(weights.shape)}")
    if weights.size == 0:
        raise ValueError("Expected at least one source weight")
    if np.any(weights < 0):
        raise ValueError("Source weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Source weights must sum to a positive value")
    return weights / total


def _to_flat_array(value: np.ndarray | Any, *, dtype: np.dtype) -> np.ndarray:
    out = np.asarray(value, dtype=dtype)
    if out.ndim == 0:
        return out.reshape(1)
    return out.reshape(-1)


def _pad_last_dim(
    array: np.ndarray, *, target_dim: int, fill_value: float
) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim != 1:
        raise ValueError(f"Expected 1D stats array, got shape {tuple(array.shape)}")
    if int(array.shape[0]) > int(target_dim):
        raise ValueError(
            f"Stats dim {int(array.shape[0])} exceeds target_dim {int(target_dim)}"
        )
    padded = np.full((int(target_dim),), fill_value, dtype=array.dtype)
    valid = np.zeros((int(target_dim),), dtype=np.bool_)
    padded[: int(array.shape[0])] = array
    valid[: int(array.shape[0])] = True
    return padded, valid


def merge_weighted_stats(
    stats_by_source: list[dict[str, dict[str, np.ndarray]]],
    source_weights: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    if not stats_by_source:
        return {}
    if len(stats_by_source) != int(source_weights.shape[0]):
        raise ValueError("stats_by_source and source_weights length mismatch")

    weights = _normalize_weights(np.asarray(source_weights, dtype=np.float64))
    merged: dict[str, dict[str, np.ndarray]] = {}
    feature_keys = {key for stats in stats_by_source for key in stats}
    for feature_key in feature_keys:
        present = [
            (stats[feature_key], weights[idx])
            for idx, stats in enumerate(stats_by_source)
            if feature_key in stats
        ]
        if not present:
            continue

        local_weights = np.asarray([weight for _, weight in present], dtype=np.float64)
        means = [
            _to_flat_array(entry["mean"], dtype=np.float64) for entry, _ in present
        ]
        stds = [_to_flat_array(entry["std"], dtype=np.float64) for entry, _ in present]
        mins = [_to_flat_array(entry["min"], dtype=np.float64) for entry, _ in present]
        maxs = [_to_flat_array(entry["max"], dtype=np.float64) for entry, _ in present]
        counts = [
            _to_flat_array(
                entry.get("count", np.asarray([0], dtype=np.int64)), dtype=np.int64
            )
            for entry, _ in present
        ]
        feature_dim = max(int(arr.shape[0]) for arr in means)

        mean_arrays: list[np.ndarray] = []
        std_arrays: list[np.ndarray] = []
        min_arrays: list[np.ndarray] = []
        max_arrays: list[np.ndarray] = []
        valid_masks: list[np.ndarray] = []
        for mu, sigma, min_value, max_value in zip(
            means, stds, mins, maxs, strict=True
        ):
            mu_pad, valid = _pad_last_dim(mu, target_dim=feature_dim, fill_value=0.0)
            sigma_pad, _ = _pad_last_dim(sigma, target_dim=feature_dim, fill_value=0.0)
            min_pad, _ = _pad_last_dim(
                min_value, target_dim=feature_dim, fill_value=np.inf
            )
            max_pad, _ = _pad_last_dim(
                max_value, target_dim=feature_dim, fill_value=-np.inf
            )
            mean_arrays.append(mu_pad)
            std_arrays.append(sigma_pad)
            min_arrays.append(min_pad)
            max_arrays.append(max_pad)
            valid_masks.append(valid)

        valid_matrix = np.stack(valid_masks, axis=0)
        weight_matrix = valid_matrix.astype(np.float64) * local_weights[:, None]
        denom = weight_matrix.sum(axis=0)
        if np.any(denom <= 0.0):
            raise ValueError(
                f"Feature {feature_key!r} has no valid stats coverage on some dimensions"
            )
        normalized_weight_matrix = weight_matrix / denom[None, :]

        mean = np.zeros((feature_dim,), dtype=np.float64)
        for weight_row, mu in zip(normalized_weight_matrix, mean_arrays, strict=True):
            mean = mean + (weight_row * mu)

        variance = np.zeros((feature_dim,), dtype=np.float64)
        for weight_row, mu, sigma in zip(
            normalized_weight_matrix, mean_arrays, std_arrays, strict=True
        ):
            variance = variance + (weight_row * ((sigma**2) + ((mu - mean) ** 2)))

        feature_stats: dict[str, np.ndarray] = {
            "mean": mean,
            "std": np.sqrt(variance),
            "min": np.minimum.reduce(np.stack(min_arrays, axis=0)),
            "max": np.maximum.reduce(np.stack(max_arrays, axis=0)),
        }
        if all(int(count.shape[0]) == 1 for count in counts):
            feature_stats["count"] = np.asarray(
                [sum(int(count[0]) for count in counts)],
                dtype=np.int64,
            )
        else:
            count_arrays = [
                _pad_last_dim(
                    count.astype(np.int64), target_dim=feature_dim, fill_value=0
                )[0]
                for count in counts
            ]
            feature_stats["count"] = np.sum(np.stack(count_arrays, axis=0), axis=0)

        quantile_keys = {
            stat_key
            for entry, _ in present
            for stat_key in entry
            if stat_key.startswith("q") and stat_key[1:].isdigit()
        }
        for quantile_key in quantile_keys:
            if all(quantile_key in entry for entry, _ in present):
                values = [
                    _pad_last_dim(
                        _to_flat_array(entry[quantile_key], dtype=np.float64),
                        target_dim=feature_dim,
                        fill_value=0.0,
                    )[0]
                    for entry, _ in present
                ]
                q_value = np.zeros((feature_dim,), dtype=np.float64)
                for weight_row, value in zip(
                    normalized_weight_matrix, values, strict=True
                ):
                    q_value = q_value + (weight_row * value)
                feature_stats[quantile_key] = q_value

        merged[feature_key] = feature_stats
    return merged


def build_explicit_mixed_stats(sources: list[Any]) -> dict[str, dict[str, np.ndarray]]:
    stats_by_source = []
    for source in sources:
        stats = getattr(source, "stats", None)
        if stats is None:
            stats = getattr(getattr(source, "meta", None), "stats", None)
        if stats is None:
            raise ValueError("Each mixed source must expose stats directly or via source.meta.stats")
        stats_by_source.append(deepcopy(stats))
    source_weights = np.asarray(
        [float(source.weight) for source in sources], dtype=np.float64
    )
    return merge_weighted_stats(stats_by_source, source_weights)


class LogicalSource:
    def __init__(
        self,
        *,
        source_index: int,
        config: DatasetMixSourceConfig,
        meta: LeRobotDatasetMetadata,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Any = None,
        default_tolerance_s: float = 1e-4,
        shared_dataset_cache: dict[tuple[Any, ...], LeRobotDataset] | None = None,
        request_image_deltas: tuple[int, ...] | None = None,
        global_filtering_cfg: dict[str, Any] | None = None,
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
                f"Feature remapping for source '{config.name}' references unknown "
                f"feature keys: {unknown_feature_keys}"
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
        self.tolerance_s = float(
            default_tolerance_s if config.tolerance_s is None else config.tolerance_s
        )
        self._shared_dataset_cache = (
            {} if shared_dataset_cache is None else shared_dataset_cache
        )


        self._request_image_deltas = (
            None
            if request_image_deltas is None
            else tuple(int(delta) for delta in request_image_deltas)
        )
        self._global_filtering_cfg = (
            None if global_filtering_cfg is None else deepcopy(global_filtering_cfg)
        )
        self._filter_cache_path: str | None = None
        self._filter_summary: dict[str, Any] | None = None
        self._kept_anchor_values: np.ndarray | None = None
        self._kept_offsets_start: np.ndarray | None = None
        self._kept_counts: np.ndarray | None = None
        self._maybe_apply_action_frame_filtering()

    def _selected_filter_camera_keys(self) -> tuple[str, ...]:
        if self.config.camera_keys is not None:
            return tuple(str(key) for key in self.config.camera_keys)
        if self.config.camera_map is not None:
            return tuple(
                dict.fromkeys(
                    str(camera_key) for camera_key in self.config.camera_map.values()
                )
            )
        return tuple(sorted(self.camera_keys))

    def _infer_request_image_deltas(self, camera_key: str) -> tuple[int, ...]:
        if self._request_image_deltas is not None:
            return self._request_image_deltas
        if self.delta_timestamps is None or camera_key not in self.delta_timestamps:
            raise ValueError(
                f"Cannot infer request_image_deltas for source={self.name!r}: missing camera {camera_key!r}."
            )
        fps = float(self.meta.fps)
        deltas_seconds = self.delta_timestamps[camera_key]
        return tuple(int(round(float(delta_s) * fps)) for delta_s in deltas_seconds)

    def _maybe_apply_action_frame_filtering(self) -> None:
        source_filtering = self.config.filtering
        global_filtering = self._global_filtering_cfg
        if source_filtering is None and global_filtering is None:
            return

        from common.action_frame_filtering import build_action_frame_filter
        from common.action_frame_filtering import normalize_filtering_config

        filtering_cfg = normalize_filtering_config(
            global_filtering=global_filtering,
            source_filtering=source_filtering,
        )
        if filtering_cfg is None or not bool(filtering_cfg.get("apply_at_sampling", True)):
            return

        camera_dataset_keys = self._selected_filter_camera_keys()
        if len(camera_dataset_keys) == 0:
            raise ValueError(
                f"Source {self.name!r} resolved zero camera keys for filtering"
            )
        request_image_deltas = self._infer_request_image_deltas(camera_dataset_keys[0])
        action_key = self.config.action_key
        if action_key is None and ACTION in self.features:
            action_key = ACTION
        motion_cfg = dict(filtering_cfg.get("motion", {}))
        camera_aggregate_reduce = str(motion_cfg.get("aggregate_reduce", "mean"))

        result = build_action_frame_filter(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            video_backend=self.config.video_backend,
            tolerance_s=self.tolerance_s,
            request_image_deltas=request_image_deltas,
            camera_dataset_keys=camera_dataset_keys,
            camera_aggregate_reduce=camera_aggregate_reduce,
            action_key=action_key,
            episode_ids=self.index.episode_indices.astype(np.int32),
            candidate_start=self.index.dataset_from_index.astype(np.int64),
            candidate_end=self.index.dataset_to_index.astype(np.int64),
            filtering_cfg=filtering_cfg,
            split="train",
        )

        self._filter_cache_path = result.cache_path
        self._filter_summary = dict(result.summary)
        self._kept_anchor_values = result.kept_anchor_values.astype(np.int64)
        self._kept_offsets_start = result.kept_offsets_start.astype(np.int64)
        self._kept_counts = result.kept_counts.astype(np.int64)

        cache_status = str(self._filter_summary.get("cache", "unknown"))
        before = int(self._filter_summary.get("anchors_before", 0))
        after = int(self._filter_summary.get("anchors_after", 0))
        if cache_status == "miss":
            logger.warning(
                "[mixed-filter] regenerated cache source=%s repo=%s cache=%s before=%d after=%d",
                self.name,
                self.repo_id,
                self._filter_cache_path,
                before,
                after,
            )
        else:
            logger.info(
                "[mixed-filter] cache_hit source=%s repo=%s cache=%s before=%d after=%d",
                self.name,
                self.repo_id,
                self._filter_cache_path,
                before,
                after,
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
    def supervision(self) -> str:
        return self.config.supervision

    @property
    def num_frames(self) -> int:
        if self._kept_counts is not None:
            return int(self._kept_counts.sum())
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
    def action_supervised(self) -> bool:
        return self.supervision == "multitask"

    @property
    def latent_supervised(self) -> bool:
        return True

    @property
    def dataset_identity(self) -> tuple[str, str | None, str | None]:
        return (self.repo_id, str(Path(self.meta.root).resolve()), self.revision)

    def get_effective_lengths(self, drop_n_last_frames: int = 0) -> np.ndarray:
        base_lengths = (
            self._kept_counts if self._kept_counts is not None else self.index.lengths
        )
        return np.maximum(base_lengths - int(drop_n_last_frames), 0)

    def flat_index_to_anchor(self, index: int, *, drop_n_last_frames: int = 0) -> int:
        effective_lengths = self.get_effective_lengths(drop_n_last_frames)
        total_effective = int(effective_lengths.sum())
        if index < 0 or index >= total_effective:
            raise IndexError(f"Index {index} out of bounds for source '{self.name}'.")

        cumulative_lengths = np.cumsum(effective_lengths, dtype=np.int64)
        episode_pos = int(np.searchsorted(cumulative_lengths, index, side="right"))
        prev_total = 0 if episode_pos == 0 else int(cumulative_lengths[episode_pos - 1])
        local_offset = int(index - prev_total)
        if self._kept_anchor_values is None or self._kept_offsets_start is None:
            return int(self.index.dataset_from_index[episode_pos] + local_offset)
        global_offset = int(self._kept_offsets_start[episode_pos]) + local_offset
        return int(self._kept_anchor_values[global_offset])

    def metadata(self) -> MixedSourceMetadata:
        return MixedSourceMetadata(
            name=self.name,
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            weight=self.weight,
            supervision=self.supervision,
            episodes=self.selected_episodes,
            video_backend=self.config.video_backend,
            tolerance_s=self.tolerance_s,
            num_frames=self.num_frames,
            filter_cache_path=self._filter_cache_path,
            filter_summary=(
                None if self._filter_summary is None else deepcopy(self._filter_summary)
            ),
            feature_key_mapping=deepcopy(self.feature_key_mapping),
            retained_features=self.retained_features,
        )

    def _get_dataset(self) -> LeRobotDataset:
        cache_key = (
            self.repo_id,
            str(Path(self.meta.root).resolve()),
            self.revision,
            self.config.video_backend,
            float(self.tolerance_s),
            repr(self.raw_delta_timestamps),
            id(self.image_transforms),
        )
        dataset = self._shared_dataset_cache.get(cache_key)
        if dataset is None:
            dataset = LeRobotDataset(
                self.repo_id,
                root=self.meta.root,
                episodes=None,
                image_transforms=self.image_transforms,
                delta_timestamps=self.raw_delta_timestamps,
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
            relative_index = int(
                dataset._absolute_to_relative_idx[int(anchor_abs_index)]
            )
        if _debug_mixed_dataset():
            logger.info(
                "[mixed] source=%s supervision=%s anchor_abs=%s relative=%s",
                self.name,
                self.supervision,
                int(anchor_abs_index),
                int(relative_index),
            )

        item = dataset[relative_index]
        if self.feature_key_mapping:
            remapped_item: dict[str, Any] = {}
            for key, value in item.items():
                target_key = _remap_output_key(key, self.feature_key_mapping)
                if target_key in remapped_item:
                    raise ValueError(
                        f"Feature remapping for source '{self.name}' causes duplicate "
                        f"item key {target_key!r}."
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
        item["hlrp_action_supervised"] = torch.tensor(
            self.action_supervised, dtype=torch.bool
        )
        item["hlrp_latent_supervised"] = torch.tensor(
            self.latent_supervised, dtype=torch.bool
        )
        item["hlrp_source_name"] = self.name
        item["dataset_source_index"] = torch.tensor(
            self.source_index, dtype=torch.int64
        )
        item["dataset_source_name"] = self.name
        item["dataset_source_repo_id"] = self.repo_id
        item["dataset_source_root"] = "" if self.root is None else self.root
        item["dataset_source_revision"] = "" if self.revision is None else self.revision
        return item


def _build_virtual_episodes(sources: list[LogicalSource]) -> pd.DataFrame:
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
    sources: list[LogicalSource],
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
            "supervision": metadata.supervision,
            "episodes": list(metadata.episodes),
            "video_backend": metadata.video_backend,
            "tolerance_s": metadata.tolerance_s,
            "num_frames": metadata.num_frames,
            "filter_cache_path": metadata.filter_cache_path,
            "filter_summary": metadata.filter_summary,
            "feature_key_mapping": deepcopy(metadata.feature_key_mapping),
            "retained_features": (
                None
                if metadata.retained_features is None
                else list(metadata.retained_features)
            ),
        }
        for metadata in (source.metadata() for source in sources)
    ]
    info["mix_path"] = str(mix_path)
    info["logical_repo_id"] = logical_repo_id
    return info


def validate_mixed_sources(
    sources: list[LogicalSource],
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
    base_stats_signature = _stats_signature(base_source.stats)

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
        if _stats_signature(source.stats) != base_stats_signature:
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


class MixedLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        logical_repo_id: str,
        mix_path: str | Path,
        sources: list[LogicalSource],
        enforce_matching_fps: bool = True,
        enforce_matching_delta_timestamps: bool = True,
        allow_visual_shape_mismatch: bool = False,
    ) -> None:
        super().__init__()
        validate_mixed_sources(
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
        self._cumulative_frames = np.cumsum(
            [source.num_frames for source in self.sources], dtype=np.int64
        )

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
        return (
            0 if len(self._cumulative_frames) == 0 else int(self._cumulative_frames[-1])
        )

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

        source_index = int(
            np.searchsorted(self._cumulative_frames, int(index), side="right")
        )
        previous_total = (
            0 if source_index == 0 else int(self._cumulative_frames[source_index - 1])
        )
        logical_index = int(index) - previous_total
        anchor_abs_index = self.sources[source_index].flat_index_to_anchor(
            logical_index
        )
        return self.sources[source_index].get_item(anchor_abs_index)

    def build_sampler(
        self, *, seed: int | None = None, drop_n_last_frames: int = 0
    ) -> WeightedSourceSampler:
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
        return WeightedSourceSampler(
            sources=self.sources,
            source_weights=self.source_weights,
            num_samples=num_samples,
            seed=0 if seed is None else int(seed),
            drop_n_last_frames=drop_n_last_frames,
        )
