#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pprint import pformat

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.mixed_dataset import (
    LogicalSource,
    MixedLeRobotDataset,
    load_dataset_mix_config,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata,
    *,
    source_camera_keys: set[str] | None = None,
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the resulting dict is empty.
    """
    observation_delta_indices = cfg.observation_delta_indices
    if hasattr(cfg, "get_observation_delta_indices_for_fps"):
        observation_delta_indices = cfg.get_observation_delta_indices_for_fps(
            ds_meta.fps
        )

    delta_timestamps = {}
    configured_camera_keys = getattr(cfg, "camera_keys", None)
    allowed_camera_keys = (
        None
        if configured_camera_keys is None
        else {str(k) for k in configured_camera_keys}
    )
    if source_camera_keys is not None:
        if allowed_camera_keys is None:
            allowed_camera_keys = set(source_camera_keys)
        else:
            allowed_camera_keys = allowed_camera_keys.intersection(source_camera_keys)
    configured_input_features = getattr(cfg, "input_features", None)
    visual_input_keys = None
    if isinstance(configured_input_features, dict) and len(configured_input_features) > 0:
        visual_input_keys = set()
        for key, feature in configured_input_features.items():
            feature_type = str(getattr(feature, "type", "")).upper()
            if feature_type == "VISUAL":
                visual_input_keys.add(str(key))
    selected_observation_image_keys = 0
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            if (
                allowed_camera_keys is not None
                and key.startswith(f"{OBS_PREFIX}images.")
                and key not in allowed_camera_keys
            ):
                continue
            if (
                visual_input_keys is not None
                and key.startswith(f"{OBS_PREFIX}images.")
                and key not in visual_input_keys
            ):
                continue
            delta_timestamps[key] = [
                i / ds_meta.fps for i in cfg.observation_delta_indices
            ]
            if key.startswith(f"{OBS_PREFIX}images."):
                selected_observation_image_keys += 1

    if (
        source_camera_keys is not None
        and cfg.observation_delta_indices is not None
        and selected_observation_image_keys == 0
    ):
        raise ValueError(
            "Source camera selection resolved to zero observation image keys after policy filtering. "
            f"Requested source cameras={sorted(source_camera_keys)} "
            f"policy.camera_keys={getattr(cfg, 'camera_keys', None)}"
        )

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def _resolve_source_camera_keys(
    source_cfg,
    source_meta: LeRobotDatasetMetadata,
) -> set[str] | None:
    source_camera_keys = source_cfg.camera_keys
    source_camera_map = source_cfg.camera_map
    if source_camera_keys is None and source_camera_map is None:
        return None

    if source_camera_keys is not None:
        selected = {str(key) for key in source_camera_keys}
    else:
        selected = {str(camera_key) for camera_key in source_camera_map.values()}

    available_cameras = {
        key
        for key, feature in source_meta.features.items()
        if feature.get("dtype") in {"image", "video"}
    }
    missing = sorted(key for key in selected if key not in available_cameras)
    if missing:
        raise ValueError(
            f"Mix source {source_cfg.name!r} requested unknown camera keys: {missing}"
        )
    return selected


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | MixedLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MixedLeRobotDataset
    """
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
        if cfg.dataset.image_transforms.enable
        else None
    )

    if cfg.dataset.mix_path is not None:
        if cfg.dataset.streaming:
            raise ValueError("Mixed datasets do not support streaming mode.")

        mix_cfg = load_dataset_mix_config(cfg.dataset.mix_path)
        visual_target_size = getattr(cfg.policy, "image_size", None)
        sources = []
        shared_dataset_cache = {}
        request_image_deltas = None
        if cfg.policy.observation_delta_indices is not None:
            request_image_deltas = tuple(int(x) for x in cfg.policy.observation_delta_indices)
        global_filtering_cfg = getattr(cfg.dataset, "filtering", None)
        for source_index, source_cfg in enumerate(mix_cfg.sources):
            source_meta = LeRobotDatasetMetadata(
                source_cfg.repo_id,
                root=source_cfg.root,
                revision=source_cfg.revision,
            )

            if (
                getattr(cfg.policy, "type", None) == "latent_smol"
                and getattr(cfg.policy, "head_mode", None) == "latent"
                and getattr(cfg.policy, "lam_future_frames", 0) <= 0
            ):
                cfg.policy.lam_future_frames = max(
                    1, round(source_meta.fps * cfg.policy.lam_future_seconds)
                )
                logging.info(
                    f"Computed lam_future_frames={cfg.policy.lam_future_frames} (fps={source_meta.fps})"
                )

            source_selected_camera_keys = _resolve_source_camera_keys(source_cfg, source_meta)
            source_delta_timestamps = resolve_delta_timestamps(
                cfg.policy,
                source_meta,
                source_camera_keys=source_selected_camera_keys,
            )
            sources.append(
                LogicalSource(
                    source_index=source_index,
                    config=source_cfg,
                    meta=source_meta,
                    delta_timestamps=source_delta_timestamps,
                    image_transforms=image_transforms,
                    default_tolerance_s=cfg.tolerance_s,
                    shared_dataset_cache=shared_dataset_cache,
                    retained_features=mix_cfg.retained_features,
                    visual_target_size=visual_target_size,
                    request_image_deltas=request_image_deltas,
                    global_filtering_cfg=global_filtering_cfg,
                )
            )

        dataset = MixedLeRobotDataset(
            logical_repo_id=cfg.dataset.repo_id,
            mix_path=mix_cfg.path,
            sources=sources,
            enforce_matching_fps=mix_cfg.enforce_matching_fps,
            enforce_matching_delta_timestamps=mix_cfg.enforce_matching_delta_timestamps,
            allow_visual_shape_mismatch=mix_cfg.allow_visual_shape_mismatch,
        )
    elif isinstance(cfg.dataset.repo_id, str):
        ds_meta = LeRobotDatasetMetadata(
            cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
        )

        # Compute lam_future_frames for latent_smol in latent mode before resolving delta timestamps
        if (
            getattr(cfg.policy, "type", None) == "latent_smol"
            and getattr(cfg.policy, "head_mode", None) == "latent"
            and getattr(cfg.policy, "lam_future_frames", 0) <= 0
        ):
            cfg.policy.lam_future_frames = max(
                1, round(ds_meta.fps * cfg.policy.lam_future_seconds)
            )
            logging.info(
                f"Computed lam_future_frames={cfg.policy.lam_future_frames} (fps={ds_meta.fps})"
            )

        delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
        if not cfg.dataset.streaming:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                video_backend=cfg.dataset.video_backend,
                tolerance_s=cfg.tolerance_s,
            )
        else:
            dataset = StreamingLeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                episodes=cfg.dataset.episodes,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                revision=cfg.dataset.revision,
                max_num_shards=cfg.num_workers,
                tolerance_s=cfg.tolerance_s,
            )
    else:
        raise NotImplementedError("The MultiLeRobotDataset isn't supported for now.")

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    return dataset
