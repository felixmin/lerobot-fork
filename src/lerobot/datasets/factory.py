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
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.mixed_dataset import (
    LogicalSource,
    MixedLeRobotDataset,
    load_dataset_mix_config,
)
from lerobot.datasets.multi_dataset import MultiLeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.transforms import ImageTransforms
from lerobot.utils.constants import ACTION, OBS_PREFIX, REWARD

IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
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
    for key in ds_meta.features:
        if key == REWARD and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and observation_delta_indices is not None:
            delta_timestamps[key] = [
                i / ds_meta.fps for i in observation_delta_indices
            ]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps


def make_dataset(
    cfg: TrainPipelineConfig,
) -> LeRobotDataset | MixedLeRobotDataset | MultiLeRobotDataset:
    """Handles the logic of setting up delta timestamps and image transforms before creating a dataset.

    Args:
        cfg (TrainPipelineConfig): A TrainPipelineConfig config which contains a DatasetConfig and a PreTrainedConfig.

    Raises:
        NotImplementedError: The MultiLeRobotDataset is currently deactivated.

    Returns:
        LeRobotDataset | MultiLeRobotDataset
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
        for source_index, source_cfg in enumerate(mix_cfg.sources):
            source_meta = LeRobotDatasetMetadata(
                source_cfg.repo_id,
                root=source_cfg.root,
                revision=source_cfg.revision,
            )

            source_delta_timestamps = resolve_delta_timestamps(cfg.policy, source_meta)
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
        dataset = MultiLeRobotDataset(
            cfg.dataset.repo_id,
            # TODO(aliberts): add proper support for multi dataset
            # delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            video_backend=cfg.dataset.video_backend,
        )
        logging.info(
            "Multiple datasets were provided. Applied the following index mapping to the provided datasets: "
            f"{pformat(dataset.repo_id_to_index, indent=2)}"
        )

    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(
                    stats, dtype=torch.float32
                )

    return dataset
