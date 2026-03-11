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
from collections.abc import Iterator
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _debug_sampler() -> bool:
    value = os.environ.get("HLRP_STAGE3_DEBUG_DATASET", "")
    return value.lower() in {"1", "true", "yes", "on"}


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(
                        start_index + drop_n_first_frames,
                        end_index - drop_n_last_frames,
                    )
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")

    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Weights must sum to a positive value.")

    return weights / total


class WeightedSourceSampler(torch.utils.data.Sampler[tuple[int, int]]):
    def __init__(
        self,
        *,
        sources: list,
        source_weights: np.ndarray,
        num_samples: int,
        seed: int,
        drop_n_last_frames: int = 0,
    ) -> None:
        self.sources = list(sources)
        self.source_weights = _normalize_weights(source_weights)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.drop_n_last_frames = int(drop_n_last_frames)
        self._epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self.num_samples <= 0:
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        effective_lengths = [
            source.get_effective_lengths(self.drop_n_last_frames)
            for source in self.sources
        ]
        source_weights = self.source_weights.copy()
        for source_index, lengths in enumerate(effective_lengths):
            if int(lengths.sum()) == 0:
                source_weights[source_index] = 0.0
        source_weights = _normalize_weights(source_weights)

        source_ids = torch.multinomial(
            torch.as_tensor(source_weights, dtype=torch.float64),
            num_samples=self.num_samples,
            replacement=True,
            generator=generator,
        ).tolist()

        for source_id in source_ids:
            lengths = effective_lengths[source_id]
            episode_weights = _normalize_weights(lengths.astype(np.float64))
            episode_pos = int(
                torch.multinomial(
                    torch.as_tensor(episode_weights, dtype=torch.float64),
                    num_samples=1,
                    replacement=True,
                    generator=generator,
                ).item()
            )
            offset = int(
                torch.randint(
                    int(lengths[episode_pos]), size=(1,), generator=generator
                ).item()
            )
            anchor = int(
                self.sources[source_id].index.dataset_from_index[episode_pos] + offset
            )
            if _debug_sampler():
                logger.info(
                    "[sampler] source=%s episode=%s offset=%s anchor=%s",
                    getattr(self.sources[source_id], "name", source_id),
                    int(self.sources[source_id].index.episode_indices[episode_pos]),
                    offset,
                    anchor,
                )
            yield int(source_id), anchor
