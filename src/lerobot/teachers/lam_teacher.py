#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""LAM teacher adapter for LeRobot training.

This module provides an adapter for using HLRP LAM (Latent Action Quantization)
models as teachers for online label generation during training.

The LAM teacher generates discrete latent codes from frame pairs (frame_t, frame_{t+delta}),
which can be used as targets for pretraining vision-language models.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LAMTeacherConfig:
    """Configuration for LAM teacher.

    Args:
        checkpoint_path: Path to the LAM model checkpoint.
        device: Device to load the model on (e.g., "cuda", "cpu").
    """

    checkpoint_path: str
    device: str = "cuda"


class LAMTeacher(nn.Module):
    """Frozen LAM teacher for online label generation.

    This class wraps an HLRP LAM model to generate latent codes from frame pairs.
    The teacher is kept frozen during training and used only for generating targets.

    Args:
        config: LAMTeacherConfig with checkpoint path and device.
    """

    def __init__(self, config: LAMTeacherConfig):
        super().__init__()

        # Import from HLRP
        from stage2.online_lam import LAMTaskCodeProvider
        from lam.task import LAMTask

        # Use weights_only=False for LAM checkpoints (trusted source)
        # Required for PyTorch 2.6+ which changed the default
        lam_task = LAMTask.load_from_checkpoint(
            config.checkpoint_path,
            map_location=config.device,
            weights_only=False,
        )
        self.provider = LAMTaskCodeProvider(lam_task)
        self.provider.to(config.device)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @property
    def codebook_size(self) -> int:
        """Size of the LAM codebook (K)."""
        return self.provider.codebook_size

    @property
    def code_seq_len(self) -> int:
        """Length of the code sequence (S)."""
        return self.provider.code_seq_len

    @torch.no_grad()
    def codes_from_pair(self, frames: torch.Tensor) -> torch.Tensor:
        """Generate LAM codes from a frame pair.

        Args:
            frames: Frame pair tensor of shape [B, 2, 3, H, W] with float values in [0, 1].
                    frames[:, 0] is frame_t, frames[:, 1] is frame_{t+delta}.

        Returns:
            codes: Long tensor of shape [B, S] with values in {0, ..., K-1},
                   where S is the code sequence length and K is the codebook size.
        """
        # Permute [B, 2, 3, H, W] -> [B, 3, 2, H, W] for video format
        video = frames.permute(0, 2, 1, 3, 4)
        return self.provider.codes_from_video(video)


def valid_pair_from_is_pad(is_pad: torch.Tensor) -> torch.Tensor:
    """Compute valid pair mask from per-frame padding indicators.

    A frame pair is valid only if BOTH frames are valid (not padded).

    Args:
        is_pad: Boolean tensor of shape [B, 2] where True indicates a padded frame.
                is_pad[:, 0] is for frame_t, is_pad[:, 1] is for frame_{t+delta}.

    Returns:
        valid_pair: Boolean tensor of shape [B] where True indicates a valid pair
                    (both frames are not padded).
    """
    return ~is_pad[:, 0] & ~is_pad[:, 1]
