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

"""Configuration for LatentSmol policy - SmolVLA with Mode A (latent pretraining) support."""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("latent_smol")
@dataclass
class LatentSmolConfig(SmolVLAConfig):
    """LatentSmol config - inherits all SmolVLA fields, adds Mode A support.

    This config supports two training modes:
    - head_mode="latent" (Mode A): Train VLM backbone on LAQ codes from (frame_t, frame_{t+delta})
    - head_mode="action" (Mode B): Standard action head training with flow-matching (same as SmolVLA)

    Workflow:
    1. Stage 1 - Pretrain: Train with head_mode="latent" on video data
    2. Stage 2 - Finetune: Load pretrained checkpoint, train with head_mode="action" on robot data
    """

    # Override SmolVLA defaults for Stage 1 (train full backbone)
    # SmolVLA defaults: freeze_vision_encoder=True, train_expert_only=True
    # For latent pretraining, we want to train the full VLM
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Mode selection
    head_mode: str = "latent"  # "latent" (Mode A) or "action" (Mode B)

    # LAQ-specific (only used when head_mode="latent")
    laq_checkpoint_path: str | None = None
    laq_loss_weight: float = 1.0
    laq_codebook_size: int = 8
    laq_code_seq_len: int = 4
    laq_future_seconds: float = 1.0
    laq_future_frames: int = -1  # Computed from dataset fps at runtime
    laq_camera_key: str = "observation.images.proprio"
    laq_resize_hw: tuple[int, int] = (256, 256)

    @property
    def observation_delta_indices(self) -> list[int]:
        """Return observation delta indices based on head_mode.

        For latent mode: returns [0, delta_frames] to get frame pair (t=0, t=delta)
        For action mode: returns [0] (single frame, same as SmolVLA)
        """
        if self.head_mode == "latent":
            if self.laq_future_frames <= 0:
                raise ValueError(
                    "laq_future_frames must be >0 for head_mode='latent'. "
                    "This is computed automatically from dataset fps in make_dataset()."
                )
            return [0, self.laq_future_frames]
        return [0]  # Mode B: single frame (inherit SmolVLA behavior)
