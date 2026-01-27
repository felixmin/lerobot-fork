#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
LatentSmol Policy:

A variant of SmolVLA that supports two training modes:
1. Mode A (head_mode="latent"): Latent pretraining with LAQ codes
2. Mode B (head_mode="action"): Standard action head training (same as SmolVLA)

This enables a two-stage training workflow:
- Stage 1: Pretrain backbone on video data using LAQ codes as targets
- Stage 2: Finetune on robot data with action head

Usage:
```python
# Stage 1: Latent pretraining
policy = LatentSmolPolicy.from_pretrained(
    "lerobot/smolvla_base",
    config=LatentSmolConfig(head_mode="latent", laq_checkpoint_path="...")
)

# Stage 2: Action finetuning
policy = LatentSmolPolicy.from_pretrained(
    "outputs/latent_pretrain/checkpoint_XXXXX",
    config=LatentSmolConfig(head_mode="action")
)
```
"""

from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.latent_smol.configuration_latent_smol import LatentSmolConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad,
)
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def _pool_hidden(
    hidden: torch.Tensor,  # [B, L, D]
    mask: torch.Tensor,  # [B, L] bool (True = valid token)
) -> torch.Tensor:
    """Masked mean pooling over sequence dimension."""
    mask_float = mask.float().unsqueeze(-1)  # [B, L, 1]
    pooled = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)
    return pooled  # [B, D]


class LatentSmolFlowMatching(VLAFlowMatching):
    """Extended VLAFlowMatching with latent head for Mode A training."""

    def __init__(self, config: LatentSmolConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__(config, rtc_processor)

        # Latent action head (only used when head_mode=latent)
        if config.head_mode == "latent":
            # Use VLM hidden size (not expert size) since prefix lives in VLM space
            hidden_dim = self.vlm_with_expert.config.text_config.hidden_size  # 576 for SmolVLM2-500M
            self.laq_head = nn.Linear(
                hidden_dim,
                config.laq_code_seq_len * config.laq_codebook_size  # 4 * 8 = 32
            )

    def forward_latent(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        laq_codes: torch.Tensor,  # [B, S] target codes
        laq_valid_pair: torch.Tensor,  # [B] bool
    ) -> tuple[torch.Tensor, dict]:
        """
        Mode A forward: predict LAQ codes from prefix only.
        No action suffix / flow-matching involved.
        """
        # Get prefix embeddings (image + language + state)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Forward through VLM backbone (prefix only, no suffix)
        att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1

        # Use fill_kv_cache=True to force self-attention (not cross-attention)
        # since we only have prefix embeddings in latent mode (no action suffix)
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],  # No suffix
            use_cache=False,
            fill_kv_cache=True,  # Forces self-attention path
        )
        prefix_out = outputs_embeds[0]  # First element is prefix output

        # Pool using prefix mask
        pooled = _pool_hidden(prefix_out, prefix_pad_masks)  # [B, D]

        # Project to logits
        logits = self.laq_head(pooled)  # [B, S*K]
        logits = logits.view(-1, self.config.laq_code_seq_len, self.config.laq_codebook_size)  # [B, S, K]

        # CE loss (only on valid pairs)
        valid_logits = logits[laq_valid_pair]  # [V, S, K]
        valid_codes = laq_codes[laq_valid_pair]  # [V, S]

        if valid_logits.numel() > 0:
            loss = F.cross_entropy(
                valid_logits.reshape(-1, self.config.laq_codebook_size),
                valid_codes.reshape(-1)
            )
            preds = valid_logits.argmax(dim=-1)
            accuracy = (preds == valid_codes).float().mean()

            # Per-position accuracy
            per_pos_acc = (preds == valid_codes).float().mean(dim=0)  # [S]

            # Confidence (softmax probability of predicted class)
            probs = F.softmax(valid_logits, dim=-1)  # [V, S, K]
            pred_probs = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1)  # [V, S]
            mean_confidence = pred_probs.mean()
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            accuracy = torch.tensor(0.0, device=logits.device)
            per_pos_acc = torch.zeros(self.config.laq_code_seq_len, device=logits.device)
            mean_confidence = torch.tensor(0.0, device=logits.device)
            preds = None

        loss_dict = {
            "laq_loss": loss.item(),
            "laq_accuracy": accuracy.item(),
            "laq_valid_pairs": laq_valid_pair.sum().item(),
            "laq_confidence": mean_confidence.item(),
        }

        # Add per-position accuracy
        for i, acc in enumerate(per_pos_acc.tolist()):
            loss_dict[f"laq_acc_pos{i}"] = acc

        # Store tensors for visualization (detached)
        loss_dict["_logits"] = logits.detach()  # [B, S, K]
        loss_dict["_codes_gt"] = laq_codes.detach()  # [B, S]
        loss_dict["_codes_pred"] = preds.detach() if preds is not None else None  # [V, S]
        loss_dict["_valid_mask"] = laq_valid_pair.detach()  # [B]

        return loss * self.config.laq_loss_weight, loss_dict


class LatentSmolPolicy(PreTrainedPolicy):
    """Wrapper class around LatentSmolFlowMatching to train and run inference within LeRobot.

    Supports two training modes:
    - head_mode="latent": Train backbone on LAQ codes (Mode A)
    - head_mode="action": Train action head with flow-matching (Mode B, same as SmolVLA)
    """

    config_class = LatentSmolConfig
    name = "latent_smol"

    def __init__(
        self,
        config: LatentSmolConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = LatentSmolFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Dispatch to latent or action forward based on head_mode."""
        if self.config.head_mode == "latent":
            return self.forward_latent(batch)
        else:
            # Mode B: existing SmolVLA action forward (unchanged)
            return self.forward_action(batch, noise, time, reduction)

    def forward_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Mode A: latent code prediction.

        CRITICAL: Must use frame_t (index 0), NOT frame_{t+delta} (index -1).
        SmolVLA's prepare_images/prepare_state use [:, -1, ...] which would
        accidentally condition on the future frame. Use latent-specific versions.
        """
        # Use frame at t=0 (NOT t=-1 which is the future frame)
        images, img_masks = self.prepare_images_latent(batch)  # Custom: uses [:, 0]
        state = self.prepare_state_latent(batch)  # Custom: uses [:, 0]
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        return self.model.forward_latent(
            images, img_masks, lang_tokens, lang_masks, state,
            laq_codes=batch["laq_codes"],
            laq_valid_pair=batch["laq_valid_pair"],
        )

    def prepare_images_latent(self, batch: dict[str, Tensor]):
        """Prepare images for latent mode - SAME as SmolVLA but index [:, 0] not [:, -1].

        CRITICAL: Must apply identical preprocessing as SmolVLA (resize_with_pad, *2-1 scaling).
        Only difference: index t=0 instead of t=-1, and use {key}_is_pad mask.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]
            if img.ndim == 5:  # [B, T, C, H, W]
                img = img[:, 0]  # Take t=0, not t=-1 <- ONLY CHANGE from SmolVLA

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device

            # Mask from _is_pad (per-time offset), NOT _padding_mask
            is_pad_key = f"{key}_is_pad"
            if is_pad_key in batch:
                is_pad = batch[is_pad_key]  # [B, 2]
                mask = ~is_pad[:, 0]  # Valid if NOT padded at t=0
            elif f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)

            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state_latent(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepare state for latent mode - SAME as SmolVLA but index [:, 0, :] not [:, -1, :].

        Uses pad_vector utility (same as SmolVLA) to pad to max_state_dim.
        """
        state = batch.get(OBS_STATE)
        if state is None:
            return None
        if state.ndim == 3:  # [B, T, state_dim]
            state = state[:, 0, :]  # Take t=0, not t=-1 <- ONLY CHANGE from SmolVLA

        # Use pad_vector utility (same as SmolVLA)
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def forward_action(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Mode B: existing SmolVLA action loss (flow-matching).

        This is identical to SmolVLAPolicy.forward().
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_id_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to images (used for action mode).

        Identical to SmolVLAPolicy.prepare_images().
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state (used for action mode). Identical to SmolVLAPolicy.prepare_state()."""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action. Identical to SmolVLAPolicy.prepare_action()."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    # ===== Aloha-specific transforms (copied from SmolVLA) =====

    def _pi_aloha_decode_state(self, state):
        from lerobot.policies.smolvla.modeling_smolvla import aloha_gripper_to_angular

        # Flip the joints
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        from lerobot.policies.smolvla.modeling_smolvla import aloha_gripper_from_angular

        # Flip the joints
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        from lerobot.policies.smolvla.modeling_smolvla import aloha_gripper_from_angular_inv

        # Flip the joints
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    # ===== Inference methods =====

    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations."""
        if self.config.head_mode == "latent":
            raise NotImplementedError("head_mode='latent' is for pretraining only, not inference")

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled
