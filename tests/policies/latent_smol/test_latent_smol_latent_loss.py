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

"""Unit tests for LatentSmol policy latent loss computation."""

import pytest
import torch
import torch.nn.functional as F


class TestLatentSmolCELoss:
    """Tests for cross-entropy loss with masking in latent mode."""

    def test_ce_loss_shape(self):
        """Test that CE loss handles correct shapes [B, S, K]."""
        batch_size = 4
        code_seq_len = 4  # S
        codebook_size = 8  # K

        # Simulated logits from lam_head (requires_grad for backprop)
        logits = torch.randn(batch_size, code_seq_len, codebook_size, requires_grad=True)

        # Simulated target codes
        codes = torch.randint(0, codebook_size, (batch_size, code_seq_len))

        # All valid pairs
        valid_pair = torch.ones(batch_size, dtype=torch.bool)

        # Compute masked loss
        valid_logits = logits[valid_pair]
        valid_codes = codes[valid_pair]

        loss = F.cross_entropy(
            valid_logits.reshape(-1, codebook_size),
            valid_codes.reshape(-1)
        )

        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad

    def test_ce_loss_with_masking(self):
        """Test CE loss with partial valid pairs."""
        batch_size = 4
        code_seq_len = 4
        codebook_size = 8

        logits = torch.randn(batch_size, code_seq_len, codebook_size)
        codes = torch.randint(0, codebook_size, (batch_size, code_seq_len))

        # Only first two samples are valid
        valid_pair = torch.tensor([True, True, False, False])

        valid_logits = logits[valid_pair]  # [2, 4, 8]
        valid_codes = codes[valid_pair]  # [2, 4]

        assert valid_logits.shape == (2, code_seq_len, codebook_size)
        assert valid_codes.shape == (2, code_seq_len)

        loss = F.cross_entropy(
            valid_logits.reshape(-1, codebook_size),
            valid_codes.reshape(-1)
        )

        assert loss.ndim == 0

    def test_ce_loss_no_valid_pairs(self):
        """Test CE loss when no pairs are valid."""
        batch_size = 4
        code_seq_len = 4
        codebook_size = 8

        logits = torch.randn(batch_size, code_seq_len, codebook_size)
        codes = torch.randint(0, codebook_size, (batch_size, code_seq_len))

        # No valid pairs
        valid_pair = torch.zeros(batch_size, dtype=torch.bool)

        valid_logits = logits[valid_pair]  # [0, 4, 8]
        valid_codes = codes[valid_pair]  # [0, 4]

        assert valid_logits.numel() == 0

        # Handle empty case as in the actual implementation
        if valid_logits.numel() > 0:
            loss = F.cross_entropy(
                valid_logits.reshape(-1, codebook_size),
                valid_codes.reshape(-1)
            )
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_accuracy_computation(self):
        """Test accuracy computation with predictions."""
        batch_size = 4
        code_seq_len = 4
        codebook_size = 8

        # Create logits where predictions match targets
        codes = torch.randint(0, codebook_size, (batch_size, code_seq_len))
        logits = torch.zeros(batch_size, code_seq_len, codebook_size)

        # Set high logit for correct class
        for b in range(batch_size):
            for s in range(code_seq_len):
                logits[b, s, codes[b, s]] = 10.0

        valid_pair = torch.ones(batch_size, dtype=torch.bool)
        valid_logits = logits[valid_pair]
        valid_codes = codes[valid_pair]

        preds = valid_logits.argmax(dim=-1)
        accuracy = (preds == valid_codes).float().mean()

        assert accuracy.item() == 1.0  # Perfect accuracy

    def test_accuracy_with_masking(self):
        """Test accuracy computation with masked samples."""
        batch_size = 4
        code_seq_len = 4
        codebook_size = 8

        codes = torch.randint(0, codebook_size, (batch_size, code_seq_len))
        logits = torch.zeros(batch_size, code_seq_len, codebook_size)

        # Only set correct logits for first 2 samples
        for b in range(2):
            for s in range(code_seq_len):
                logits[b, s, codes[b, s]] = 10.0

        # Only first 2 are valid (and have correct predictions)
        valid_pair = torch.tensor([True, True, False, False])
        valid_logits = logits[valid_pair]
        valid_codes = codes[valid_pair]

        preds = valid_logits.argmax(dim=-1)
        accuracy = (preds == valid_codes).float().mean()

        assert accuracy.item() == 1.0


class TestLatentSmolConfig:
    """Tests for LatentSmolConfig."""

    def test_config_head_mode_latent(self):
        """Test config defaults for latent mode."""
        from lerobot.policies.latent_smol.configuration_latent_smol import LatentSmolConfig

        config = LatentSmolConfig()
        assert config.head_mode == "latent"
        assert config.freeze_vision_encoder is False
        assert config.train_expert_only is False

    def test_config_observation_delta_indices_latent(self):
        """Test observation_delta_indices for latent mode."""
        from lerobot.policies.latent_smol.configuration_latent_smol import LatentSmolConfig

        config = LatentSmolConfig(lam_future_frames=10)
        assert config.observation_delta_indices == [0, 10]

    def test_config_observation_delta_indices_action(self):
        """Test observation_delta_indices for action mode."""
        from lerobot.policies.latent_smol.configuration_latent_smol import LatentSmolConfig

        config = LatentSmolConfig(head_mode="action")
        assert config.observation_delta_indices == [0]

    def test_config_latent_mode_requires_future_frames(self):
        """Test that latent mode raises error if lam_future_frames not set."""
        from lerobot.policies.latent_smol.configuration_latent_smol import LatentSmolConfig

        config = LatentSmolConfig(head_mode="latent", lam_future_frames=-1)
        with pytest.raises(ValueError, match="lam_future_frames must be >0"):
            _ = config.observation_delta_indices


class TestPoolHidden:
    """Tests for masked mean pooling function."""

    def test_pool_hidden_all_valid(self):
        """Test pooling with all valid tokens."""
        from lerobot.policies.latent_smol.modeling_latent_smol import _pool_hidden

        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        hidden = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        pooled = _pool_hidden(hidden, mask)

        assert pooled.shape == (batch_size, hidden_dim)
        # Should be equal to simple mean when all valid
        expected = hidden.mean(dim=1)
        assert torch.allclose(pooled, expected)

    def test_pool_hidden_partial_valid(self):
        """Test pooling with partial valid tokens."""
        from lerobot.policies.latent_smol.modeling_latent_smol import _pool_hidden

        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        hidden = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, :5] = True  # Only first 5 tokens valid

        pooled = _pool_hidden(hidden, mask)

        assert pooled.shape == (batch_size, hidden_dim)
        # Should be mean of first 5 tokens only
        expected = hidden[:, :5].mean(dim=1)
        assert torch.allclose(pooled, expected)

    def test_pool_hidden_different_valid_per_batch(self):
        """Test pooling with different valid lengths per batch."""
        from lerobot.policies.latent_smol.modeling_latent_smol import _pool_hidden

        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        hidden = torch.randn(batch_size, seq_len, hidden_dim)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[0, :3] = True  # First batch: 3 valid tokens
        mask[1, :7] = True  # Second batch: 7 valid tokens

        pooled = _pool_hidden(hidden, mask)

        assert pooled.shape == (batch_size, hidden_dim)
        # Check each batch independently
        assert torch.allclose(pooled[0], hidden[0, :3].mean(dim=0))
        assert torch.allclose(pooled[1], hidden[1, :7].mean(dim=0))
