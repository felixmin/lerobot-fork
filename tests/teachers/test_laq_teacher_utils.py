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

"""Unit tests for LAQ teacher utility functions."""

import pytest
import torch


class TestValidPairFromIsPad:
    """Tests for valid_pair_from_is_pad function."""

    def test_valid_pair_from_is_pad_basic(self):
        """Test basic valid pair mask computation."""
        from lerobot.teachers.laq_teacher import valid_pair_from_is_pad

        # Test various combinations of padding
        is_pad = torch.tensor([
            [False, False],  # Both valid -> True
            [True, False],   # First padded -> False
            [False, True],   # Second padded -> False
            [True, True],    # Both padded -> False
        ])
        valid = valid_pair_from_is_pad(is_pad)
        assert valid.tolist() == [True, False, False, False]

    def test_valid_pair_from_is_pad_all_valid(self):
        """Test when all pairs are valid."""
        from lerobot.teachers.laq_teacher import valid_pair_from_is_pad

        is_pad = torch.tensor([
            [False, False],
            [False, False],
            [False, False],
        ])
        valid = valid_pair_from_is_pad(is_pad)
        assert valid.all()
        assert valid.shape == (3,)

    def test_valid_pair_from_is_pad_none_valid(self):
        """Test when no pairs are valid."""
        from lerobot.teachers.laq_teacher import valid_pair_from_is_pad

        is_pad = torch.tensor([
            [True, False],
            [False, True],
            [True, True],
        ])
        valid = valid_pair_from_is_pad(is_pad)
        assert not valid.any()
        assert valid.shape == (3,)

    def test_valid_pair_from_is_pad_empty(self):
        """Test with empty batch."""
        from lerobot.teachers.laq_teacher import valid_pair_from_is_pad

        is_pad = torch.zeros(0, 2, dtype=torch.bool)
        valid = valid_pair_from_is_pad(is_pad)
        assert valid.shape == (0,)


class TestFramesPermutation:
    """Tests for frame tensor permutation used in LAQ teacher."""

    def test_frames_permutation_shape(self):
        """Test that permutation produces correct shape for video format."""
        # Input: [B, 2, C, H, W] -> Output: [B, C, 2, H, W]
        frames = torch.randn(2, 2, 3, 64, 64)
        video = frames.permute(0, 2, 1, 3, 4)
        assert video.shape == (2, 3, 2, 64, 64)

    def test_frames_permutation_values(self):
        """Test that permutation preserves values correctly."""
        frames = torch.randn(2, 2, 3, 64, 64)
        video = frames.permute(0, 2, 1, 3, 4)

        # Video[:, :, 0] should equal frames[:, 0] (frame at t=0)
        assert torch.allclose(video[:, :, 0], frames[:, 0])

        # Video[:, :, 1] should equal frames[:, 1] (frame at t=delta)
        assert torch.allclose(video[:, :, 1], frames[:, 1])

    def test_frames_permutation_batch_independence(self):
        """Test that permutation maintains batch independence."""
        batch_size = 4
        frames = torch.randn(batch_size, 2, 3, 32, 32)

        # Set each batch element to a different constant for easy verification
        for b in range(batch_size):
            frames[b] = b

        video = frames.permute(0, 2, 1, 3, 4)

        # Each batch element should still have its constant value
        for b in range(batch_size):
            assert torch.allclose(video[b], torch.full_like(video[b], b))


class TestLAQTeacherConfig:
    """Tests for LAQTeacherConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from lerobot.teachers.laq_teacher import LAQTeacherConfig

        config = LAQTeacherConfig(checkpoint_path="/path/to/checkpoint")
        assert config.checkpoint_path == "/path/to/checkpoint"
        assert config.device == "cuda"

    def test_config_custom_device(self):
        """Test custom device configuration."""
        from lerobot.teachers.laq_teacher import LAQTeacherConfig

        config = LAQTeacherConfig(checkpoint_path="/path/to/checkpoint", device="cpu")
        assert config.device == "cpu"
