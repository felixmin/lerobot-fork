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

"""LatentSmol policy package.

Keep this module lightweight: importing config submodules should not eagerly import
heavy dependencies (e.g. transformers) via the policy model.
"""

from .configuration_latent_smol import LatentSmolConfig as LatentSmolConfig

__all__ = ["LatentSmolConfig"]

