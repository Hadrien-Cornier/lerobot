# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path

import pytest
import torch


@pytest.fixture
def frozen_components(tmp_path: Path) -> Path:
    """Real random-weight Wan/UMT5 assets; never download a released checkpoint."""
    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    root = tmp_path / "frozen"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        vae = diffusers.AutoencoderKLWan(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=48,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            dropout=0.0,
            is_residual=True,
            in_channels=12,
            out_channels=12,
            patch_size=2,
            scale_factor_spatial=16,
            scale_factor_temporal=4,
            latents_mean=[0.1] * 48,
            latents_std=[0.7] * 48,
        )
        vae.save_pretrained(root / "vae")
        encoder = transformers.UMT5EncoderModel(
            transformers.UMT5Config(
                vocab_size=16, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2, dropout_rate=0
            )
        )
        encoder.save_pretrained(root / "text_encoder")
    tokenizer = transformers.T5TokenizerFast(
        vocab=[("<pad>", 0.0), ("</s>", 0.0), ("<unk>", 0.0), ("▁", -2.0), ("move", -1.0)],
        extra_ids=0,
    )
    tokenizer.save_pretrained(root / "tokenizer")
    return root
