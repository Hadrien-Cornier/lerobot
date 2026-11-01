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

"""Offline CPU oracles for the pinned reference's robot-video and UMT5 preprocessing.

Reference: robbyant-research/Zero-WAM, 08e2c4ae41e2b63573a299825cebe6753481407c,
wan_va/wan_va_server.py: VA_Server._encode_initial_obs, _encode_obs,
_get_t5_prompt_embeds and normalize_latents. The numerical oracles are inline so these
tests do not need a reference checkout, network access or its CUDA server initialization.
These tests cover robot observations and ASCII task prompts, not ICL video decoding or
the documented difference in action quantile denominators.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytest.importorskip("diffusers")

from diffusers.pipelines.wan.pipeline_wan import prompt_clean  # noqa: E402

from lerobot.policies.zero_wam.modeling_zero_wam import ZeroWAMPolicy  # noqa: E402
from tests.policies.zero_wam.test_zero_wam import tiny_config  # noqa: E402


@pytest.fixture
def real_policy(frozen_components: Path) -> ZeroWAMPolicy:
    policy = ZeroWAMPolicy(
        tiny_config(in_channels=48, out_channels=48, frozen_components_path=str(frozen_components))
    )
    policy._ensure_frozen_modules()
    return policy


@torch.no_grad()
def test_robot_camera_preprocessing_matches_reference(real_policy: ZeroWAMPolicy) -> None:
    policy = real_policy
    vae = policy._vae
    rng = np.random.default_rng(3)
    # Distinct cameras, non-square source images, a first frame and two continuation chunks.
    images = rng.integers(0, 256, size=(3, 9, 48, 64, 3), dtype=np.uint8)
    clips = [torch.from_numpy(camera).permute(0, 3, 1, 2).float() / 255 for camera in images]
    cache = [None] * vae._cached_conv_counts["encoder"]
    for start, end in ((0, 1), (1, 5), (5, 9)):
        videos = []
        for camera in images:
            if start == 0:
                # The reference initial-observation path scales before interpolation.
                frames = torch.from_numpy(camera[0].copy()).float() / 255 * 2 - 1
                frames = F.interpolate(
                    frames.permute(2, 0, 1)[None], size=(32, 32), mode="bilinear", align_corners=False
                ).unsqueeze(2)
            else:
                # Its continuation path interpolates uint8 values before scaling.
                frames = torch.from_numpy(camera[start:end].copy()).float().permute(3, 0, 1, 2)
                frames = F.interpolate(frames, size=(32, 32), mode="bilinear", align_corners=False)[None]
                frames = frames / 255 * 2 - 1
            videos.append(frames)
        video = torch.cat(videos)
        # Pinned reference modules/utils.py: patchify then cached encoder and quant_conv.
        b, c, t, h, w = video.shape
        video = video.view(b, c, t, h // 2, 2, w // 2, 2)
        video = video.permute(0, 1, 6, 4, 2, 3, 5).contiguous().view(b, c * 4, t, h // 2, w // 2)
        encoded = vae.quant_conv(vae.encoder(video, feat_cache=cache, feat_idx=[0]))
        mu, _ = encoded.chunk(2, dim=1)
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inv_std = (1 / torch.tensor(vae.config.latents_std)).view(1, -1, 1, 1, 1)
        expected = torch.cat((((mu.float() - mean) * inv_std).to(mu)).split(1), dim=-1)
        actual = policy._encode_cameras([camera[start:end] for camera in clips], streaming=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prompt", ["", " move &amp;amp;  move\n", "move " * 20])
@torch.no_grad()
def test_text_preprocessing_matches_reference(real_policy: ZeroWAMPolicy, prompt: str) -> None:
    policy = real_policy
    policy._ensure_text_encoder()
    tokens = policy._frozen["tokenizer"](
        [prompt_clean(prompt)],
        padding="max_length",
        max_length=policy.config.max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    hidden = policy._frozen["text_encoder"](tokens.input_ids, tokens.attention_mask).last_hidden_state
    length = int(tokens.attention_mask.gt(0).sum())
    expected = torch.cat([hidden[:, :length], torch.zeros_like(hidden[:, length:])], dim=1)
    torch.testing.assert_close(policy._encode_text(prompt), expected, rtol=0, atol=0)
