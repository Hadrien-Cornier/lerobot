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

"""CPU dry run of the training CLI, including real frozen components and checkpoint state."""

import json
import os
import subprocess
import sys
from pathlib import Path

import draccus
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("diffusers")
pytest.importorskip("datasets")
pytest.importorskip("accelerate")

from lerobot.configs.default import DatasetConfig  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.zero_wam.configuration_zero_wam import ZeroWAMConfig  # noqa: E402


def test_training_cli_resume_matches_uninterrupted(tmp_path: Path, frozen_components: Path) -> None:
    camera = "observation.images.image"
    dataset_root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="test/zero-wam-resume",
        root=dataset_root,
        fps=10,
        use_videos=False,
        image_writer_threads=1,
        features={
            camera: {"dtype": "image", "shape": (3, 32, 32), "names": ["channels", "height", "width"]},
            "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        },
    )
    rng = np.random.default_rng(1)
    for _ in range(2):
        for _ in range(12):
            dataset.add_frame(
                {
                    camera: rng.integers(0, 256, size=(3, 32, 32), dtype=np.uint8),
                    "action": rng.uniform(-1, 1, size=2).astype(np.float32),
                    "task": "move",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    policy = ZeroWAMConfig(
        num_attention_heads=2,
        attention_head_dim=24,
        in_channels=48,
        out_channels=48,
        action_dim=2,
        text_dim=16,
        freq_dim=16,
        ffn_dim=64,
        num_layers=2,
        action_inner_dim=48,
        action_ffn_dim=64,
        num_mcp_modules=2,
        mcp_hidden_collect_layers=(0, 1),
        mcp_loss_weights=(0.5, 0.25),
        rope_max_seq_len=64,
        attention_backend="sdpa",
        gradient_checkpointing=False,
        dtype=torch.float32,
        frozen_dtype=torch.float32,
        device="cpu",
        frozen_components_path=str(frozen_components),
        max_sequence_length=8,
        obs_cam_keys=[camera],
        height=32,
        width=32,
        video_frame_stride=1,
        frame_chunk_size=3,
        used_action_channel_ids=[0, 1],
        train_num_latent_frames=5,
        scheduler_warmup_steps=4,
        push_to_hub=False,
    )
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/zero-wam-resume", root=dataset_root),
        policy=policy,
        output_dir=tmp_path / "uninterrupted",
        num_workers=0,
        batch_size=2,
        steps=4,
        save_freq=2,
        log_freq=1,
        env_eval_freq=0,
        seed=123,
    )
    config_path = tmp_path / "train.json"
    with draccus.config_type("json"), config_path.open("w") as stream:
        draccus.dump(cfg, stream, indent=2)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    first = subprocess.run(
        [*command, f"--config_path={config_path}"], env=env, capture_output=True, text=True, timeout=120
    )
    assert first.returncode == 0, first.stdout + first.stderr
    checkpoint = cfg.output_dir / "checkpoints" / "000002"
    resumed_dir = tmp_path / "resumed"
    second = subprocess.run(
        [
            *command,
            f"--config_path={checkpoint / 'pretrained_model' / 'train_config.json'}",
            "--resume=true",
            f"--output_dir={resumed_dir}",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    expected = cfg.output_dir / "checkpoints" / "000004"
    actual = resumed_dir / "checkpoints" / "000004"
    for relative_path in (
        "pretrained_model/model.safetensors",
        "training_state/optimizer_state.safetensors",
        "training_state/rng_state.safetensors",
    ):
        left, right = load_file(expected / relative_path), load_file(actual / relative_path)
        assert left.keys() == right.keys(), relative_path
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=f"{relative_path}: {key}")
    for filename in ("training_step.json", "scheduler_state.json", "optimizer_param_groups.json"):
        left = json.loads((expected / "training_state" / filename).read_text())
        right = json.loads((actual / "training_state" / filename).read_text())
        assert left == right, filename
    assert json.loads((actual / "training_state" / "training_step.json").read_text())["step"] == 4
    initial = load_file(checkpoint / "pretrained_model" / "model.safetensors")
    final = load_file(actual / "pretrained_model" / "model.safetensors")
    assert any(not torch.equal(initial[key], final[key]) for key in initial)
