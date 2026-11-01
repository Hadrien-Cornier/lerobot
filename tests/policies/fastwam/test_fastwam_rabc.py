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
from types import SimpleNamespace

import pytest
import torch

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig  # noqa: E402
from lerobot.rewards.sarm.rabc import RABCWeights  # noqa: E402
from lerobot.utils.sample_weighting import SampleWeightingConfig, make_sample_weighter  # noqa: E402


@pytest.mark.parametrize("horizon", [8, 16])
def test_rabc_factory_uses_full_action_horizon(tmp_path: Path, horizon: int) -> None:
    # Execution can consume fewer actions than the model predicts. Progress weights
    # must describe the entire supervised target, rather than n_action_steps.
    cfg = FastWAMConfig(action_horizon=horizon, n_action_steps=4, base_model_id=None, device="cpu")
    progress = tmp_path / "progress.parquet"
    pd.DataFrame(
        {
            "index": range(40),
            "episode_index": [0] * 40,
            "progress_sparse": [0.0] * 20 + [1.0] * 10 + [0.0] * 10,
        }
    ).to_parquet(progress)
    device = torch.device("cpu")
    weighter = make_sample_weighter(
        SampleWeightingConfig(type="rabc", progress_path=str(progress)), SimpleNamespace(config=cfg), device
    )
    assert isinstance(weighter, RABCWeights)
    assert weighter.chunk_size == horizon
    batch = {"index": torch.tensor([0, 15, 25, 35])}
    expected = RABCWeights(progress, chunk_size=horizon, device=device).compute_batch_weights(batch)[0]
    actual = weighter.compute_batch_weights(batch)[0]
    torch.testing.assert_close(actual, expected)
    assert actual.unique().numel() > 1
