#!/usr/bin/env python

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

"""Plugin registration, one QF3 update, and flow-actor select_action."""

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

import math  # noqa: E402

from lerobot.rl.algorithms.qf3.smoke import run_api_smoke  # noqa: E402


def test_qf3_api_smoke():
    result = run_api_smoke()
    assert result["policy_type"] == "flow_actor"
    assert result["algorithm_type"] == "qf3"
    assert result["actor_param_l2"] > 0.0
    assert result["critic_param_l2"] > 0.0
    assert result["filtered_grad_max_error"] < 1e-5
    assert result["select_action_zero_init_max_abs_error"] < 1e-6
    losses = result["losses"]
    extra = result["extra"]
    assert isinstance(losses, dict)
    assert isinstance(extra, dict)
    assert math.isfinite(losses["loss_actor"])
    assert math.isfinite(losses["loss_critic"])
    assert extra["bound_violation"] < 1e-5
    assert result["optimization_step"] == 1
