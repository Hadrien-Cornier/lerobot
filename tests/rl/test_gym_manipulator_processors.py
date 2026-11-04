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

import gymnasium as gym
import numpy as np
import pytest
import torch

pytest.importorskip("datasets", reason="install lerobot[dataset]")

from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.processor import DataProcessorPipeline, TransitionKey, create_transition, identity_transition
from lerobot.processor.hil_processor import DISCRETE_PENALTY_KEY, TELEOP_ACTION_KEY
from lerobot.rl.gym_manipulator import make_processors
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_STATE


@pytest.fixture(params=[False, True], ids=["fresh", "reloaded"])
def gym_hil_processors(request: pytest.FixtureRequest, tmp_path: Path) -> tuple:
    # The gym_hil branch builds processors without accessing env or teleoperator hardware.
    processors = make_processors(gym.Env(), None, HILSerlRobotEnvConfig(name="gym_hil"), device="cpu")
    if not request.param:
        return processors
    reloaded = []
    for index, pipeline in enumerate(processors):
        directory = tmp_path / str(index)
        pipeline.save_pretrained(directory, config_filename="pipeline.json")
        reloaded.append(
            DataProcessorPipeline.from_pretrained(
                directory,
                config_filename="pipeline.json",
                to_transition=identity_transition,
                to_output=identity_transition,
            )
        )
    return tuple(reloaded)


def test_gym_hil_reset_without_action_or_teleoperation(gym_hil_processors: tuple) -> None:
    env_processor, _ = gym_hil_processors
    result = env_processor(create_transition(observation={"agent_pos": np.array([1.0, 2.0])}))
    torch.testing.assert_close(result[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[1.0, 2.0]]))
    assert result[TransitionKey.ACTION] is None
    assert result[TransitionKey.INFO] == {}
    assert result[TransitionKey.COMPLEMENTARY_DATA] == {}


@pytest.mark.parametrize("intervention", [False, True])
def test_gym_hil_step_preserves_teleop_data_and_routes_action(
    gym_hil_processors: tuple, intervention: bool
) -> None:
    env_processor, action_processor = gym_hil_processors
    teleop_action = np.array([0.5, -0.5], dtype=np.float32)
    result = env_processor(
        create_transition(
            observation={"agent_pos": np.array([1.0, 2.0])},
            action=np.array([0.0, 0.0], dtype=np.float32),
            info={
                "is_intervention": intervention,
                TELEOP_ACTION_KEY: teleop_action,
                DISCRETE_PENALTY_KEY: -0.1,
            },
        )
    )
    assert result[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION] is intervention
    complementary = result[TransitionKey.COMPLEMENTARY_DATA]
    torch.testing.assert_close(complementary[TELEOP_ACTION_KEY], torch.from_numpy(teleop_action))
    assert complementary[DISCRETE_PENALTY_KEY] == -0.1
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.zeros(1, 2))

    policy_action = torch.tensor([[0.1, 0.2]])
    result[TransitionKey.ACTION] = policy_action
    output = action_processor(result)
    expected = teleop_action if intervention else policy_action.numpy()[0]
    np.testing.assert_array_equal(output[TransitionKey.ACTION], expected)
