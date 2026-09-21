"""Run real LeRobot evaluation on deterministic Gymnasium environments.

Select the unchanged before/after checkout with PYTHONPATH. No evaluation or
rollout function is mocked. The environment and constant policy are teaching fixtures.
"""

import argparse
import json
import subprocess
from contextlib import closing
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch
from lerobot.envs.utils import FreezeAfterEpisodeEnd
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy

from lerobot.scripts import lerobot_eval


class TeachingPolicy(PreTrainedPolicy):
    config_class = ACTConfig
    name = "evaluation_teaching_fixture"

    def __init__(self):
        super().__init__(ACTConfig(device="cpu"))

    def reset(self):
        pass

    def get_optim_params(self):
        return {}

    def forward(self, batch):
        raise NotImplementedError("This fixture is for inference only")

    def select_action(self, batch, **kwargs):
        return torch.zeros((batch["observation.state"].shape[0], 1))

    def predict_action_chunk(self, batch, **kwargs):
        return self.select_action(batch).unsqueeze(1)


class Identity:
    def __call__(self, value):
        return value

    def reset(self):
        pass


class TeachingEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_fps": 10}
    task = "count_steps"
    task_description = "Deterministic reward fixture; no robot physics"
    _max_episode_steps = 4

    def __init__(self, short, contaminate):
        self.short = short
        self.contaminate = contaminate
        self.episode = -1
        self.step_index = 0
        self.events = []
        self.observation_space = gym.spaces.Dict(
            {"agent_pos": gym.spaces.Box(-100, 100, (1,), np.float32)}
        )
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)

    def observation(self):
        return {"agent_pos": np.array([self.step_index], dtype=np.float32)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.step_index = 0
        return self.observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.step_index += 1
        if self.short and self.contaminate and self.episode > 0:
            reward, terminated, success = 100.0, True, True
        elif self.short:
            reward = float([-2, -1][self.step_index - 1])
            terminated, success = self.step_index == 2, False
        else:
            reward = float(self.step_index)
            terminated = self.step_index == 4
            success = terminated
        self.events.append(
            {
                "episode": self.episode,
                "step": self.step_index,
                "reward": reward,
                "terminated": terminated,
                "success": success,
            }
        )
        return self.observation(), reward, terminated, False, {"is_success": success}


def make_env(scenario):
    def factory(short):
        env = TeachingEnv(short, scenario == "next_episode")
        return FreezeAfterEpisodeEnd(env) if scenario == "frozen_tail" else env

    return gym.vector.SyncVectorEnv(
        [lambda: factory(True), lambda: factory(False)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def run(scenario):
    processors = {
        key: Identity()
        for key in (
            "env_preprocessor",
            "env_postprocessor",
            "preprocessor",
            "postprocessor",
        )
    }
    # Separate fresh environments make raw-rollout inspection independent of evaluation.
    with closing(make_env(scenario)) as env:
        raw = lerobot_eval.rollout(env, TeachingPolicy(), **processors)
        events = [sub.unwrapped.events for sub in env.envs]
    with closing(make_env(scenario)) as env:
        result = lerobot_eval.eval_policy(
            env, TeachingPolicy(), **processors, n_episodes=2
        )
    done_index = raw["done"].int().argmax(dim=1)
    valid = torch.arange(raw["done"].shape[1])[None, :] <= done_index[:, None]
    # Pedagogical calculation only: the actual result above comes from eval_policy.
    boundary_only_max = (raw["reward"] * valid).max(dim=1).values.tolist()
    expected = [
        {"sum_reward": -3.0, "max_reward": -1.0, "success": False},
        {"sum_reward": 10.0, "max_reward": 4.0, "success": True},
    ]
    actual = [{k: ep[k] for k in expected[0]} for ep in result["per_episode"]]
    return {
        "scenario": scenario,
        "raw": {k: raw[k].tolist() for k in ("reward", "done", "success")},
        "environment_step_events": events,
        "valid_mask": valid.tolist(),
        "boundary_only_max": boundary_only_max,
        "expected": expected,
        "actual": actual,
        "matches_expected": actual == expected,
        "aggregated": {
            k: v for k, v in result["aggregated"].items() if not k.startswith("eval_")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect", choices=["bug", "fixed"], required=True)
    args = parser.parse_args()
    source = Path(lerobot_eval.__file__).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    cases = [run(name) for name in ("frozen_tail", "negative_tail", "next_episode")]
    report = {
        "revision": revision,
        "source": str(source),
        "gymnasium": gym.__version__,
        "torch": torch.__version__,
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert all(case["matches_expected"] == (args.expect == "fixed") for case in cases)
    if args.expect == "bug":
        buggy = [(-3.0, 0.0, False), (-5.0, 0.0, False), (97.0, 100.0, True)]
        for case, (total, maximum, success) in zip(cases, buggy, strict=True):
            assert case["actual"][0] == {
                "sum_reward": total,
                "max_reward": maximum,
                "success": success,
            }
            assert case["actual"][1] == case["expected"][1]
