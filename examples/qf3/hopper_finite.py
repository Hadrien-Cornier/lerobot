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

"""One finite Hopper-v4 run of QF3 or the direct-Q ablation.

This is a single-process CleanRL-style loop around ``QF3Algorithm``. It is the
bounded check against figure ``fig:mujoco-curves``, not that figure's 500k-step,
3-seed, four-task comparison. The actor loss is ``qf3_actor_terms``.

Deviations from the paper and from CleanRL's ``td3_continuous_action.py`` are
written into ``config.json`` in the output directory. The script does not add
exploration noise on top of the flow sample, and the critic bootstrap action
stays unclamped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.flow_actor.configuration_flow_actor import FlowActorConfig
from lerobot.policies.flow_actor.modeling_flow_actor import FlowActorPolicy
from lerobot.rl.algorithms.qf3.configuration_qf3 import QF3AlgorithmConfig
from lerobot.rl.algorithms.qf3.qf3_algorithm import QF3Algorithm
from lerobot.utils.constants import ACTION, OBS_STATE

# Hopper pair selected in table tab:mujoco-hyperparameters.
HOPPER_ALPHA = 1.0
HOPPER_LAMBDA_CFM = 0.2
# CleanRL td3_continuous_action.py defaults read 2026-10-22. The paper inherits
# the harness and does not restate these.
LEARNING_STARTS = 25_000
BATCH_SIZE = 256
BUFFER_SIZE = 1_000_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 5


class Replay:
    """Fixed ring buffer. ``done`` is true termination, not a time limit."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        i = self.pos
        self.state[i] = state
        self.action[i] = action
        self.reward[i, 0] = reward
        self.next_state[i] = next_state
        self.done[i, 0] = done
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        index = rng.integers(0, len(self), size=batch_size)
        return {
            "state": torch.from_numpy(self.state[index]),
            "action": torch.from_numpy(self.action[index]),
            "reward": torch.from_numpy(self.reward[index]),
            "next_state": torch.from_numpy(self.next_state[index]),
            "done": torch.from_numpy(self.done[index]),
        }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _flat(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in module.parameters()])


def _require_mujoco():
    try:
        import gymnasium as gym
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Hopper-v4 needs gymnasium, mujoco, and imageio. Run with `uv run --with mujoco "
            "--with imageio python examples/qf3/hopper_finite.py ...`."
        ) from exc
    return gym, mujoco


def _build(seed: int, variant: str, state_dim: int, action_dim: int) -> tuple[FlowActorPolicy, QF3Algorithm]:
    torch.manual_seed(seed)
    policy_config = FlowActorConfig(
        device="cpu",
        hidden_dims=[256, 256],
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
    )
    policy = FlowActorPolicy(policy_config)
    algorithm_config = QF3AlgorithmConfig(
        policy_config=policy_config,
        alpha=HOPPER_ALPHA,
        lambda_cfm=HOPPER_LAMBDA_CFM,
        clip_velocity=variant == "qf3",
        hidden_dims=[256, 256],
        euler_steps=policy_config.euler_steps,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        polyak_tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_frequency=2,
    )
    algorithm = QF3Algorithm(policy, algorithm_config)
    algorithm.make_optimizers_and_scheduler()
    return policy, algorithm


def _policy_action(policy: FlowActorPolicy, state: np.ndarray, noise: torch.Tensor | None) -> np.ndarray:
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    batch = {OBS_STATE: state_tensor}
    action = policy.select_action(batch) if noise is None else policy.select_action(batch, noise=noise)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _evaluate(
    policy: FlowActorPolicy,
    env,
    env_step: int,
    episodes: int,
    seed: int,
    writer: csv.DictWriter,
) -> list[float]:
    """Five episodes from epsilon = 0. The policy weights are not updated."""
    returns: list[float] = []
    action_dim = int(np.prod(env.action_space.shape))
    zeros = torch.zeros(1, action_dim)
    for episode_index in range(episodes):
        # Fixed per episode index, not per checkpoint, so an unchanged policy
        # repeats the same returns. env_step is only the log key.
        state, _ = env.reset(seed=seed + 100_000 + episode_index)
        state = np.asarray(state, dtype=np.float32)
        total = 0.0
        length = 0
        done = False
        while not done:
            raw = _policy_action(policy, state, zeros)
            action = np.clip(raw, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = np.asarray(next_state, dtype=np.float32)
            total += float(reward)
            length += 1
            done = bool(terminated or truncated)
        returns.append(total)
        writer.writerow(
            {
                "env_step": env_step,
                "episode_index": episode_index,
                "return": total,
                "length": length,
            }
        )
    return returns


def _write_summary(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite Hopper-v4 QF3 or direct-Q run.")
    parser.add_argument("--variant", choices=("qf3", "direct_q"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prediction-file", type=Path)
    parser.add_argument("--require-prediction", action="store_true")
    parser.add_argument("--git-sha", default="", help="Recorded SHA. Used when the run tree has no .git.")
    args = parser.parse_args()
    if args.total_steps < 1:
        raise SystemExit("--total-steps must be positive")
    if args.require_prediction and (args.prediction_file is None or not args.prediction_file.is_file()):
        raise SystemExit("--require-prediction needs an existing --prediction-file")

    gym, mujoco = _require_mujoco()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    replay_rng = np.random.default_rng(args.seed)

    env = gym.make("Hopper-v4")
    eval_env = gym.make("Hopper-v4")
    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    policy, algorithm = _build(args.seed, args.variant, state_dim, action_dim)
    actor_init = _flat(policy.velocity).clone()
    critic_init = _flat(algorithm.critic).clone()
    replay = Replay(args.buffer_size, state_dim, action_dim)

    args.out.mkdir(parents=True, exist_ok=True)
    prediction_sha = (
        _sha256(args.prediction_file) if args.prediction_file and args.prediction_file.is_file() else ""
    )
    config = {
        "label": "simulation",
        "env_id": "Hopper-v4",
        "variant": args.variant,
        "clip_velocity": algorithm.config.clip_velocity,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "alpha": algorithm.config.alpha,
        "lambda_cfm": algorithm.config.lambda_cfm,
        "gamma": algorithm.config.gamma,
        "polyak_tau": algorithm.config.polyak_tau,
        "policy_noise": algorithm.config.policy_noise,
        "noise_clip": algorithm.config.noise_clip,
        "policy_frequency": algorithm.config.policy_frequency,
        "actor_lr": algorithm.config.actor_lr,
        "critic_lr": algorithm.config.critic_lr,
        "euler_steps": algorithm.config.euler_steps,
        "hidden_dims": [256, 256],
        "threads": args.threads,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "action_low": env.action_space.low.tolist(),
        "action_high": env.action_space.high.tolist(),
        "git_sha": args.git_sha or _git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "mujoco": mujoco.__version__,
        "prediction_sha256": prediction_sha,
        "deviations": [
            "Horizon and seed count are the CLI values, not 500k steps and 3 seeds.",
            "direct_q removes the velocity clip and keeps actor lr 3e-4. The figure compares QSM, FlowRL, and DIPO, not this ablation.",
            "No extra N(0, 0.1) exploration noise. After warmup the executed action is the flow sample, clipped to the Hopper box.",
            "Critic bootstrap actions are not clamped and not passed through tanh.",
            "Critic hidden activation is SiLU. CleanRL TD3 uses ReLU.",
            "Eval actions use epsilon=0. The paper's 10k and 20k Hopper means differ while several methods still match.",
            "TD bootstrap uses true termination only. Time-limit truncation still bootstraps.",
        ],
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    episode_file = (args.out / "episodes.csv").open("w", newline="")
    update_file = (args.out / "updates.csv").open("w", newline="")
    eval_file = (args.out / "evals.csv").open("w", newline="")
    episode_writer = csv.DictWriter(
        episode_file, fieldnames=["env_step", "episode_index", "return", "length"]
    )
    update_writer = csv.DictWriter(
        update_file,
        fieldnames=[
            "env_step",
            "critic_loss",
            "actor_updated",
            "actor_loss",
            "cfm",
            "q_mean",
            "mask_fraction",
            "bound_violation",
        ],
    )
    eval_writer = csv.DictWriter(eval_file, fieldnames=["env_step", "episode_index", "return", "length"])
    episode_writer.writeheader()
    update_writer.writeheader()
    eval_writer.writeheader()

    stop = {"flag": False}

    def _stop(signum: int, frame: object) -> None:
        del frame
        stop["flag"] = True
        print(f"signal {signum}: will stop after this step", flush=True)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    started = time.time()
    status = "complete"
    actor_updates = 0
    critic_updates = 0
    first_actor_loss = None
    last_actor_loss = None
    first_critic_loss = None
    last_critic_loss = None
    eval_means: dict[str, float] = {}
    episode_index = 0
    episode_return = 0.0
    episode_length = 0
    completed_steps = 0

    state, _ = env.reset(seed=args.seed)
    state = np.asarray(state, dtype=np.float32)
    env.action_space.seed(args.seed)
    zero_returns = _evaluate(policy, eval_env, 0, args.eval_episodes, args.seed, eval_writer)
    eval_file.flush()
    eval_means["0"] = float(np.mean(zero_returns))
    print(f"eval env_step=0 mean_return={eval_means['0']}", flush=True)

    for env_step in range(args.total_steps):
        if stop["flag"]:
            status = "stopped"
            break
        if env_step < args.learning_starts:
            action = np.asarray(env.action_space.sample(), dtype=np.float32)
        else:
            action = np.clip(_policy_action(policy, state, None), env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.asarray(next_state, dtype=np.float32)
        replay.add(state, action, float(reward), next_state, 1.0 if terminated else 0.0)
        episode_return += float(reward)
        episode_length += 1
        state = next_state
        if terminated or truncated:
            episode_writer.writerow(
                {
                    "env_step": env_step + 1,
                    "episode_index": episode_index,
                    "return": episode_return,
                    "length": episode_length,
                }
            )
            episode_index += 1
            episode_return = 0.0
            episode_length = 0
            state, _ = env.reset()
            state = np.asarray(state, dtype=np.float32)

        if env_step > args.learning_starts:
            algorithm.optimization_step = env_step
            stats = algorithm.update(iter([replay.sample(args.batch_size, replay_rng)]))
            critic_updates += 1
            critic_loss = stats.losses["loss_critic"]
            if first_critic_loss is None:
                first_critic_loss = critic_loss
            last_critic_loss = critic_loss
            actor_updated = "loss_actor" in stats.losses
            row = {
                "env_step": env_step,
                "critic_loss": critic_loss,
                "actor_updated": int(actor_updated),
                "actor_loss": "",
                "cfm": "",
                "q_mean": "",
                "mask_fraction": "",
                "bound_violation": "",
            }
            if actor_updated:
                actor_updates += 1
                actor_loss = stats.losses["loss_actor"]
                if first_actor_loss is None:
                    first_actor_loss = actor_loss
                last_actor_loss = actor_loss
                row["actor_loss"] = actor_loss
                row["cfm"] = stats.losses["loss_cfm"]
                row["q_mean"] = stats.extra["q_mean"]
                row["mask_fraction"] = stats.extra["mask_fraction"]
                row["bound_violation"] = stats.extra["bound_violation"]
            update_writer.writerow(row)
            if not np.isfinite(critic_loss) or (actor_updated and not np.isfinite(float(row["actor_loss"]))):
                status = "nonfinite"
                completed_steps = env_step + 1
                break

        completed = env_step + 1
        completed_steps = completed
        if completed % 1000 == 0:
            update_file.flush()
            episode_file.flush()
        if args.eval_interval > 0 and completed % args.eval_interval == 0:
            returns = _evaluate(policy, eval_env, completed, args.eval_episodes, args.seed, eval_writer)
            eval_file.flush()
            episode_file.flush()
            update_file.flush()
            mean_return = float(np.mean(returns))
            eval_means[str(completed)] = mean_return
            elapsed = time.time() - started
            print(
                f"eval env_step={completed} mean_return={mean_return} "
                f"sps={completed / elapsed:.2f} actor_updates={actor_updates}",
                flush=True,
            )

    wall_seconds = time.time() - started
    actor_l2 = torch.linalg.vector_norm(_flat(policy.velocity) - actor_init).item()
    critic_l2 = torch.linalg.vector_norm(_flat(algorithm.critic) - critic_init).item()
    summary = {
        "label": "simulation",
        "status": status,
        "variant": args.variant,
        "seed": args.seed,
        "env_steps_completed": completed_steps,
        "wall_seconds": wall_seconds,
        "actor_updates": actor_updates,
        "critic_updates": critic_updates,
        "actor_l2_from_init": actor_l2,
        "critic_l2_from_init": critic_l2,
        "first_critic_loss": first_critic_loss,
        "last_critic_loss": last_critic_loss,
        "first_actor_loss": first_actor_loss,
        "last_actor_loss": last_actor_loss,
        "eval_mean_return": eval_means,
        "training_episodes": episode_index,
        "git_sha": config["git_sha"],
        "prediction_sha256": prediction_sha,
    }
    # The step counter above is the last eval checkpoint, which misses a stop
    # between checkpoints. Record the true completed env steps from the loop.
    if status == "complete":
        summary["env_steps_completed"] = args.total_steps
    _write_summary(args.out / "summary.json", summary)
    episode_file.close()
    update_file.close()
    eval_file.close()
    env.close()
    eval_env.close()
    print(
        json.dumps({"status": status, "wall_seconds": wall_seconds, "eval_mean_return": eval_means}),
        flush=True,
    )
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
