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

"""One synthetic QF3 update and one flow-actor rollout.

This is a Dry run. It checks the registered plugin names, the filtered
gradient, ``select_action``, and a single ``QF3Algorithm.update``. It is not
a benchmark and it does not roll out an environment.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.flow_actor.configuration_flow_actor import FlowActorConfig
from lerobot.policies.flow_actor.modeling_flow_actor import FlowActorPolicy
from lerobot.rl.algorithms.factory import get_algorithm_class, make_algorithm, make_algorithm_config
from lerobot.rl.algorithms.qf3.configuration_qf3 import QF3AlgorithmConfig
from lerobot.rl.algorithms.qf3.losses import qf3_actor_terms
from lerobot.rl.algorithms.qf3.qf3_algorithm import QF3Algorithm
from lerobot.utils.constants import ACTION, OBS_STATE


def _flat(module: nn.Module) -> Tensor:
    return torch.cat([param.detach().reshape(-1) for param in module.parameters()])


class _FixedVelocity(nn.Module):
    def __init__(self, velocity: Tensor) -> None:
        super().__init__()
        self.velocity = nn.Parameter(velocity.clone())

    def forward(self, state: Tensor, action_sample: Tensor, flow_time: Tensor) -> Tensor:
        del action_sample, flow_time
        return self.velocity.expand(state.shape[0], -1)


class _SumQ(nn.Module):
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        del state
        return action.sum(dim=-1, keepdim=True)


def filtered_gradient_max_error() -> float:
    """Compare ``qf3_actor_terms`` autograd with the paper's strict mask.

    ``alpha = 1``, ``lambda_cfm = 0.3``, ``tau = 0.5``, ``Q = sum(action)``.
    Expected gradients are ``d(-Q)/dv + lambda_cfm * d(CFM)/dv``.
    """
    state = torch.zeros(1, 2)
    action = torch.zeros(1, 2)
    noise = torch.zeros(1, 2)
    flow_time = torch.tensor([[0.5]])
    q_net = _SumQ()
    cases = (
        (torch.tensor([[0.0, 3.0]]), torch.tensor([[-0.5, 1.8]])),
        (torch.tensor([[1.0, -1.0]]), torch.tensor([[0.6, -0.6]])),
        (torch.tensor([[0.25, -0.25]]), torch.tensor([[-0.35, -0.65]])),
    )
    max_error = 0.0
    for velocity, expected in cases:
        actor = _FixedVelocity(velocity)
        terms = qf3_actor_terms(
            actor,
            q_net,
            state,
            action,
            noise,
            flow_time,
            alpha=1.0,
            lambda_cfm=0.3,
            use_q=True,
            clip=True,
        )
        terms.loss.backward()
        if actor.velocity.grad is None:
            raise AssertionError("filtered gradient check produced no gradient")
        max_error = max(max_error, (actor.velocity.grad - expected).abs().max().item())
    if max_error > 1e-5:
        raise AssertionError(f"filtered gradient max error {max_error}")
    return max_error


def _synthetic_batch(batch_size: int = 8, state_dim: int = 2, action_dim: int = 2) -> dict[str, Tensor]:
    generator = torch.Generator()
    generator.manual_seed(0)
    state = torch.rand(batch_size, state_dim, generator=generator) * 2.0 - 1.0
    action = 0.25 * state
    reward = -((action - state).square().sum(dim=-1, keepdim=True))
    next_state = 0.5 * state
    done = torch.zeros(batch_size, 1)
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
    }


def _iterator(batch: dict[str, Tensor]) -> Iterator[dict[str, Tensor]]:
    yield batch


def run_api_smoke() -> dict[str, object]:
    """Register the plugins, roll out ``select_action``, and run one ``update``."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    gradient_error = filtered_gradient_max_error()

    policy_cls = get_policy_class("flow_actor")
    algorithm_cls = get_algorithm_class("qf3")
    if policy_cls is not FlowActorPolicy:
        raise TypeError(f"flow_actor resolved to {policy_cls}")
    if algorithm_cls is not QF3Algorithm:
        raise TypeError(f"qf3 resolved to {algorithm_cls}")

    config = FlowActorConfig(
        device="cpu",
        hidden_dims=[32, 32],
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )
    policy = policy_cls(config)
    if not isinstance(policy, FlowActorPolicy):
        raise TypeError(type(policy).__name__)
    if policy.velocity.output.weight.abs().sum().item() != 0.0:
        raise AssertionError("flow actor last layer was not zero at init")

    state = torch.tensor([[0.2, -0.4]])
    noise = torch.tensor([[0.3, -0.7]])
    rolled = policy.select_action({OBS_STATE: state}, noise=noise)
    if rolled.shape != (1, 2):
        raise AssertionError(f"select_action shape {tuple(rolled.shape)}")
    # Zero velocity field: Euler leaves the supplied noise unchanged.
    if (rolled - noise).abs().max().item() > 1e-6:
        raise AssertionError("zero-init Euler rollout did not return the supplied noise")
    zeros = policy.select_action({OBS_STATE: state}, noise=torch.zeros(1, 2))
    if zeros.abs().max().item() > 1e-6:
        raise AssertionError("zero-init rollout from epsilon=0 was not zero")

    pre, post = make_pre_post_processors(config)
    del pre, post

    algo_cfg = make_algorithm_config(
        "qf3",
        policy_config=config,
        hidden_dims=[32, 32],
        policy_frequency=1,
    )
    if not isinstance(algo_cfg, QF3AlgorithmConfig):
        raise TypeError(type(algo_cfg).__name__)
    algorithm = make_algorithm(algo_cfg, policy)
    if not isinstance(algorithm, QF3Algorithm):
        raise TypeError(type(algorithm).__name__)
    algorithm.make_optimizers_and_scheduler()

    actor_before = _flat(policy.velocity)
    critic_before = _flat(algorithm.critic)
    stats = algorithm.update(iter(_iterator(_synthetic_batch())))
    actor_l2 = torch.linalg.vector_norm(_flat(policy.velocity) - actor_before).item()
    critic_l2 = torch.linalg.vector_norm(_flat(algorithm.critic) - critic_before).item()
    if not (actor_l2 > 0.0 and critic_l2 > 0.0):
        raise AssertionError(f"parameters did not move: actor {actor_l2} critic {critic_l2}")
    for name, value in {**stats.losses, **stats.extra}.items():
        if not math.isfinite(value):
            raise AssertionError(f"{name} is not finite: {value}")
    if stats.extra["identity_error"] > 1e-5:
        raise AssertionError(f"path identity failed: {stats.extra['identity_error']}")
    if stats.extra["bound_violation"] > 1e-5:
        raise AssertionError(f"clip bound failed: {stats.extra['bound_violation']}")

    after = policy.select_action({OBS_STATE: state}, noise=noise)
    if after.shape != (1, 2) or not torch.isfinite(after).all():
        raise AssertionError("select_action after the update was not a finite action")

    return {
        "label": "dry_run",
        "seed": 0,
        "device": "cpu",
        "policy_type": policy.config.type,
        "algorithm_type": algorithm.config.type,
        "hidden_dims": [32, 32],
        "filtered_grad_max_error": gradient_error,
        "select_action_zero_init_max_abs_error": (rolled - noise).abs().max().item(),
        "actor_param_l2": actor_l2,
        "critic_param_l2": critic_l2,
        "losses": stats.losses,
        "extra": stats.extra,
        "optimization_step": algorithm.optimization_step,
    }
