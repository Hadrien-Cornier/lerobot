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

"""CPU smoke test for the QF3 actor objective.

This is not a registered LeRobot algorithm and not a benchmark.
It checks the clipped one-step actor loss from Kim et al., arXiv:2610.08789v1
(section "QF3 Actor Objective"), on a fixed synthetic replay buffer.

Source labels used below:
- eq:actor-cfm-loss
- eq:clipped-reconstruction
- eq:actor-loss
- eq:q-gradient-mask
- the unclipped ablation L_direct in the same section
"""

from __future__ import annotations

import copy
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

# Paper Table tab:qf3-arch uses width 256. This smoke uses a smaller width.
SMOKE_HIDDEN = 32
# Single MuJoCo setting named in the analysis section (alpha=1, lambda_cfm=0.3).
PAPER_ALPHA = 1.0
PAPER_LAMBDA_CFM = 0.3
# Analysis section calls 3e-4 the default actor learning rate.
PAPER_ACTOR_LR = 3e-4
# Appendix: sinusoidal time embedding dimension and Euler steps.
PAPER_TIME_DIM = 64
PAPER_EULER_STEPS = 5


@dataclass(frozen=True)
class SmokeConfig:
    """Knobs for the dry-run. Paper values and CleanRL stand-ins are named in the report."""

    seed: int = 0
    state_dim: int = 2
    action_dim: int = 2
    hidden: int = SMOKE_HIDDEN
    time_dim: int = PAPER_TIME_DIM
    buffer_size: int = 128
    batch_size: int = 32
    critic_updates: int = 200
    alpha: float = PAPER_ALPHA
    lambda_cfm: float = PAPER_LAMBDA_CFM
    actor_lr: float = PAPER_ACTOR_LR
    critic_lr: float = PAPER_ACTOR_LR
    gamma: float = 0.99
    # CleanRL TD3 defaults. The paper says the MuJoCo critic matches that baseline
    # and does not restate these four numbers.
    polyak_tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    euler_steps: int = PAPER_EULER_STEPS


@dataclass
class ActorTerms:
    """Tensors from one actor-objective evaluation."""

    loss: Tensor
    cfm: Tensor
    q_mean: Tensor
    mask_fraction: Tensor
    identity_error: Tensor
    endpoint_gap: Tensor
    bound_violation: Tensor


class SinusoidalTimeEncoding(nn.Module):
    """Fixed sinusoidal embedding phi(tau) from the QF3 appendix."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2 or dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even and at least 2, got {dim}")
        half = dim // 2
        index = torch.arange(half, dtype=torch.float32)
        omega = torch.exp(-index * math.log(10000.0) / float(half - 1))
        self.register_buffer("omega", omega)

    def forward(self, flow_time: Tensor) -> Tensor:
        # flow_time is (B, 1). phi is [sin(tau * omega), cos(tau * omega)].
        angles = flow_time * self.omega.view(1, -1)
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class FlowActor(nn.Module):
    """MLP velocity field v(s, x_tau, tau). Last layer starts at zero, as in the appendix."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int, time_dim: int) -> None:
        super().__init__()
        self.time_enc = SinusoidalTimeEncoding(time_dim)
        in_dim = state_dim + action_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: Tensor, action_sample: Tensor, flow_time: Tensor) -> Tensor:
        encoded_time = self.time_enc(flow_time)
        return self.net(torch.cat((state, action_sample, encoded_time), dim=-1))


class QNetwork(nn.Module):
    """Scalar Q(s, a). Two of these form the twin critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        in_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((state, action), dim=-1))


class TwinQ(nn.Module):
    """TD3 twin Q. The actor uses q1 only, matching CleanRL TD3."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden)
        self.q2 = QNetwork(state_dim, action_dim, hidden)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state, action), self.q2(state, action)


class FixedVelocity(nn.Module):
    """Test double. The velocity is a parameter and does not depend on the input."""

    def __init__(self, velocity: Tensor) -> None:
        super().__init__()
        self.velocity = nn.Parameter(velocity.clone())

    def forward(self, state: Tensor, action_sample: Tensor, flow_time: Tensor) -> Tensor:
        del action_sample, flow_time
        return self.velocity.expand(state.shape[0], -1)


class SumActionQ(nn.Module):
    """Test double. Q equals the sum of action coordinates, so dQ/da_i = 1."""

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        del state
        return action.sum(dim=-1, keepdim=True)


def flow_path(noise: Tensor, replay_action: Tensor, flow_time: Tensor) -> tuple[Tensor, Tensor]:
    """Linear CFM path. x_tau = (1-tau) eps + tau a_buf, u_buf = a_buf - eps."""
    x_tau = (1.0 - flow_time) * noise + flow_time * replay_action
    u_buf = replay_action - noise
    return x_tau, u_buf


def qf3_actor_terms(
    actor: nn.Module,
    q1: nn.Module,
    state: Tensor,
    replay_action: Tensor,
    noise: Tensor,
    flow_time: Tensor,
    *,
    alpha: float,
    lambda_cfm: float,
    use_q: bool,
    clip: bool,
) -> ActorTerms:
    """QF3 actor objective, or the named ablation of it.

    clip=True is eq:actor-loss. clip=False is the paper's direct-Q ablation.
    use_q=False keeps only the CFM term. The CFM residual is never clipped.
    """
    x_tau, u_buf = flow_path(noise, replay_action, flow_time)
    velocity = actor(state, x_tau, flow_time)
    delta = velocity - u_buf
    # eq:q-gradient-mask uses a strict indicator, |delta_i| < alpha.
    # torch.clamp lets a gradient through at exactly +/- alpha, so the mask is applied by hand.
    # The forward value is still the clipped residual. CFM uses the raw residual.
    if clip:
        delta_clip = delta.clamp(-alpha, alpha)
        mask = (delta.abs() < alpha).to(dtype=delta.dtype)
        delta_for_q = delta_clip.detach() + (delta - delta.detach()) * mask
    else:
        delta_for_q = delta
    endpoint = replay_action + (1.0 - flow_time) * delta_for_q
    reconstruction = x_tau + (1.0 - flow_time) * u_buf
    cfm = delta.square().sum(dim=-1).mean()
    if use_q:
        q_values = q1(state, endpoint)
        loss = -q_values.mean() + lambda_cfm * cfm
        q_mean = q_values.mean().detach()
    else:
        with torch.no_grad():
            q_mean = q1(state, endpoint).mean()
        loss = lambda_cfm * cfm
    inside = delta.abs() < alpha
    allowed = (1.0 - flow_time) * alpha
    excess = (endpoint - replay_action).abs() - allowed
    return ActorTerms(
        loss=loss,
        cfm=cfm.detach(),
        q_mean=q_mean,
        mask_fraction=inside.float().mean().detach(),
        identity_error=(reconstruction - replay_action).abs().max().detach(),
        endpoint_gap=(endpoint - replay_action).abs().max().detach(),
        bound_violation=excess.max().clamp(min=0).detach(),
    )


def euler_integrate(actor: nn.Module, state: Tensor, x0: Tensor, n_steps: int) -> Tensor:
    """Explicit Euler from tau=0 to tau=1. Times are k/N for k = 0..N-1."""
    if n_steps < 1:
        raise ValueError(f"euler steps must be positive, got {n_steps}")
    step = 1.0 / n_steps
    sample = x0
    batch = state.shape[0]
    for k in range(n_steps):
        flow_time = torch.full((batch, 1), k * step, dtype=state.dtype)
        velocity = actor(state, sample, flow_time)
        sample = sample + step * velocity
    return sample


def critic_td_loss(
    critic: TwinQ,
    target_actor: nn.Module,
    target_critic: TwinQ,
    state: Tensor,
    action: Tensor,
    reward: Tensor,
    next_state: Tensor,
    done: Tensor,
    euler_noise: Tensor,
    smooth_noise: Tensor,
    cfg: SmokeConfig,
) -> Tensor:
    """Twin-Q TD3 target. Next action is a 5-step flow sample plus clipped noise.

    The action is not squashed with tanh and is not clamped to [-1, 1].
    The paper rejects endpoint tanh and does not state an action bound.
    """
    with torch.no_grad():
        next_action = euler_integrate(target_actor, next_state, euler_noise, cfg.euler_steps)
        clipped_noise = (smooth_noise * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
        next_action = next_action + clipped_noise
        next_q1, next_q2 = target_critic(next_state, next_action)
        target = reward + cfg.gamma * (1.0 - done) * torch.minimum(next_q1, next_q2)
    q1, q2 = critic(state, action)
    return (q1 - target).square().mean() + (q2 - target).square().mean()


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """target <- (1-tau) target + tau source. CleanRL writes the same update."""
    for src, dst in zip(source.parameters(), target.parameters(), strict=True):
        dst.mul_(1.0 - tau).add_(src, alpha=tau)


def flat_parameters(module: nn.Module) -> Tensor:
    return torch.cat([param.detach().reshape(-1) for param in module.parameters()])


def parameter_l2(left: Tensor, right: Tensor) -> float:
    return torch.linalg.vector_norm(left - right).item()


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def build_models(cfg: SmokeConfig) -> tuple[FlowActor, TwinQ, FlowActor, TwinQ]:
    torch.manual_seed(cfg.seed)
    actor = FlowActor(cfg.state_dim, cfg.action_dim, cfg.hidden, cfg.time_dim)
    critic = TwinQ(cfg.state_dim, cfg.action_dim, cfg.hidden)
    target_actor = copy.deepcopy(actor)
    target_critic = copy.deepcopy(critic)
    freeze_module(target_actor)
    freeze_module(target_critic)
    return actor, critic, target_actor, target_critic


def make_replay(cfg: SmokeConfig) -> dict[str, Tensor]:
    """Fixed buffer. Reward is -(a - s)^2. Behavior action is 0.25 * s. No environment."""
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    state = torch.rand(cfg.buffer_size, cfg.state_dim, generator=generator) * 2.0 - 1.0
    action = 0.25 * state
    reward = -((action - state).square().sum(dim=-1, keepdim=True))
    next_state = 0.5 * state
    done = torch.zeros(cfg.buffer_size, 1)
    steps = cfg.critic_updates
    batch = cfg.batch_size
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "index": torch.randint(0, cfg.buffer_size, (steps, batch), generator=generator),
        "cfm_noise": torch.randn(steps, batch, cfg.action_dim, generator=generator),
        "flow_time": torch.rand(steps, batch, 1, generator=generator),
        "euler_noise": torch.randn(steps, batch, cfg.action_dim, generator=generator),
        "smooth_noise": torch.randn(steps, batch, cfg.action_dim, generator=generator),
    }


def batch_at(data: dict[str, Tensor], step: int) -> dict[str, Tensor]:
    index = data["index"][step]
    return {
        "state": data["state"][index],
        "action": data["action"][index],
        "reward": data["reward"][index],
        "next_state": data["next_state"][index],
        "done": data["done"][index],
        "cfm_noise": data["cfm_noise"][step],
        "flow_time": data["flow_time"][step],
        "euler_noise": data["euler_noise"][step],
        "smooth_noise": data["smooth_noise"][step],
    }


def terms_to_dict(terms: ActorTerms) -> dict[str, float]:
    return {
        "actor_loss": terms.loss.item(),
        "cfm": terms.cfm.item(),
        "q_mean": terms.q_mean.item(),
        "mask_fraction": terms.mask_fraction.item(),
        "identity_error": terms.identity_error.item(),
        "endpoint_gap": terms.endpoint_gap.item(),
        "bound_violation": terms.bound_violation.item(),
    }


def probe(
    actor: nn.Module,
    critic: TwinQ,
    target_actor: nn.Module,
    target_critic: TwinQ,
    data: dict[str, Tensor],
    cfg: SmokeConfig,
    *,
    use_q: bool,
    clip: bool,
) -> dict[str, float]:
    """Evaluate step-0 tensors. Caller decides whether gradients are enabled."""
    sample = batch_at(data, 0)
    actor_terms = qf3_actor_terms(
        actor,
        critic.q1,
        sample["state"],
        sample["action"],
        sample["cfm_noise"],
        sample["flow_time"],
        alpha=cfg.alpha,
        lambda_cfm=cfg.lambda_cfm,
        use_q=use_q,
        clip=clip,
    )
    td = critic_td_loss(
        critic,
        target_actor,
        target_critic,
        sample["state"],
        sample["action"],
        sample["reward"],
        sample["next_state"],
        sample["done"],
        sample["euler_noise"],
        sample["smooth_noise"],
        cfg,
    )
    values = terms_to_dict(actor_terms)
    values["objective"] = values.pop("actor_loss")
    values["critic_loss"] = td.item()
    return values


def train_mode(
    actor: FlowActor,
    critic: TwinQ,
    target_actor: FlowActor,
    target_critic: TwinQ,
    data: dict[str, Tensor],
    cfg: SmokeConfig,
    mode: str,
) -> dict[str, float | int | None]:
    """Run critic updates. The actor updates every policy_frequency critic steps.

    mode qf3: clipped actor loss.
    mode direct_q: paper ablation with no velocity clip.
    mode cfm_only: CFM term only.
    mode no_step: qf3 loss, backward, and no optimizer step.
    """
    if mode not in {"qf3", "direct_q", "cfm_only", "no_step"}:
        raise ValueError(f"unknown mode {mode}")
    use_q = mode != "cfm_only"
    clip = mode != "direct_q"
    do_step = mode != "no_step"
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    first_actor_loss: float | None = None
    last_actor_loss: float | None = None
    actor_updates = 0
    for step in range(cfg.critic_updates):
        sample = batch_at(data, step)
        td = critic_td_loss(
            critic,
            target_actor,
            target_critic,
            sample["state"],
            sample["action"],
            sample["reward"],
            sample["next_state"],
            sample["done"],
            sample["euler_noise"],
            sample["smooth_noise"],
            cfg,
        )
        critic_opt.zero_grad(set_to_none=True)
        td.backward()
        if do_step:
            critic_opt.step()
        if step % cfg.policy_frequency == 0:
            terms = qf3_actor_terms(
                actor,
                critic.q1,
                sample["state"],
                sample["action"],
                sample["cfm_noise"],
                sample["flow_time"],
                alpha=cfg.alpha,
                lambda_cfm=cfg.lambda_cfm,
                use_q=use_q,
                clip=clip,
            )
            actor_opt.zero_grad(set_to_none=True)
            terms.loss.backward()
            if do_step:
                actor_opt.step()
                polyak_update(actor, target_actor, cfg.polyak_tau)
                polyak_update(critic, target_critic, cfg.polyak_tau)
            loss_value = terms.loss.item()
            if first_actor_loss is None:
                first_actor_loss = loss_value
            last_actor_loss = loss_value
            actor_updates += 1
    return {
        "actor_updates": actor_updates,
        "first_train_actor_loss": first_actor_loss,
        "last_train_actor_loss": last_actor_loss,
    }


def check_close(actual: Tensor, expected: Tensor, name: str, tol: float = 1e-5) -> float:
    error = (actual - expected).abs().max().item()
    if error > tol:
        raise AssertionError(f"{name} max abs error {error} exceeds {tol}: got {actual}, expected {expected}")
    return error


def run_gradient_checks(cfg: SmokeConfig) -> dict[str, float]:
    """Autograd of the real actor loss against the filtered-gradient formula."""
    state = torch.zeros(1, cfg.action_dim)
    action = torch.zeros(1, cfg.action_dim)
    noise = torch.zeros(1, cfg.action_dim)
    flow_time = torch.tensor([[0.5]])
    q_net = SumActionQ()
    cases = (
        ("inside_and_outside", torch.tensor([[0.0, 3.0]]), torch.tensor([[-0.5, 1.8]])),
        ("both_at_clip_edge", torch.tensor([[1.0, -1.0]]), torch.tensor([[0.6, -0.6]])),
        ("both_inside", torch.tensor([[0.25, -0.25]]), torch.tensor([[-0.35, -0.65]])),
    )
    max_error = 0.0
    for name, velocity, expected in cases:
        actor = FixedVelocity(velocity)
        terms = qf3_actor_terms(
            actor,
            q_net,
            state,
            action,
            noise,
            flow_time,
            alpha=cfg.alpha,
            lambda_cfm=cfg.lambda_cfm,
            use_q=True,
            clip=True,
        )
        terms.loss.backward()
        if actor.velocity.grad is None:
            raise AssertionError(f"{name} produced no velocity gradient")
        max_error = max(max_error, check_close(actor.velocity.grad, expected, name))
    # Unclipped direct Q must be allowed to leave the (1-tau)*alpha ball.
    direct = FixedVelocity(torch.tensor([[0.0, 3.0]]))
    direct_terms = qf3_actor_terms(
        direct,
        q_net,
        state,
        action,
        noise,
        flow_time,
        alpha=cfg.alpha,
        lambda_cfm=cfg.lambda_cfm,
        use_q=True,
        clip=False,
    )
    if direct_terms.endpoint_gap.item() <= (1.0 - flow_time.item()) * cfg.alpha:
        raise AssertionError("direct Q stayed inside the velocity clip; the ablation is not unclipped")
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    rand_noise = torch.randn(16, cfg.action_dim, generator=generator)
    rand_action = torch.randn(16, cfg.action_dim, generator=generator)
    rand_time = torch.rand(16, 1, generator=generator)
    x_tau, u_buf = flow_path(rand_noise, rand_action, rand_time)
    identity_error = (x_tau + (1.0 - rand_time) * u_buf - rand_action).abs().max().item()
    if identity_error > 1e-6:
        raise AssertionError(f"CFM path identity failed with error {identity_error}")
    return {"filtered_grad_max_error": max_error, "path_identity_max_error": identity_error}


def one_optimizer_step(
    actor: FlowActor,
    critic: TwinQ,
    data: dict[str, Tensor],
    cfg: SmokeConfig,
) -> dict[str, float]:
    """One Adam step on the QF3 actor loss with the critic held fixed."""
    sample = batch_at(data, 0)
    before = qf3_actor_terms(
        actor,
        critic.q1,
        sample["state"],
        sample["action"],
        sample["cfm_noise"],
        sample["flow_time"],
        alpha=cfg.alpha,
        lambda_cfm=cfg.lambda_cfm,
        use_q=True,
        clip=True,
    )
    before_value = before.loss.item()
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    opt.zero_grad(set_to_none=True)
    before.loss.backward()
    grad_sq = 0.0
    for param in actor.parameters():
        if param.grad is not None:
            grad_sq += param.grad.detach().square().sum().item()
    opt.step()
    with torch.no_grad():
        after = qf3_actor_terms(
            actor,
            critic.q1,
            sample["state"],
            sample["action"],
            sample["cfm_noise"],
            sample["flow_time"],
            alpha=cfg.alpha,
            lambda_cfm=cfg.lambda_cfm,
            use_q=True,
            clip=True,
        )
    after_value = after.loss.item()
    if not grad_sq > 0.0:
        raise AssertionError("actor gradient norm was zero on the one-step check")
    if not after_value < before_value:
        raise AssertionError(
            f"one Adam step did not reduce the fixed-critic actor loss: before={before_value} after={after_value}"
        )
    return {
        "actor_loss_before": before_value,
        "actor_loss_after": after_value,
        "actor_loss_delta": after_value - before_value,
        "grad_l2": math.sqrt(grad_sq),
    }


def copy_models(
    actor: FlowActor,
    critic: TwinQ,
    target_actor: FlowActor,
    target_critic: TwinQ,
) -> tuple[FlowActor, TwinQ, FlowActor, TwinQ]:
    copied_actor = copy.deepcopy(actor)
    copied_critic = copy.deepcopy(critic)
    copied_target_actor = copy.deepcopy(target_actor)
    copied_target_critic = copy.deepcopy(target_critic)
    freeze_module(copied_target_actor)
    freeze_module(copied_target_critic)
    return copied_actor, copied_critic, copied_target_actor, copied_target_critic


def assert_finite(values: dict[str, float], label: str) -> None:
    for key, value in values.items():
        if not math.isfinite(value):
            raise AssertionError(f"{label} {key} is not finite: {value}")


def run_smoke(cfg: SmokeConfig) -> dict[str, object]:
    checks = run_gradient_checks(cfg)
    actor, critic, target_actor, target_critic = build_models(cfg)
    last = actor.net[-1]
    if last.weight.abs().sum().item() != 0.0 or last.bias.abs().sum().item() != 0.0:
        raise AssertionError("flow actor last layer was not zero at init")
    data = make_replay(cfg)
    step_actor, step_critic, _, _ = copy_models(actor, critic, target_actor, target_critic)
    one_step = one_optimizer_step(step_actor, step_critic, data, cfg)

    runs: dict[str, object] = {}
    final_actor_vectors: dict[str, Tensor] = {}
    for mode in ("qf3", "cfm_only", "direct_q", "no_step"):
        mode_actor, mode_critic, mode_target_actor, mode_target_critic = copy_models(
            actor, critic, target_actor, target_critic
        )
        use_q = mode != "cfm_only"
        clip = mode != "direct_q"
        initial_q1 = copy.deepcopy(mode_critic.q1)
        with torch.no_grad():
            before = probe(
                mode_actor,
                mode_critic,
                mode_target_actor,
                mode_target_critic,
                data,
                cfg,
                use_q=use_q,
                clip=clip,
            )
        actor_before = flat_parameters(mode_actor)
        critic_before = flat_parameters(mode_critic)
        trace = train_mode(
            mode_actor,
            mode_critic,
            mode_target_actor,
            mode_target_critic,
            data,
            cfg,
            mode,
        )
        with torch.no_grad():
            after = probe(
                mode_actor,
                mode_critic,
                mode_target_actor,
                mode_target_critic,
                data,
                cfg,
                use_q=use_q,
                clip=clip,
            )
            sample = batch_at(data, 0)
            frozen_terms = qf3_actor_terms(
                mode_actor,
                initial_q1,
                sample["state"],
                sample["action"],
                sample["cfm_noise"],
                sample["flow_time"],
                alpha=cfg.alpha,
                lambda_cfm=cfg.lambda_cfm,
                use_q=use_q,
                clip=clip,
            )
        frozen_probe = terms_to_dict(frozen_terms)
        frozen_probe["objective"] = frozen_probe.pop("actor_loss")
        assert_finite(before, f"{mode} before")
        assert_finite(after, f"{mode} after")
        assert_finite(frozen_probe, f"{mode} frozen critic")
        if after["identity_error"] > 1e-5 or before["identity_error"] > 1e-5:
            raise AssertionError(f"{mode} broke the CFM path identity")
        if clip and (float(before["bound_violation"]) > 1e-5 or float(after["bound_violation"]) > 1e-5):
            raise AssertionError(f"{mode} left the velocity trust region: {before} {after}")
        runs[mode] = {
            "probe_before": before,
            "probe_after": after,
            "probe_after_initial_q1": frozen_probe,
            "actor_param_l2": parameter_l2(flat_parameters(mode_actor), actor_before),
            "critic_param_l2": parameter_l2(flat_parameters(mode_critic), critic_before),
            **trace,
        }
        final_actor_vectors[mode] = flat_parameters(mode_actor)

    qf3 = runs["qf3"]
    no_step = runs["no_step"]
    if not isinstance(qf3, dict) or not isinstance(no_step, dict):
        raise AssertionError("missing run records")
    if float(no_step["actor_param_l2"]) != 0.0 or float(no_step["critic_param_l2"]) != 0.0:
        raise AssertionError(
            f"no_step changed parameters: actor {no_step['actor_param_l2']} critic {no_step['critic_param_l2']}"
        )
    no_step_before = no_step["probe_before"]
    no_step_after = no_step["probe_after"]
    if not isinstance(no_step_before, dict) or not isinstance(no_step_after, dict):
        raise AssertionError("no_step probe missing")
    for key, value in no_step_before.items():
        if value != no_step_after[key]:
            raise AssertionError(f"no_step probe {key} changed: {value} -> {no_step_after[key]}")
    if float(qf3["actor_param_l2"]) <= 0.0:
        raise AssertionError("qf3 actor parameters did not move")
    qf3_before = qf3["probe_before"]
    qf3_after = qf3["probe_after"]
    qf3_frozen = qf3["probe_after_initial_q1"]
    if (
        not isinstance(qf3_before, dict)
        or not isinstance(qf3_after, dict)
        or not isinstance(qf3_frozen, dict)
    ):
        raise AssertionError("qf3 probe missing")
    if qf3_before["objective"] == qf3_after["objective"] and qf3_before["cfm"] == qf3_after["cfm"]:
        raise AssertionError("qf3 fixed-probe objective and CFM were bitwise unchanged")
    direct_q = runs["direct_q"]
    if not isinstance(direct_q, dict):
        raise AssertionError("direct_q run missing")
    direct_before = direct_q["probe_before"]
    if not isinstance(direct_before, dict):
        raise AssertionError("direct_q probe missing")
    if float(direct_before["bound_violation"]) <= 0.0:
        raise AssertionError(
            "direct_q probe stayed inside the clip, so the ablation did not leave the trust region"
        )
    if parameter_l2(final_actor_vectors["qf3"], final_actor_vectors["cfm_only"]) <= 0.0:
        raise AssertionError("qf3 and cfm_only ended at the same actor parameters")
    if parameter_l2(final_actor_vectors["qf3"], final_actor_vectors["direct_q"]) <= 0.0:
        raise AssertionError("qf3 and direct_q ended at the same actor parameters")
    if float(qf3_before["mask_fraction"]) >= 1.0:
        raise AssertionError("initial clip mask was inactive, so this run did not exercise filtering")

    return {
        "label": "dry_run",
        "seed": cfg.seed,
        "device": "cpu",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": asdict(cfg),
        "unit_checks": checks,
        "one_step_fixed_critic": one_step,
        "runs": runs,
        "controls": {
            "no_step_actor_param_l2": no_step["actor_param_l2"],
            "no_step_critic_param_l2": no_step["critic_param_l2"],
            "qf3_vs_cfm_only_actor_l2": parameter_l2(
                final_actor_vectors["qf3"], final_actor_vectors["cfm_only"]
            ),
            "qf3_vs_direct_q_actor_l2": parameter_l2(
                final_actor_vectors["qf3"], final_actor_vectors["direct_q"]
            ),
            "qf3_probe_objective_delta": qf3_after["objective"] - qf3_before["objective"],
            "qf3_frozen_q1_objective_delta": qf3_frozen["objective"] - qf3_before["objective"],
            "qf3_probe_cfm_delta": qf3_after["cfm"] - qf3_before["cfm"],
            "qf3_probe_critic_loss_delta": qf3_after["critic_loss"] - qf3_before["critic_loss"],
        },
        "deviations": [
            "Hidden width is 32. Table tab:qf3-arch uses 256 for MuJoCo.",
            "Replay is a fixed synthetic buffer. There is no environment and no new data collection.",
            "No tanh and no [-1, 1] action clamp. The paper rejects endpoint tanh and states no action bound.",
            "polyak_tau, policy_noise, noise_clip, and policy_frequency are CleanRL TD3 defaults. The paper does not restate them.",
            "The actor maximizes Q1 only. CleanRL TD3 does this. The paper writes a single Q. Robomimic uses the min over 10 heads.",
            "CFM is the batch mean of the squared L2 from eq:actor-cfm-loss, not the supplement 1/d_a form.",
            "No lambda_base term. That anchor is only in the fine-tuning experiments.",
            "gamma is 0.99. The MuJoCo section does not restate it.",
            "Hidden-layer initialization is the PyTorch default. The paper only specifies a zero last layer.",
            "Critic loss is the sum of two mean squared Bellman errors, as in CleanRL TD3.",
        ],
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = SmokeConfig()
    result = run_smoke(cfg)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    controls = result["controls"]
    one_step = result["one_step_fixed_critic"]
    qf3 = result["runs"]["qf3"]
    if not isinstance(controls, dict) or not isinstance(one_step, dict) or not isinstance(qf3, dict):
        raise AssertionError("result schema changed")
    before = qf3["probe_before"]
    after = qf3["probe_after"]
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise AssertionError("qf3 probe schema changed")
    print("dry_run seed=0 device=cpu")
    print(
        "one_step_fixed_critic "
        f"actor_loss_before={one_step['actor_loss_before']!r} "
        f"actor_loss_after={one_step['actor_loss_after']!r} "
        f"delta={one_step['actor_loss_delta']!r}"
    )
    print(
        "qf3_probe "
        f"objective {before['objective']!r} -> {after['objective']!r} "
        f"cfm {before['cfm']!r} -> {after['cfm']!r} "
        f"critic_loss {before['critic_loss']!r} -> {after['critic_loss']!r} "
        f"frozen_q1_objective_delta={controls['qf3_frozen_q1_objective_delta']!r}"
    )
    print(
        "controls "
        f"qf3_actor_param_l2={qf3['actor_param_l2']!r} "
        f"no_step_actor_param_l2={controls['no_step_actor_param_l2']!r} "
        f"qf3_vs_cfm_only_actor_l2={controls['qf3_vs_cfm_only_actor_l2']!r} "
        f"qf3_vs_direct_q_actor_l2={controls['qf3_vs_direct_q_actor_l2']!r}"
    )


if __name__ == "__main__":
    main()
