# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE
from lerobot.utils.random_utils import seeded_context


def test_unused_environment_state_preserves_actions_and_bet_loss() -> None:
    config = VQBeTConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        pretrained_backbone_weights=None,
        n_obs_steps=2,
        n_action_pred_token=2,
        action_chunk_size=2,
        n_vqvae_training_steps=1,
        vqvae_n_embed=4,
        vqvae_embedding_dim=16,
        vqvae_enc_hidden_dim=16,
        gpt_input_dim=32,
        gpt_output_dim=32,
        gpt_hidden_dim=32,
        gpt_n_layer=1,
        gpt_n_head=2,
    )
    reference = VQBeTPolicy(config)
    batch = {
        OBS_STATE: torch.randn(1, config.n_obs_steps, 2),
        OBS_IMAGE: torch.rand(1, config.n_obs_steps, 3, 96, 96),
        ACTION: torch.randn(1, len(config.action_delta_indices), 2),
    }
    reference(batch)  # Complete the codebook stage so the next forward trains BeT.
    extended_config = deepcopy(config)
    extended_config.input_features[OBS_ENV_STATE] = PolicyFeature(FeatureType.ENV, (3,))
    extended = VQBeTPolicy(extended_config)
    extended.load_state_dict(reference.state_dict())
    reference.eval()
    extended.eval()
    extended_batch = {**batch, OBS_ENV_STATE: torch.randn(1, config.n_obs_steps, 3)}
    with seeded_context(42):
        expected_loss, _ = reference(batch)
    with seeded_context(42):
        actual_loss, _ = extended(extended_batch)
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    assert extended.vqbet.action_token.grad is not None
    assert torch.isfinite(extended.vqbet.action_token.grad).all()
    observation = {key: value[:, -1] for key, value in batch.items() if key != ACTION}
    extended_observation = {**observation, OBS_ENV_STATE: extended_batch[OBS_ENV_STATE][:, -1]}
    with seeded_context(42):
        expected_action = reference.select_action(dict(observation))
    with seeded_context(42):
        actual_action = extended.select_action(extended_observation)
    torch.testing.assert_close(actual_action, expected_action)
