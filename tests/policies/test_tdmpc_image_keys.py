# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from copy import deepcopy

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.utils.random_utils import seeded_context


@pytest.mark.parametrize("use_mpc", [False, True])
def test_named_camera_matches_canonical_camera(use_mpc: bool) -> None:
    config = TDMPCConfig(
        device="cpu",
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
            OBS_IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 96, 96)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        use_mpc=use_mpc,
        horizon=3,
        state_encoder_hidden_dim=16,
        image_encoder_hidden_dim=4,
        mlp_dim=32,
        latent_dim=8,
        cem_iterations=1,
        n_gaussian_samples=8,
        n_pi_samples=2,
        n_elites=2,
    )
    reference = TDMPCPolicy(config).eval()
    named_config = deepcopy(config)
    camera = "observation.images.front"
    named_config.input_features[camera] = named_config.input_features.pop(OBS_IMAGE)
    named = TDMPCPolicy(named_config).eval()
    named.load_state_dict(reference.state_dict())
    observation = {OBS_STATE: torch.randn(1, 2), OBS_IMAGE: torch.rand(1, 3, 96, 96)}
    with seeded_context(42):
        expected = reference.select_action(dict(observation))
    with seeded_context(42):
        actual = named.select_action({OBS_STATE: observation[OBS_STATE], camera: observation[OBS_IMAGE]})
    assert actual.shape == (1, 2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
