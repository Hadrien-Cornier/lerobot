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

"""Compatibility imports for recipe definitions shared with inference."""

from lerobot.configs.recipe import (
    DEFAULT_BINDINGS,
    PLACEHOLDER_RE,
    MessageRole,
    MessageStream,
    MessageTurn,
    RecipeRoute,
    TrainingRecipe,
    load_recipe,
    render_message_turns,
    resolve_recipe_override,
)

__all__ = [
    "DEFAULT_BINDINGS",
    "PLACEHOLDER_RE",
    "MessageRole",
    "MessageStream",
    "MessageTurn",
    "RecipeRoute",
    "TrainingRecipe",
    "load_recipe",
    "render_message_turns",
    "resolve_recipe_override",
]
