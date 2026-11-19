"""Test functionality of distillation benchmark.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

import numpy as np
import pytest
import tensorflow as tf
from gymnasium.envs.registration import make

from ibs.common.utils import resolve_path
from ibs.envs.distillation.distillation_env import DEFAULT_CONFIG, DistillationEnv

MAX_EPISODE_STEPS = 10

PATH_TO_POLICY_MODEL = resolve_path("../resources/envs/distillation/policy_models/run_33/checkpoint_000020", __file__)
POLICY_BUFFER_SIZE = 20


def test_env_fixed_init_with_policy():
    """Test functionality of env using make() and a pre-trained policy with fixed initialisation."""
    env = make(
        "Distillation-v0",
        init_method="fixed",
        max_episode_length=MAX_EPISODE_STEPS,
    )
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    observations, actions, rewards, infos = [], [], [], []
    terminated = truncated = False
    obs, info = env.reset()
    model_input_buffer = info["model_input_buffer"]
    step_count = 0
    while not (terminated or truncated):
        policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
        action = policy_model([policy_input])[0]
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        model_input_buffer = info["model_input_buffer"]
        step_count += 1

    print(np.sum(rewards))

    assert np.isclose(np.sum(rewards), -1.3220, atol=0.001)


def test_env_from_data_init():
    """Test functionality of env when applying random action, make sure return is below 0."""
    config = {**DEFAULT_CONFIG}
    config.update(
        {
            "max_episode_length": MAX_EPISODE_STEPS,
            "init_method": "from_data",
        }
    )
    env = DistillationEnv(config)

    observations, actions, rewards, infos = [], [], [], []
    terminated = truncated = False
    obs, _ = env.reset()
    step_count = 0
    while not (terminated or truncated):
        action = env.action_space.sample()  # for testing purpose only
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        step_count += 1

    assert step_count <= MAX_EPISODE_STEPS, "max_episode_length was not applied"
    assert np.sum(rewards) < 0.0


def test_reward_model_vs_computed_deviation():
    """Test that the reward from learned models deviates no more than 5% from the computed reward."""
    env = make(
        "Distillation-v0",
        init_method="fixed",
        max_episode_length=MAX_EPISODE_STEPS,
    )
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    rewards_computed, rewards_from_models = [], []
    terminated = truncated = False
    obs, info = env.reset()
    model_input_buffer = info["model_input_buffer"]
    while not (terminated or truncated):
        policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
        action = policy_model([policy_input])[0]
        obs, reward, terminated, truncated, info = env.step(action)
        rewards_computed.append(reward)
        rewards_from_models.append(info["reward_from_models"])
        model_input_buffer = info["model_input_buffer"]

    total_computed = np.sum(rewards_computed)
    total_from_models = np.sum(rewards_from_models)

    # Relative deviation should not exceed 5%
    relative_deviation = np.abs(total_computed - total_from_models) / np.abs(total_computed)
    assert relative_deviation < 0.01, (
        f"Relative deviation {relative_deviation:.4f} exceeds 5% "
        f"(computed={total_computed:.4f}, model={total_from_models:.4f})"
    )


def test_episode_end_is_truncated_not_terminated():
    """Test that reaching the step limit is reported as truncated (there is no terminal state)."""
    config = {**DEFAULT_CONFIG}
    config.update({"max_episode_length": 5, "init_method": "fixed"})
    env = DistillationEnv(config)
    env.reset()

    for step in range(1, 6):
        _, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert terminated is False
        if truncated:
            break
    assert truncated
    assert step <= 5


def test_max_episode_steps_config_key_is_rejected():
    """Test that the gym-style key 'max_episode_steps' is rejected instead of being silently ignored."""
    config = {**DEFAULT_CONFIG, "max_episode_steps": 10}
    with pytest.raises(ValueError, match="max_episode_length"):
        DistillationEnv(config)
