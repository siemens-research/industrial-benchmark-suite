"""Test functionality of real-cartpole benchmark.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

import numpy as np
import pytest
import tensorflow as tf
from gymnasium.envs.registration import make

from ibs.common.utils import resolve_path
from ibs.envs.real_cartpole.real_cartpole_env import DEFAULT_CONFIG_MLP, RealCartPoleEnv


def test_env_fixed_init_with_policy():
    """Test functionality of env when applying constant action, make sure return is below 0."""
    env = make(
        "RealCartPole-v0",
        init_method="fixed",
        max_episode_length=10,
    )

    PATH_TO_POLICY_MODEL = resolve_path(
        "../resources/envs/real_cartpole/policy_models/moose_realcartpole_upswing_linux_8/checkpoint_000020", __file__
    )
    POLICY_BUFFER_SIZE = 10
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    observations, actions, rewards, infos = [], [], [], []
    rewards_from_models = []
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
        rewards_from_models.append(info["mean_reward_from_models"])
        infos.append(info)
        model_input_buffer = info["model_input_buffer"]
        step_count += 1

    assert np.isclose(np.sum(rewards), -3.636, atol=0.001)


def test_env_from_data_init():
    """Test functionality of env when applying constant action, make sure return is below 0."""
    config = {**DEFAULT_CONFIG_MLP}
    config.update(
        {
            "max_episode_length": 10,
            "init_method": "from_data",
        }
    )
    env = RealCartPoleEnv(config)

    observations, actions, rewards, infos = [], [], [], []
    rewards_from_models = []
    terminated = truncated = False
    obs, _ = env.reset()
    step_count = 0
    while not (terminated or truncated):
        action = np.array([0.1], dtype=np.float32)  # for testing purpose only
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        rewards_from_models.append(info["mean_reward_from_models"])
        infos.append(info)
        step_count += 1

    assert step_count <= 10, "max_episode_length was not applied"
    assert np.sum(rewards) < 0.0


def test_step_limit_is_truncated_not_terminated():
    """Test that reaching the step limit (state within limits) is reported as truncated, not terminated."""
    config = {**DEFAULT_CONFIG_MLP}
    config.update({"max_episode_length": 5, "init_method": "fixed"})
    env = RealCartPoleEnv(config)
    env.reset()
    # keep the state within the observation space to isolate the step limit
    env.observation_space.contains = lambda _: True

    for step in range(1, 6):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        assert terminated is False
        assert truncated is (step == 5)


def test_leaving_observation_space_is_terminated_with_penalty():
    """Test that leaving the allowed state range is a failure: terminated (not truncated) with step penalty."""
    config = {**DEFAULT_CONFIG_MLP}
    config.update({"max_episode_length": 5, "init_method": "fixed"})
    env = RealCartPoleEnv(config)
    env.reset()
    env.observation_space.contains = lambda _: False

    _, reward, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False
    # the base reward is in [-0.69, 0], the penalty for the 4 missed steps is -4
    assert -4.69 <= reward <= -4.0


def test_max_episode_steps_config_key_is_rejected():
    """Test that the gym-style key 'max_episode_steps' is rejected instead of being silently ignored."""
    config = {**DEFAULT_CONFIG_MLP, "max_episode_steps": 10}
    with pytest.raises(ValueError, match="max_episode_length"):
        RealCartPoleEnv(config)
