"""Test functionality of four-tank benchmark."""

import numpy as np
import tensorflow as tf
from gymnasium.envs.registration import make

from ibs.common.utils import resolve_path
from ibs.envs.four_tank.four_tank_env import DEFAULT_CONFIG, FourTankEnv


def test_env_fixed_init_with_policy():
    """Test functionality of env using make() and a pre-trained policy with fixed initialisation."""
    env = make(
        "FourTank-v0",
        init_method="fixed",
        max_episode_length=10,
    )
    PATH_TO_POLICY_MODEL = resolve_path("../resources/envs/four_tank/policy_models/run_05/checkpoint_000020", __file__)
    POLICY_BUFFER_SIZE = 12
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    observations, actions, rewards, infos = [], [], [], []
    terminated = False
    obs, info = env.reset()
    model_input_buffer = info["model_input_buffer"]
    step_count = 0
    while not terminated:
        policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
        action = policy_model([policy_input])[0]
        obs, reward, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        model_input_buffer = info["model_input_buffer"]
        step_count += 1

    assert np.isclose(np.sum(rewards), -31.4127, atol=0.001)


def test_env_from_data_init():
    """Test functionality of env when applying random action, make sure return is below 0."""
    config = {**DEFAULT_CONFIG}
    config.update(
        {
            "max_episode_steps": 10,
            "init_method": "from_data",
        }
    )
    env = FourTankEnv(config)

    observations, actions, rewards, infos = [], [], [], []
    terminated = False
    obs, _ = env.reset()
    step_count = 0
    while not terminated:
        action = env.action_space.sample()  # for testing purpose only
        obs, reward, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        step_count += 1

    assert np.sum(rewards) < 0.0


def test_reward_model_vs_computed_deviation():
    """Test that the reward from learned models deviates no more than 1% from the computed reward."""
    PATH_TO_POLICY_MODEL = resolve_path("../resources/envs/four_tank/policy_models/run_05/checkpoint_000020", __file__)
    POLICY_BUFFER_SIZE = 12

    env = make(
        "FourTank-v0",
        init_method="fixed",
        max_episode_length=10,
    )
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    rewards_computed, rewards_from_models = [], []
    terminated = False
    obs, info = env.reset()
    model_input_buffer = info["model_input_buffer"]
    while not terminated:
        policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
        action = policy_model([policy_input])[0]
        obs, reward, terminated, _, info = env.step(action)
        rewards_computed.append(reward)
        rewards_from_models.append(info["reward_from_models"])
        model_input_buffer = info["model_input_buffer"]

    total_computed = np.sum(rewards_computed)
    total_from_models = np.sum(rewards_from_models)

    # Relative deviation should not exceed 1%
    relative_deviation = np.abs(total_computed - total_from_models) / np.abs(total_computed)
    assert relative_deviation < 0.01, (
        f"Relative deviation {relative_deviation:.4f} exceeds 1% "
        f"(computed={total_computed:.4f}, model={total_from_models:.4f})"
    )
