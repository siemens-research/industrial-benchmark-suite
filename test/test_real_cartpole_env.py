"""Test functionality of real-cartpole benchmark."""

import numpy as np
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
            "max_episode_steps": 10,
            "init_method": "from_data",
        }
    )
    env = RealCartPoleEnv(config)

    observations, actions, rewards, infos = [], [], [], []
    rewards_from_models = []
    terminated = False
    obs, _ = env.reset()
    step_count = 0
    while not terminated:
        action = np.array([0.1], dtype=np.float32)  # for testing purpose only
        obs, reward, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        rewards_from_models.append(info["mean_reward_from_models"])
        infos.append(info)
        step_count += 1

    assert np.sum(rewards) < 0.0
