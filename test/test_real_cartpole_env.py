"""Test functionality of real-cartpole benchmark."""

import numpy as np

from ibs.envs.real_cartpole.real_cartpole_env import CONFIG_MLP, RealCartPoleEnv

MAX_EPISODE_STEPS = 10


def test_env_fixed_init():
    """Test functionality of env when applying constant action, make sure return is below 0."""
    config = {**CONFIG_MLP}
    config.update(
        {
            "max_episode_steps": MAX_EPISODE_STEPS,
            "init_method": "fixed",
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


def test_env_from_data_init():
    """Test functionality of env when applying constant action, make sure return is below 0."""
    config = {**CONFIG_MLP}
    config.update(
        {
            "max_episode_steps": MAX_EPISODE_STEPS,
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
