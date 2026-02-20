"""Test functionality of real-cartpole benchmark."""

import numpy as np

from ibs.envs.four_tank.four_tank_env import DEFAULT_CONFIG, FourTankEnv

MAX_EPISODE_STEPS = 10


def test_env_random_action():
    """Test functionality of env when applying random action, make sure return is below 0."""
    config = {**DEFAULT_CONFIG}
    config.update(
        {
            "max_episode_steps": MAX_EPISODE_STEPS,
            "init_method": "fixed",
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
