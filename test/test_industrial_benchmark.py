"""Test functionality of the industrial benchmark (core IDS + Gymnasium wrapper)."""

import numpy as np

from ibs.common.utils import resolve_path
from ibs.envs.industrial_benchmark.industrial_benchmark_env import IBGym, IDS

REFERENCE_DATA_DIR = resolve_path("../resources/envs/industrial_benchmark/test_data", __file__)


def _all_states(env):
    s = env.state
    return [
        s.operational_cost_buffer[0], s.operational_cost_buffer[1],
        s.operational_cost_buffer[2], s.operational_cost_buffer[3],
        s.operational_cost_buffer[4], s.operational_cost_buffer[5],
        s.operational_cost_buffer[6], s.operational_cost_buffer[7],
        s.operational_cost_buffer[8], s.operational_cost_buffer[9],
        s.current_operational_cost, s.hidden_gain, s.hidden_velocity,
        s.hidden_shift, s.gs_domain, s.gs_sys_response, s.gs_phi_idx,
        s.effective_gain, s.effective_velocity, s.miscalibration,
        s.consumption, s.setpoint, s.velocity, s.gain, s.shift,
        s.fatigue, s.fatigue_base, s.operational_cost_conv, s.reward,
    ]


def test_trajectories_match_reference_data():
    """Verify that IDS trajectories exactly match pre-recorded reference CSVs."""
    np.random.seed(501)
    setpoints = [np.random.randint(1, 100) for _ in range(10)]
    T = 1000

    for i in range(10):
        env = IDS(setpoints[i], initial_seed=1005 + i)

        markov_states_all = None
        for j in range(T):
            action = 2 * np.random.rand(3) - 1
            env.step(action)
            row = np.array(_all_states(env))
            if j == 0:
                markov_states_all = row
            else:
                markov_states_all = np.vstack([markov_states_all, row])

        reference = np.genfromtxt(REFERENCE_DATA_DIR / f"markovStates{i}.csv", delimiter=",")
        np.testing.assert_array_almost_equal(reference, markov_states_all)


def test_gym_wrapper_runs_episodes():
    """Run several episodes and verify finite returns."""
    env = IBGym(70)
    obs, info = env.reset()

    assert obs.shape == (6,)
    assert isinstance(info, dict)

    discount = 0.97
    returns = []
    for _ in range(10):
        acc_return = 0.0
        for i in range(100):
            state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            acc_return += reward * discount**i
        returns.append(acc_return / 100.0)

    assert len(returns) == 10
    assert all(np.isfinite(r) for r in returns)


def test_discrete_action_space():
    """Verify discrete action mode works."""
    env = IBGym(50, action_type="discrete")
    obs, info = env.reset()

    assert obs.shape == (6,)
    state, reward, terminated, truncated, info = env.step(0)
    assert np.isfinite(reward)


def test_include_past_observation():
    """Verify include_past observation stacking."""
    n_past = 5
    env = IBGym(50, observation_type="include_past", n_past_timesteps=n_past)
    obs, info = env.reset()

    assert obs.shape == (6 * n_past,)
    state, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert state.shape == (6 * n_past,)
