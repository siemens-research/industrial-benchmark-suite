"""Tests for the data-only CCBench generators, the CCBench-v1 config and the code string gate of the factories."""

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from scipy import signal

import ibs.envs  # noqa: F401  # pylint: disable=unused-import
from ibs.common.config import load_config
from ibs.envs.ccbench.ccbench_env import DEFAULT_CONFIG_FILE
from ibs.envs.ccbench.ccbench_env import LEGACY_V0_CONFIG_FILE
from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.ccbench_env import create_ccbench_env
from ibs.envs.ccbench.ccbench_env import create_ccbench_env_v0
from ibs.envs.ccbench.config_utils import code_strings
from ibs.envs.ccbench.exploration_agent import ExplorationAgent
from ibs.envs.ccbench.functions import norm_filter
from ibs.envs.ccbench.functions import window_filter
from ibs.envs.ccbench.generators import apply_filter
from ibs.envs.ccbench.generators import function_callback
from ibs.envs.ccbench.generators import lagged_terms
from ibs.envs.ccbench.generators import modulated_sine
from ibs.envs.ccbench.history import EpisodeHistory

_START = pd.Timestamp("2000-01-01T00:00:00Z")
_FREQ = pd.Timedelta(seconds=1)


def _ts(step: int) -> pd.Timestamp:
    return _START + step * _FREQ


def _history(**channels) -> EpisodeHistory:
    history = EpisodeHistory(_START, _FREQ, 16, channels)
    for name, values in channels.items():
        for step, value in enumerate(values):
            history.append(name, _ts(step), value)
    return history


def _terms(*factors, coef=1.0):
    return lagged_terms(channel_name="y", spec={"lt_terms": [{"coef": coef, "factors": list(factors)}]})


def test_lagged_terms_resolves_lags_by_timestamp():
    # "a" has already been computed for step 4, "b" only up to step 3 (computed after the target)
    history = _history(a=[0.0, 1.0, 2.0, 3.0, 4.0], b=[10.0, 11.0, 12.0, 13.0])
    now = _ts(4)
    assert _terms({"input": "a"}).step(now, history) == 4.0
    assert _terms({"input": "a", "lag": 1}).step(now, history) == 3.0
    assert _terms({"input": "b", "lag": 1}).step(now, history) == 13.0
    assert _terms({"input": "b", "lag": 4}).step(now, history) == 10.0
    assert np.isnan(_terms({"input": "a", "lag": 5}).step(now, history))
    assert np.isnan(_terms({"input": "b", "lag": 5}).step(now, history))
    with pytest.raises(ValueError, match="lag 0"):
        _terms({"input": "b"}).step(now, history)


def test_lagged_terms_arithmetic():
    history = _history(a=[1.0, 2.0, 4.0], b=[3.0, 5.0, 10.0])
    now = _ts(2)
    gen = lagged_terms(
        channel_name="y",
        spec={
            "lt_terms": [
                {"coef": 2.0, "factors": [{"input": "a", "power": 2}, {"input": "b", "lag": 1, "power": -1}]},
                {"coef": 3.0, "factors": [{"input": "b", "offset": 1.0, "scale": -0.5}]},
                {"coef": 0.5},
            ]
        },
    )
    assert gen.step(now, history) == pytest.approx(2.0 * 4.0**2 / 5.0 + 3.0 * (1.0 - 0.5 * 10.0) + 0.5)
    assert gen.step(now, history, cur_val=1.0) == pytest.approx(1.0 + 6.4 - 12.0 + 0.5)
    assert _terms({"input": "a", "power": -2}).step(now, history) == 1.0 / 16.0


def test_lagged_terms_matches_legacy_formula_exactly():
    rng = np.random.default_rng(0)
    bp, ds, at, wave = (rng.uniform(1.0, 30.0, 10) for _ in range(4))
    history = _history(BP=bp, DS=ds, AT=at, wave=wave)
    now = _ts(9)
    od = lagged_terms(
        channel_name="OD",
        spec={
            "lt_terms": [
                {"coef": 1.0, "factors": [{"input": "wave"}]},
                {"coef": 1.5, "factors": [{"input": "BP", "lag": 3}, {"input": "DS", "lag": 1, "power": -1}]},
                {"coef": 0.1, "factors": [{"input": "AT"}]},
            ]
        },
    )
    wt = _terms(
        {"input": "BP", "lag": 5, "power": -1},
        {"input": "DS", "lag": 2, "power": -1},
        {"input": "AT", "offset": 1.0, "scale": -0.02, "power": -1},
        coef=15.0,
    )
    h = {k: history[k] for k in history}
    expected_od = h["wave"].iloc[-1] + 1.5 * h["BP"].iloc[-4] / h["DS"].iloc[-2] + 0.1 * h["AT"].iloc[-1]
    assert od.step(now, history) == expected_od
    assert wt.step(now, history) == 15.0 / h["BP"].iloc[-6] / h["DS"].iloc[-3] / (1.0 - 0.02 * h["AT"].iloc[-1])


@pytest.mark.parametrize(
    "speed, transport, expected_lag",
    [
        (2.0, {}, 3),  # 2 + 2 + 2 >= 5
        (5.0, {}, 1),
        (5.0, {"min_lag": 2}, 2),
        (0.1, {}, 4),  # clipped to max_lag
        (1.0, {"speed_scale": 2.5}, 2),
    ],
)
def test_transport_lag_depends_on_speed(speed, transport, expected_lag):
    x = [float(i) for i in range(8)]
    history = _history(v=[speed] * 8, x=x)
    gen = _terms({"input": "x", "transport": {"speed": "v", "distance": 5.0, "max_lag": 4, **transport}})
    assert gen.step(_ts(7), history) == x[7 - expected_lag]


def test_transport_lag_varies_within_episode():
    speeds = [1.0, 1.0, 1.0, 1.0, 1.0, 4.0]
    history = _history(v=speeds, x=[float(i) for i in range(6)])
    gen = _terms({"input": "x", "transport": {"speed": "v", "distance": 5.0, "max_lag": 5}})
    # 4 >= 5 fails, 4 + 1 >= 5 holds: lag 2 at the high speed of the last step
    assert gen.step(_ts(5), history) == 3.0
    short = _history(v=speeds[:3], x=[0.0, 1.0, 2.0])
    assert np.isnan(gen.step(_ts(2), short))


def test_modulated_sine_matches_legacy_lambda():
    def legacy(i, t):
        return 0.2 * np.exp(t.iloc[-1] / 5 - 1.0) * np.sin((i[-1] - i[0]).total_seconds())

    spec = {"ms_input": "HE", "ms_amplitude": 0.2, "ms_exp_scale": 0.2, "ms_exp_offset": -1.0}
    new = modulated_sine(channel_name="wave", spec=spec)
    old = function_callback(channel_name="wave", spec={"fc_inputs": ["index", "HE"], "fc_callback": legacy})
    rng = np.random.default_rng(1)
    history = EpisodeHistory(_START, _FREQ, 64, ["HE"])
    for step in range(50):
        history.append("HE", _ts(step), rng.uniform(0.0, 15.0))
        np.testing.assert_allclose(new.step(_ts(step), history), old.step(_ts(step), history), rtol=1e-12)


def test_af_window_equals_norm_filter_of_scipy_window():
    gen = apply_filter(channel_name="AT", spec={"af_window": {"name": "hann", "length": 300}})
    np.testing.assert_array_equal(gen.kernel, norm_filter(signal.windows.hann(300)))
    np.testing.assert_array_equal(window_filter("boxcar", 4, normalize=False), np.ones(4))
    with pytest.raises(ValueError, match="unknown window"):
        window_filter("kaiser", 10)


_V1_TYPES = {"wave": "modulated_sine", "OD": "lagged_terms", "WT": "lagged_terms"}


def _fixed_action(step: int) -> dict:
    return {
        "BP_action": 1.5 * np.sin(0.25 * step),
        "DS_action": 0.8 * np.cos(0.15 * step),
        "HE_action": 0.04 * np.sin(0.1 * step),
    }


def _run(config_file, seed: int, steps: int = 200):
    env = create_ccbench_env(config_file=config_file, episode_length=steps + 1)
    env.reset(seed=seed)
    rewards = [env.step(_fixed_action(step))[1] for step in range(1, steps + 1)]
    return env, env.get_episode_data(include_temporary=True, include_warmup=True), np.array(rewards)


def test_v1_config_has_no_code_strings_and_data_only_generators():
    assert code_strings(load_config(config_file=DEFAULT_CONFIG_FILE)) == []
    assert len(code_strings(load_config(config_file=LEGACY_V0_CONFIG_FILE))) == 5
    env = create_ccbench_env()
    for channel, typ in _V1_TYPES.items():
        assert typ in env.process_spec[channel]["type"]


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_v1_config_matches_legacy_string_formulas(seed: int):
    env_v1, data_v1, rewards_v1 = _run(DEFAULT_CONFIG_FILE, seed)
    env_legacy, data_legacy, rewards_legacy = _run(LEGACY_V0_CONFIG_FILE, seed)
    assert env_v1._warmup_rows == env_legacy._warmup_rows == 5  # pylint: disable=protected-access
    assert list(data_v1.columns) == list(data_legacy.columns)
    for channel in ("BP", "DS", "HE", "AT", "WT", "BP_action", "DS_action", "HE_action"):
        np.testing.assert_array_equal(data_v1[channel].to_numpy(), data_legacy[channel].to_numpy(), err_msg=channel)
    np.testing.assert_allclose(data_v1["wave"].to_numpy(), data_legacy["wave"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(data_v1["OD"].to_numpy(), data_legacy["OD"].to_numpy(), atol=1e-5)
    np.testing.assert_allclose(rewards_v1, rewards_legacy, atol=1e-5)


def test_v1_and_legacy_exploration_agents_generate_equal_actions():
    actions = {}
    for config_file in (DEFAULT_CONFIG_FILE, LEGACY_V0_CONFIG_FILE):
        env = create_ccbench_env(config_file=config_file)
        agent = ExplorationAgent(load_config(config_file=config_file)["action_spec"])
        agent.reset(seed=5)
        obs, _ = env.reset(seed=5)
        trajectory = []
        for _ in range(100):
            action = agent.act(obs["observation"], obs["desired_goal"], timestep=env.cur_timestep)
            trajectory.append([action[k] for k in sorted(action)])
            obs = env.step(action)[0]
        actions[config_file] = np.array(trajectory, dtype=float)
    np.testing.assert_array_equal(actions[DEFAULT_CONFIG_FILE], actions[LEGACY_V0_CONFIG_FILE])


def test_v1_runs_with_transport_delay():
    od_terms = [
        {"coef": 1.0, "factors": [{"input": "wave"}]},
        {
            "coef": 1.5,
            "factors": [
                {
                    "input": "BP",
                    "transport": {"speed": "DS", "distance": 0.25, "speed_scale": 0.0166667, "max_lag": 20},
                },
                {"input": "DS", "lag": 1, "power": -1},
            ],
        },
        {"coef": 0.1, "factors": [{"input": "AT"}]},
    ]
    env = create_ccbench_env(config_dict={"process_spec": {"OD": {"lt_terms": od_terms}}})
    obs, _ = env.reset(seed=2)
    for step in range(1, 101):
        obs = env.step(_fixed_action(step))[0]
        assert all(np.isfinite(v) for v in obs["observation"].values())


_CODE_OVERRIDE = {"process_spec": {"AT": {"af_kernel": "norm_filter(signal.windows.hann(100))", "af_window": None}}}


def test_factories_reject_code_strings_unless_trusted():
    with pytest.raises(ValueError, match="config_dict: process_spec.AT.af_kernel"):
        create_ccbench_env(config_dict=_CODE_OVERRIDE)
    with pytest.raises(ValueError, match="overrides: process_spec.AT.af_kernel"):
        create_ccbench_env(**_CODE_OVERRIDE)
    with pytest.raises(ValueError, match="Code strings"):
        create_ccbench_env_v0(config_dict={"process_spec": {"OD": {"fc_callback": "lambda *a: 0.0"}}})
    with pytest.raises(ValueError, match="Code strings"):
        gym.make("CCBench-v1", config_dict=_CODE_OVERRIDE)
    env = create_ccbench_env(config_dict=_CODE_OVERRIDE, trusted_code=True)
    assert len(env.process_spec["AT"]["af_kernel"]) == 100


def test_code_strings_from_config_files_and_direct_construction_are_accepted():
    assert "function_callback" in create_ccbench_env(config_file=LEGACY_V0_CONFIG_FILE).process_spec["OD"]["type"]
    assert "function_callback" in create_ccbench_env_v0().process_spec["OD"]["type"]
    env = CCBenchEnv(config_file=LEGACY_V0_CONFIG_FILE)
    assert callable(env.process_spec["OD"]["fc_callback"])


# CCBench-v0 trajectory recorded before the data-only generators were added (seed 3, `_fixed_action`)
_V0_CHANNELS = ["AT", "BP", "DS", "HE", "OD", "WT", "wave"]
_V0_OBSERVATIONS = {
    0: [6.8814448102659975, 12.0, 2.685193337148995, 4.368105065960997, np.nan, np.nan, 0.0],
    8: [
        6.8809131870852855,
        21.13455333352382,
        7.391687676218506,
        4.50366838942293,
        4.670308099846216,
        0.27474464084364736,
        0.1791730793921787,
    ],
    40: [
        6.854372310317187,
        22.56888240117322,
        1.1818406647666189,
        5.013875157969027,
        15.0,
        1.0595845004541202,
        0.14943674893603276,
    ],
}
_V0_REWARDS = {8: -3.2099472593101366, 40: -7.904584500454121}


def test_gym_make_v0_reproduces_legacy_trajectory():
    env = gym.make("CCBench-v0").unwrapped
    obs, _ = env.reset(seed=3)
    observations = {0: obs["observation"]}
    rewards = {}
    for step in range(1, 41):
        obs, reward = env.step(_fixed_action(step))[:2]
        observations[step] = obs["observation"]
        rewards[step] = reward
    for step, expected in _V0_OBSERVATIONS.items():
        assert sorted(observations[step]) == _V0_CHANNELS
        actual = [observations[step][c] for c in _V0_CHANNELS]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, err_msg=f"step {step}")
    for step, expected in _V0_REWARDS.items():
        np.testing.assert_allclose(rewards[step], expected, rtol=1e-12)
