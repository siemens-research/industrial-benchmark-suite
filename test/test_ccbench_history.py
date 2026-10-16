"""Tests for the preallocated episode history of the CCBench environment."""

import numpy as np
import pandas as pd
import pytest

from ibs.envs.ccbench.ccbench_env import create_ccbench_env
from ibs.envs.ccbench.history import EpisodeHistory

_START = pd.Timestamp("2000-01-01T00:00:00Z")
_FREQ = pd.Timedelta(seconds=1)


def test_views_contain_rows_written_per_channel():
    history = EpisodeHistory(start=_START, freq=_FREQ, capacity=4, channels=["a", "b"])
    assert len(history["a"]) == 0
    history.append("a", _START, 1.0)
    history.append("b", _START, 2.0)
    history.append("a", _START + _FREQ, 3.0)
    pd.testing.assert_series_equal(
        history["a"], pd.Series([1.0, 3.0], index=pd.DatetimeIndex([_START, _START + _FREQ]), name="a")
    )
    assert history["b"].tolist() == [2.0]
    assert history["b"].index[-1] == _START
    assert set(history) == {"a", "b"}


def test_capacity_grows():
    history = EpisodeHistory(start=_START, freq=_FREQ, capacity=2, channels=["a"])
    for i in range(5):
        history.append("a", _START + i * _FREQ, float(i))
    assert history.capacity >= 5
    assert history["a"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert history["a"].index[-1] == _START + 4 * _FREQ


def test_unexpected_timestamp_raises():
    history = EpisodeHistory(start=_START, freq=_FREQ, capacity=2, channels=["a"])
    with pytest.raises(ValueError):
        history.append("a", _START + _FREQ, 1.0)


def test_episode_longer_than_preallocated_history():
    env = create_ccbench_env(episode_length=3, warmup_steps=0)
    env.reset(seed=0)
    capacity = env.history.capacity
    action = {"BP_action": 0.0, "DS_action": 0.0, "HE_action": 0.0}
    for _ in range(capacity + 5):
        env.step(action)
    data = env.get_episode_data()
    assert len(data) == capacity + 6
    assert np.all(np.isfinite(data["AT"].to_numpy()))
