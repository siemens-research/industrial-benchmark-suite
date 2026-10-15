"""Tests for the CCBench channel generators."""

import numpy as np
import pandas as pd

from ibs.envs.ccbench import generators

_TIMESTEP = pd.Timestamp("2000-01-01T00:00:00Z")


def _draw(gen: generators.ChannelGenerator, steps: int = 50) -> np.ndarray:
    return np.array([gen.step(_TIMESTEP, {}, None) for _ in range(steps)])


def _make_generators(rng: np.random.Generator) -> list:
    return [
        generators.spike_action_gen(channel_name="a", spec={"sag_dist": "normal"}, rng=rng),
        generators.spike_action_gen(channel_name="b", spec={"sag_dist": "uniform"}, rng=rng),
        generators.random_walk(channel_name="c", spec={"rw_dist": "uniform"}, rng=rng),
        generators.random_walk_overlay(channel_name="d", spec={"rwo_scales": [1.0, 2.0]}, rng=rng),
        generators.white_noise(channel_name="e", spec={"wn_dist": "uniform"}, rng=rng),
    ]


def _draw_all(seed: int) -> list:
    return [_draw(gen) for gen in _make_generators(np.random.default_rng(seed))]


def test_seeded_generators_are_reproducible():
    for first, second in zip(_draw_all(3), _draw_all(3)):
        np.testing.assert_array_equal(first, second)


def test_different_seeds_give_different_values():
    for first, second in zip(_draw_all(3), _draw_all(4)):
        assert not np.array_equal(first, second)


def test_generators_ignore_global_numpy_state():
    np.random.seed(1)
    first = _draw_all(3)
    np.random.seed(2)
    second = _draw_all(3)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_control_init_range_uses_injected_rng():
    first = generators.control_from_actions(
        channel_name="DS", spec={"cfa_init": [2.0, 10.0]}, rng=np.random.default_rng(5)
    )
    second = generators.control_from_actions(
        channel_name="DS", spec={"cfa_init": [2.0, 10.0]}, rng=np.random.default_rng(5)
    )
    assert first.last_val == second.last_val
    assert 2.0 <= first.last_val <= 10.0


def test_rwo_offset_shifts_mean():
    spec = {"rwo_offset": 100.0, "rwo_scales": [0.1], "rwo_decays": [0.5]}
    gen = generators.random_walk_overlay(channel_name="AT", spec=spec, rng=np.random.default_rng(0))
    values = _draw(gen, 200)
    assert abs(values.mean() - 100.0) < 1.0


def test_sag_limits_are_respected():
    spec = {"sag_scale": 10.0, "sag_limits": [-1.0, 1.0]}
    gen = generators.spike_action_gen(channel_name="DS_action", spec=spec, rng=np.random.default_rng(0))
    values = _draw(gen, 200)
    assert values.min() >= -1.0
    assert values.max() <= 1.0
    assert np.isclose(np.abs(values).max(), 1.0)


def test_explicit_zero_arguments_are_kept():
    spike = generators.spike_action_gen(
        scale=0.0, channel_name="a", spec={"sag_scale": 5.0}, rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(_draw(spike), 0.0)

    noise = generators.white_noise(scale=0.0, channel_name="b", spec={"wn_scale": 5.0}, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(_draw(noise), 0.0)

    control = generators.control_from_actions(init=0.0, channel_name="c", spec={"cfa_init": 5.0})
    assert control.last_val == 0.0

    walk = generators.random_walk_overlay(offset=0.0, channel_name="d", spec={"rwo_offset": 5.0, "rwo_scales": [0.0]})
    np.testing.assert_array_equal(_draw(walk), 0.0)


def test_base_generator_keeps_zero_and_defaults_none():
    gen = generators.ChannelGenerator(channel_name="x")
    assert gen.step(_TIMESTEP, {}, 0.0) == 0.0
    assert gen.step(_TIMESTEP, {}, None) == 0.0


def test_get_num_plots_counts_temp_channels():
    spec = {
        "a": {"semantics": "action", "type": ["spike_action_gen", "controls_limiter"]},
        "t": {"semantics": "temp", "type": "function_callback"},
        "o": {"semantics": "target", "type": ["function_callback", "white_noise"]},
    }
    assert generators.get_num_plots(spec) == 2
    assert generators.get_num_plots(spec, with_temporaries=True) == 3
    assert generators.get_num_plots(spec, with_intermediates=True) == 4
    assert generators.get_num_plots(spec, with_intermediates=True, with_temporaries=True) == 5
