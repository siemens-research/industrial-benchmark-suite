"""Tests for the CCBench configuration helpers.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ibs.envs.ccbench.config_utils import channel_metadata
from ibs.envs.ccbench.config_utils import code_strings
from ibs.envs.ccbench.config_utils import deep_merge
from ibs.envs.ccbench.config_utils import parse_freq
from ibs.envs.ccbench.config_utils import resolve_callbacks
from ibs.envs.ccbench.config_utils import validate_ccbench_config


def test_deep_merge_merges_nested_mappings():
    base = {"process_spec": {"DS": {"cfa_init": [2.0, 10.0], "limits": [1.0, 12.0]}, "BP": {"cfa_init": 12.0}}}
    merged = deep_merge(base, {"process_spec": {"DS": {"cfa_init": 4.0}}})
    assert merged == {"process_spec": {"DS": {"cfa_init": 4.0, "limits": [1.0, 12.0]}, "BP": {"cfa_init": 12.0}}}


def test_deep_merge_replaces_lists_and_scalars():
    merged = deep_merge({"a": [1, 2, 3], "b": 1, "c": {"d": 1}}, {"a": [4], "b": {"x": 1}, "c": 2})
    assert merged == {"a": [4], "b": {"x": 1}, "c": 2}


def test_deep_merge_none_removes_key():
    merged = deep_merge({"a": 1, "b": {"c": 1, "d": 2}}, {"a": None, "b": {"d": None}, "e": None})
    assert merged == {"b": {"c": 1}}


def test_deep_merge_does_not_modify_inputs():
    base = {"a": {"b": [1, 2]}}
    override = {"a": {"c": {"d": 1}}}
    merged = deep_merge(base, override)
    merged["a"]["b"].append(3)
    merged["a"]["c"]["d"] = 2
    assert base == {"a": {"b": [1, 2]}}
    assert override == {"a": {"c": {"d": 1}}}
    assert deep_merge(base, None) == base


def test_resolve_callbacks_evaluates_strings():
    spec = {
        "x": {"fc_callback": "lambda a: np.abs(a)", "af_kernel": "norm_filter(signal.windows.hann(5))"},
        "y": {"fc_callback": abs},
        "z": {"type": "white_noise"},
    }
    resolved = resolve_callbacks(spec)
    assert resolved["x"]["fc_callback"](-2.0) == 2.0
    assert np.isclose(np.sum(resolved["x"]["af_kernel"]), 1.0)
    assert resolved["y"]["fc_callback"] is abs
    assert isinstance(spec["x"]["fc_callback"], str)


def test_resolve_callbacks_has_no_builtins():
    with pytest.raises(NameError):
        resolve_callbacks({"x": {"fc_callback": "open"}})


def _default_config(name: str = "ccbench_default.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[1] / "resources/envs/ccbench/config" / name
    with open(config_path, "rt", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def _v1_config() -> dict:
    return _default_config("ccbench_v1.yaml")


def test_default_config_is_valid():
    validate_ccbench_config(_default_config())
    validate_ccbench_config(_v1_config())


def _od_terms(*factors) -> dict:
    return {"process_spec": {"OD": {"lt_terms": [{"coef": 1.0, "factors": list(factors)}]}}}


def _transport(**transport) -> dict:
    return _od_terms({"input": "BP", "transport": {"speed": "DS", "distance": 0.25, "max_lag": 20, **transport}})


@pytest.mark.parametrize(
    "override, message",
    [
        ({"process_spec": {"OD": {"lt_terms": []}}}, "lt_terms must be a non-empty list"),
        ({"process_spec": {"OD": {"lt_terms": None}}}, "lt_terms is required"),
        ({"process_spec": {"OD": {"lt_terms": [{"coef": float("inf")}]}}}, r"lt_terms\[0\].coef"),
        ({"process_spec": {"OD": {"lt_terms": [{"coef": 1.0, "factor": []}]}}}, "unknown keys"),
        ({"process_spec": {"OD": {"lt_terms": [{"factors": {"input": "BP"}}]}}}, "factors must be a list"),
        (_od_terms({"input": "XX", "lag": 1}), "input must name a channel"),
        (_od_terms({"input": "BP", "lag": 49}), "lag must be an integer"),
        (_od_terms({"input": "BP", "lag": 1.5}), "lag must be an integer"),
        (_od_terms({"input": "WT"}), "lag 0 requires a channel computed before"),
        (_od_terms({"input": "OD"}), "lag 0 requires a channel computed before"),
        (_od_terms({"input": "BP_action"}), "lag 0 requires a channel computed before"),
        (_od_terms({"input": "BP", "power": 0}), "power must be"),
        (_od_terms({"input": "BP", "power": 4}), "power must be"),
        (_od_terms({"input": "BP", "scale": "x"}), "scale must be a finite number"),
        (_od_terms({"input": "BP", "delay": 3}), "unknown keys"),
        (_transport(), None),
        (_od_terms({"input": "BP", "lag": 1, "transport": {"speed": "DS", "distance": 1.0, "max_lag": 2}}), "mutually"),
        (_transport(speed="WT"), "speed must name a channel computed before"),
        (_transport(distance=0.0), "distance must be a positive number"),
        (_transport(speed_scale=-1.0), "speed_scale must be a positive number"),
        (_transport(max_lag=49), "min_lag and max_lag"),
        (_transport(min_lag=0), "min_lag and max_lag"),
        (_transport(length=3), "unknown keys"),
        ({"process_spec": {"wave": {"ms_amplitude": "x"}}}, "ms_amplitude must be a finite number"),
        ({"process_spec": {"wave": {"ms_phase": float("nan")}}}, "ms_phase must be a finite number"),
        ({"process_spec": {"wave": {"ms_input": "XX"}}}, "ms_input must name a channel"),
        ({"process_spec": {"AT": {"af_window": {"name": "kaiser", "length": 10}}}}, "af_window.name"),
        ({"process_spec": {"AT": {"af_window": {"name": "hann", "length": 0}}}}, "af_window.length"),
        ({"process_spec": {"AT": {"af_window": {"normalize": "yes"}}}}, "af_window.normalize"),
        ({"process_spec": {"AT": {"af_window": {"size": 3}}}}, "unknown keys"),
        ({"process_spec": {"AT": {"af_kernel": [0.5, 0.5]}}}, "mutually exclusive"),
        ({"process_spec": {"AT": {"af_window": None}}}, "af_kernel or af_window is required"),
    ],
)
def test_invalid_v1_config_raises(override, message):
    config = deep_merge(_v1_config(), override)
    if message is None:
        validate_ccbench_config(config)
        return
    with pytest.raises(ValueError, match=message):
        validate_ccbench_config(config)


def test_code_strings_lists_paths_of_string_callbacks():
    assert code_strings(_v1_config()) == []
    assert code_strings(_default_config()) == [
        "process_spec.AT.af_kernel",
        "process_spec.wave.fc_callback",
        "process_spec.OD.fc_callback",
        "process_spec.WT.fc_callback",
        "action_spec.HE_action.af_kernel",
    ]
    assert code_strings({"process_spec": {"OD": {"af_kernel": [0.5, 0.5], "fc_callback": abs}}}) == []


@pytest.mark.parametrize(
    "override, message",
    [
        ({"process_spec": {"AT": {"semantics": "sensor"}}}, "process_spec.AT.semantics"),
        ({"action_spec": {"BP_action": {"semantics": "state"}}}, "action_spec.BP_action.semantics"),
        ({"process_spec": {"AT": {"type": ["no_such_generator"]}}}, "unknown generators"),
        ({"process_spec": {"OD": {"limits": [15.0, 0.1]}}}, "process_spec.OD.limits"),
        ({"process_spec": {"OD": {"unit": 1}}}, "process_spec.OD.unit"),
        ({"process_spec": {"DS": {"assoc_action_channel": "XX_action"}}}, "assoc_action_channel"),
        ({"process_spec": {"DS": {"cfa_init": 20.0}}}, "process_spec.DS.cfa_init"),
        ({"process_spec": {"DS": {"cfa_init": [8.0, 2.0]}}}, "process_spec.DS.cfa_init"),
        ({"process_spec": {"OD": {"vop_type": "maximize"}}}, "process_spec.OD.vop_type"),
        ({"desired_goal": {"AT": 5.0}}, "not a target channel"),
        ({"desired_goal": {"OD": 20.0}}, "outside of the channel limits"),
        ({"desired_goal": {"OD": [[0, 5.0], [10, 20.0]]}}, "outside of the channel limits"),
        ({"desired_goal": {"OD": [[5, 5.0]]}}, "must start at step 0"),
        ({"desired_goal": {"OD": [[0, 5.0], [10, 6.0], [10, 7.0]]}}, "strictly increasing"),
        ({"desired_goal": {"OD": [[0.5, 5.0]]}}, "integer step"),
        ({"desired_goal": {"OD": [[0, [1.0, 2.0]]]}}, "numeric value"),
        ({"freq": 0}, "freq"),
        ({"freq": "-1s"}, "freq"),
        ({"episode_length": 0}, "episode_length"),
        ({"episode_length": 2.5}, "episode_length"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"warmup_steps": True}, "warmup_steps"),
        ({"truncate_on_time_limit": "yes"}, "truncate_on_time_limit"),
        ({"start_timestamp": "not a timestamp"}, "start_timestamp"),
    ],
)
def test_invalid_config_raises(override, message):
    with pytest.raises(ValueError, match=message):
        validate_ccbench_config(deep_merge(_default_config(), override))


def test_validation_reports_all_errors():
    config = deep_merge(_default_config(), {"episode_length": 0, "warmup_steps": -1})
    with pytest.raises(ValueError) as excinfo:
        validate_ccbench_config(config)
    assert "episode_length" in str(excinfo.value)
    assert "warmup_steps" in str(excinfo.value)


def test_valid_goal_and_flags_are_accepted():
    override = {
        "desired_goal": {"OD": [[0, 5.0], [100, 7.5]], "WT": 0.6},
        "freq": "1s",
        "episode_length": 10,
        "warmup_steps": "auto",
        "truncate_on_time_limit": False,
        "start_timestamp": "2020-05-01T12:00:00",
    }
    validate_ccbench_config(deep_merge(_default_config(), override))


def test_parse_freq():
    assert parse_freq(None) == pd.Timedelta(seconds=1)
    assert parse_freq(2) == pd.Timedelta(seconds=2)
    assert parse_freq("5min") == pd.Timedelta(minutes=5)
    assert parse_freq(dt.timedelta(seconds=3)) == pd.Timedelta(seconds=3)
    with pytest.raises(ValueError):
        parse_freq(0)


def test_channel_metadata():
    spec = {
        "BP_action": {"semantics": "action", "limits": [-4, 4], "unit": "mbar/step"},
        "BP": {"semantics": "control", "assoc_action_channel": "BP_action", "limits": [5, 30], "unit": "mbar"},
        "wave": {"semantics": "temp"},
        "OD": {"semantics": "target", "vop_type": "setpoint", "limits": [0.1, 15.0], "unit": "mm"},
        "OD_obj": {"semantics": "objective"},
    }
    metadata = channel_metadata(spec, freq=1)
    assert list(metadata) == ["BP_action", "BP", "OD"]
    assert metadata["BP"] == {
        "semantics": "control",
        "limits": [5.0, 30.0],
        "unit": "mbar",
        "step_seconds": 1.0,
        "action_channel": "BP_action",
    }
    assert metadata["OD"]["vop_type"] == "setpoint"
    assert "wave" in channel_metadata(spec, include_temp=True)
    assert channel_metadata(spec, include_temp=True)["wave"]["limits"] == [-np.inf, np.inf]
