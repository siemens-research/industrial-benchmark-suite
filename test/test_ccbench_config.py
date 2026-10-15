"""Tests for the CCBench configuration helpers."""

import numpy as np
import pytest

from ibs.envs.ccbench.config_utils import deep_merge
from ibs.envs.ccbench.config_utils import resolve_callbacks


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
