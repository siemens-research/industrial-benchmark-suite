"""
Helpers for CCBench configuration handling.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import datetime as dt
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from ibs.common.config import load_config

_CALLBACK_KEYS = ("fc_callback", "af_kernel")
# maximum lag (in timesteps) of `lagged_terms` factors, below `ccbench_env.MAX_AUTO_WARMUP_STEPS`
MAX_LAG = 48
# maximum absolute power of `lagged_terms` factors
MAX_POWER = 3
# maximum length of `af_window` filter kernels
MAX_WINDOW_LENGTH = 2000
_FACTOR_KEYS = ("input", "lag", "scale", "offset", "power", "transport")
_TRANSPORT_KEYS = ("speed", "distance", "speed_scale", "min_lag", "max_lag")
_WINDOW_KEYS = ("name", "length", "normalize")
_SINE_NUMBER_KEYS = ("ms_amplitude", "ms_exp_scale", "ms_exp_offset", "ms_angular_frequency", "ms_phase")


def deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recursively merge two configuration mappings without modifying them.

    Nested mappings are merged key by key, all other values (including lists) of `override` replace those of
    `base`. A value of None in `override` removes the corresponding key from the result.

    Args:
        base: Mapping with the base configuration.
        override: Mapping whose values take precedence over those in `base`; may be None.

    Returns:
        New dictionary containing the merged configuration.
    """
    merged: Dict[str, Any] = {k: _copy_value(v) for k, v in base.items()}
    if override is None:
        return merged
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = _copy_value(value)
    return merged


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _copy_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_value(v) for v in value]
    return value


def resolve_callbacks(spec: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Turn callback strings of a channel specification into callables.

    The values of `fc_callback` and `af_kernel` given as strings are evaluated as Python expressions with a
    restricted namespace (`np`, `signal`, `norm_filter`, `soft_hinge`, no builtins). This is not a sandbox:
    configuration files containing such strings are trusted code and must not come from untrusted sources.

    Args:
        spec: Mapping from channel names to channel specifications (e.g. `process_spec` or `action_spec`).

    Returns:
        New dictionary with copied channel specifications, callback strings replaced by the evaluated objects.
    """
    from scipy import signal

    from ibs.envs.ccbench.functions import norm_filter
    from ibs.envs.ccbench.functions import soft_hinge

    callbacks_namespace = {
        "np": np,
        "signal": signal,
        "norm_filter": norm_filter,
        "soft_hinge": soft_hinge,
    }

    def resolve_callback(callback_spec: str):
        safe_globals: Dict[str, Any] = {"__builtins__": {}}
        safe_globals.update(callbacks_namespace)
        return eval(callback_spec, safe_globals, {})  # pylint: disable=eval-used

    resolved = {k: dict(v) for k, v in spec.items()}
    for series_spec in resolved.values():
        for key in _CALLBACK_KEYS:
            if isinstance(series_spec.get(key), str):
                series_spec[key] = resolve_callback(series_spec[key])
    return resolved


def code_strings(config: Mapping[str, Any]) -> List[str]:
    """List the code strings of a configuration, which `resolve_callbacks` would evaluate.

    Args:
        config: Configuration (or partial configuration) with optional `process_spec` and `action_spec`.

    Returns:
        Paths (e.g. `process_spec.OD.fc_callback`) of all `fc_callback` and `af_kernel` values given as strings.
    """
    paths: List[str] = []
    for spec_name in ("process_spec", "action_spec"):
        spec = config.get(spec_name)
        if not isinstance(spec, Mapping):
            continue
        for channel, channel_spec in spec.items():
            if not isinstance(channel_spec, Mapping):
                continue
            paths.extend(
                f"{spec_name}.{channel}.{key}" for key in _CALLBACK_KEYS if isinstance(channel_spec.get(key), str)
            )
    return paths


def build_ccbench_config(
    defaults: Mapping[str, Any],
    config_file: Optional[Union[str, Path]] = None,
    config_dict: Optional[Mapping[str, Any]] = None,
    config_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    data_dir_default: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a merged configuration for data generation.

    Args:
        defaults: Default configuration values.
        config_file: Optional YAML file path to load.
        config_dict: Optional mapping merged after file values.
        config_kwargs: Optional mapping merged last.
        data_dir_default: Fallback directory for output data.

    Returns:
        Merged configuration dictionary.
    """
    config_dict = {} if config_dict is None else dict(config_dict)
    config_kwargs = {} if config_kwargs is None else dict(config_kwargs)
    if "data_dir" in config_dict and config_dict["data_dir"] is None:
        config_dict.pop("data_dir")
    if "data_dir" in config_kwargs and config_kwargs["data_dir"] is None:
        config_kwargs.pop("data_dir")

    config = load_config(config_file=config_file, config_dict=config_dict, config_kwargs=config_kwargs)
    if "data_dir" not in config or config.get("data_dir") is None:
        config["data_dir"] = data_dir_default or (Path(__file__).resolve().parents[3] / "data")

    now = dt.datetime.now()
    if "date" not in config:
        config["date"] = now
    if "id" not in config:
        config["id"] = defaults.get("id", "data").format(**config)
    else:
        config["id"] = config["id"].format(**config)

    merged: Dict[str, Any] = dict(defaults)
    merged.update(config)
    return merged


CHANNEL_SEMANTICS = ("action", "control", "state", "context", "target", "temp")
VOP_TYPES = ("setpoint", "upper_limit", "lower_limit")
BOOL_CONFIG_KEYS = ("truncate_on_time_limit", "expose_temp_channels", "keep_intermediate_series")


def parse_freq(freq: Any) -> pd.Timedelta:
    """Convert a step duration specification to a `pd.Timedelta`.

    Args:
        freq: Step duration as number of seconds, as pandas timedelta string (e.g. "1s", "5min"), as
            `datetime.timedelta`/`pd.Timedelta`, or as fixed-frequency pandas offset; None means one second.

    Returns:
        Positive step duration.
    """
    if freq is None:
        return pd.Timedelta(seconds=1)
    if isinstance(freq, bool):
        raise ValueError(f"freq must be a positive duration, got {freq!r}")
    if isinstance(freq, (int, float, np.integer, np.floating)):
        result = pd.Timedelta(seconds=float(freq))
    else:
        result = pd.Timedelta(freq)
    if result is pd.NaT or result <= pd.Timedelta(0):
        raise ValueError(f"freq must be a positive duration, got {freq!r}")
    return result


def channel_metadata(
    process_spec: Mapping[str, Mapping[str, Any]], freq: Any = None, include_temp: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Describe the channels of a CCBench process specification.

    Args:
        process_spec: Channel specifications including the action channels (e.g. `CCBenchEnv.process_spec`).
        freq: Step duration, see `parse_freq`.
        include_temp: Also describe channels with semantics `temp`.

    Returns:
        Mapping from channel name to a dict with `semantics`, `limits` (`[low, high]`, infinite if unbounded),
        `unit` (None if not specified) and `step_seconds`; control channels additionally contain `action_channel`,
        target channels `vop_type`. Objective channels are not included.
    """
    step_seconds = parse_freq(freq).total_seconds()
    result: Dict[str, Dict[str, Any]] = {}
    for name, spec in process_spec.items():
        semantics = spec.get("semantics")
        if semantics not in CHANNEL_SEMANTICS or (semantics == "temp" and not include_temp):
            continue
        limits = spec.get("limits", [-np.inf, np.inf])
        entry: Dict[str, Any] = {
            "semantics": semantics,
            "limits": [float(limits[0]), float(limits[1])],
            "unit": spec.get("unit"),
            "step_seconds": step_seconds,
        }
        if semantics == "control":
            entry["action_channel"] = spec.get("assoc_action_channel")
        if semantics == "target":
            entry["vop_type"] = spec.get("vop_type")
        result[name] = entry
    return result


def validate_ccbench_config(config: Mapping[str, Any]) -> None:
    """Check a CCBench environment configuration for consistency.

    Only keys used by the environment are checked, data generation keys (e.g. `episodes`, plot settings) are
    ignored. All detected problems are collected and reported together.

    Args:
        config: Environment configuration with `process_spec`, optional `action_spec` and further environment keys.

    Raises:
        ValueError: if the configuration is invalid; the message lists all detected problems.
    """
    errors: List[str] = []
    process_spec = config.get("process_spec")
    action_spec = config.get("action_spec")
    if not isinstance(process_spec, Mapping) or len(process_spec) == 0:
        errors.append("process_spec must be a non-empty mapping of channel specifications")
        process_spec = {}
    if action_spec is None:
        action_spec = {}
    elif not isinstance(action_spec, Mapping):
        errors.append("action_spec must be a mapping of channel specifications")
        action_spec = {}

    # channels in the order of their computation by the environment (action channels not in process_spec last)
    order = list(process_spec) + [k for k in action_spec if k not in process_spec]
    for spec_name, spec in (("process_spec", process_spec), ("action_spec", action_spec)):
        for channel, channel_spec in spec.items():
            errors.extend(_validate_channel(f"{spec_name}.{channel}", channel_spec, spec_name == "action_spec"))
            if isinstance(channel_spec, Mapping):
                earlier = order[: order.index(channel)] if spec_name == "process_spec" else []
                errors.extend(_generator_param_errors(f"{spec_name}.{channel}", channel_spec, order, earlier))

    channels = {k: v for k, v in {**action_spec, **process_spec}.items() if isinstance(v, Mapping)}
    action_channels = {k for k, v in channels.items() if v.get("semantics") == "action"}
    targets = {k: v for k, v in process_spec.items() if isinstance(v, Mapping) and v.get("semantics") == "target"}
    for channel, spec in process_spec.items():
        if not isinstance(spec, Mapping):
            continue
        if spec.get("semantics") == "control":
            assoc = spec.get("assoc_action_channel")
            if assoc not in action_channels:
                errors.append(f"process_spec.{channel}.assoc_action_channel must name an action channel, got {assoc!r}")
            errors.extend(_validate_value_in_limits(f"process_spec.{channel}.cfa_init", spec.get("cfa_init"), spec))
        if spec.get("semantics") == "target" and spec.get("vop_type") not in VOP_TYPES:
            errors.append(f"process_spec.{channel}.vop_type must be one of {VOP_TYPES}, got {spec.get('vop_type')!r}")

    errors.extend(desired_goal_errors(config.get("desired_goal"), targets))

    if config.get("freq") is not None:
        try:
            parse_freq(config["freq"])
        except (ValueError, TypeError) as e:
            errors.append(f"freq: {e}")
    episode_length = config.get("episode_length")
    if episode_length is not None and (not _is_int(episode_length) or episode_length <= 0):
        errors.append(f"episode_length must be a positive integer, got {episode_length!r}")
    warmup_steps = config.get("warmup_steps")
    if warmup_steps is not None and warmup_steps != "auto" and (not _is_int(warmup_steps) or warmup_steps < 0):
        errors.append(f"warmup_steps must be 'auto' or a non-negative integer, got {warmup_steps!r}")
    for key in BOOL_CONFIG_KEYS:
        if config.get(key) is not None and not isinstance(config[key], bool):
            errors.append(f"{key} must be a boolean, got {config[key]!r}")
    if config.get("start_timestamp") is not None:
        try:
            if pd.Timestamp(config["start_timestamp"]) is pd.NaT:
                raise ValueError("NaT")
        except (ValueError, TypeError):
            errors.append(f"start_timestamp is not a valid timestamp: {config['start_timestamp']!r}")

    if errors:
        raise ValueError("Invalid CCBench configuration:\n- " + "\n- ".join(errors))


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _validate_channel(name: str, spec: Any, is_action_spec: bool) -> List[str]:
    from ibs.envs.ccbench import generators

    if not isinstance(spec, Mapping):
        return [f"{name} must be a mapping"]
    errors = []
    allowed_semantics = ("action", "temp") if is_action_spec else CHANNEL_SEMANTICS
    if spec.get("semantics") not in allowed_semantics:
        errors.append(f"{name}.semantics must be one of {allowed_semantics}, got {spec.get('semantics')!r}")
    types = spec.get("type")
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, (list, tuple)):
        errors.append(f"{name}.type must be a generator name or a list of generator names, got {types!r}")
    else:
        available = {
            k
            for k, v in vars(generators).items()
            if isinstance(v, type)
            and issubclass(v, generators.ChannelGenerator)
            and v is not generators.ChannelGenerator
        }
        unknown = [t for t in types if t not in available]
        if unknown:
            errors.append(f"{name}.type contains unknown generators {unknown}")
    if "limits" in spec:
        limits = spec["limits"]
        if (
            not isinstance(limits, (list, tuple))
            or len(limits) != 2
            or not all(_is_number(v) for v in limits)
            or not limits[0] < limits[1]
        ):
            errors.append(f"{name}.limits must be two numbers [low, high] with low < high, got {limits!r}")
    if "unit" in spec and not isinstance(spec["unit"], str):
        errors.append(f"{name}.unit must be a string, got {spec['unit']!r}")
    return errors


def _types(spec: Mapping[str, Any]) -> List[str]:
    types = spec.get("type")
    if isinstance(types, str):
        return [types]
    return list(types) if isinstance(types, (list, tuple)) else []


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and bool(np.isfinite(value))


def _generator_param_errors(name: str, spec: Mapping[str, Any], channels: List[str], earlier: List[str]) -> List[str]:
    """Check the parameters of the data-only generators `lagged_terms`, `modulated_sine` and `apply_filter`.

    Args:
        name: Name of the channel specification for error messages.
        spec: Channel specification.
        channels: Names of all channels.
        earlier: Names of the channels computed before this one in each timestep (allowed for lag 0).

    Returns:
        List of detected problems.
    """
    types = _types(spec)
    errors = []
    if "lagged_terms" in types and "lt_terms" not in spec:
        errors.append(f"{name}.lt_terms is required for generator lagged_terms")
    if "lt_terms" in spec:
        errors.extend(_terms_errors(f"{name}.lt_terms", spec["lt_terms"], channels, earlier))
    if "modulated_sine" in types:
        for key in _SINE_NUMBER_KEYS:
            if key in spec and not _is_finite_number(spec[key]):
                errors.append(f"{name}.{key} must be a finite number, got {spec[key]!r}")
        if spec.get("ms_input") is not None and spec["ms_input"] not in channels:
            errors.append(f"{name}.ms_input must name a channel, got {spec['ms_input']!r}")
    if "af_window" in spec:
        if "af_kernel" in spec:
            errors.append(f"{name}: af_window and af_kernel are mutually exclusive")
        errors.extend(_window_errors(f"{name}.af_window", spec["af_window"]))
    elif "apply_filter" in types and "af_kernel" not in spec:
        errors.append(f"{name}: af_kernel or af_window is required for generator apply_filter")
    return errors


def _terms_errors(name: str, terms: Any, channels: List[str], earlier: List[str]) -> List[str]:
    if not isinstance(terms, (list, tuple)) or len(terms) == 0:
        return [f"{name} must be a non-empty list of terms {{coef, factors}}, got {terms!r}"]
    errors = []
    for i, term in enumerate(terms):
        term_name = f"{name}[{i}]"
        if not isinstance(term, Mapping):
            errors.append(f"{term_name} must be a mapping {{coef, factors}}, got {term!r}")
            continue
        unknown = sorted(set(term) - {"coef", "factors"})
        if unknown:
            errors.append(f"{term_name} contains unknown keys {unknown}")
        if "coef" in term and not _is_finite_number(term["coef"]):
            errors.append(f"{term_name}.coef must be a finite number, got {term['coef']!r}")
        factors = term.get("factors", [])
        if not isinstance(factors, (list, tuple)):
            errors.append(f"{term_name}.factors must be a list of factors, got {factors!r}")
            continue
        for j, factor in enumerate(factors):
            errors.extend(_factor_errors(f"{term_name}.factors[{j}]", factor, channels, earlier))
    return errors


def _factor_errors(name: str, factor: Any, channels: List[str], earlier: List[str]) -> List[str]:
    if not isinstance(factor, Mapping):
        return [f"{name} must be a mapping {{input, lag, scale, offset, power}}, got {factor!r}"]
    errors = []
    unknown = sorted(set(factor) - set(_FACTOR_KEYS))
    if unknown:
        errors.append(f"{name} contains unknown keys {unknown}")
    channel = factor.get("input")
    if channel not in channels:
        errors.append(f"{name}.input must name a channel, got {channel!r}")
    if "transport" in factor:
        if "lag" in factor:
            errors.append(f"{name}: lag and transport are mutually exclusive")
        errors.extend(_transport_errors(f"{name}.transport", factor["transport"], earlier))
    else:
        lag = factor.get("lag", 0)
        if not _is_int(lag) or not 0 <= lag <= MAX_LAG:
            errors.append(f"{name}.lag must be an integer in [0, {MAX_LAG}], got {lag!r}")
        elif lag == 0 and channel in channels and channel not in earlier:
            errors.append(f"{name}: lag 0 requires a channel computed before this one ({earlier}), got {channel!r}")
    for key in ("scale", "offset"):
        if key in factor and not _is_finite_number(factor[key]):
            errors.append(f"{name}.{key} must be a finite number, got {factor[key]!r}")
    power = factor.get("power", 1)
    if not _is_finite_number(power) or power == 0 or abs(power) > MAX_POWER:
        errors.append(f"{name}.power must be a non-zero number in [-{MAX_POWER}, {MAX_POWER}], got {power!r}")
    return errors


def _transport_errors(name: str, transport: Any, earlier: List[str]) -> List[str]:
    if not isinstance(transport, Mapping):
        return [f"{name} must be a mapping {{speed, distance, speed_scale, min_lag, max_lag}}, got {transport!r}"]
    errors = []
    unknown = sorted(set(transport) - set(_TRANSPORT_KEYS))
    if unknown:
        errors.append(f"{name} contains unknown keys {unknown}")
    if transport.get("speed") not in earlier:
        errors.append(
            f"{name}.speed must name a channel computed before this one ({earlier}), got {transport.get('speed')!r}"
        )
    for key, default in (("distance", None), ("speed_scale", 1.0)):
        value: Any = transport.get(key, default)
        if not _is_finite_number(value) or value <= 0:
            errors.append(f"{name}.{key} must be a positive number, got {value!r}")
    min_lag: Any = transport.get("min_lag", 1)
    max_lag: Any = transport.get("max_lag")
    if not _is_int(min_lag) or not _is_int(max_lag) or not 1 <= min_lag <= max_lag <= MAX_LAG:
        errors.append(
            f"{name}: min_lag and max_lag must be integers with 1 <= min_lag <= max_lag <= {MAX_LAG}, "
            f"got {min_lag!r} and {max_lag!r}"
        )
    return errors


def _window_errors(name: str, window: Any) -> List[str]:
    from ibs.envs.ccbench.functions import WINDOW_NAMES

    if not isinstance(window, Mapping):
        return [f"{name} must be a mapping {{name, length, normalize}}, got {window!r}"]
    errors = []
    unknown = sorted(set(window) - set(_WINDOW_KEYS))
    if unknown:
        errors.append(f"{name} contains unknown keys {unknown}")
    if window.get("name") not in WINDOW_NAMES:
        errors.append(f"{name}.name must be one of {WINDOW_NAMES}, got {window.get('name')!r}")
    length: Any = window.get("length")
    if not _is_int(length) or not 1 <= length <= MAX_WINDOW_LENGTH:
        errors.append(f"{name}.length must be an integer in [1, {MAX_WINDOW_LENGTH}], got {length!r}")
    if "normalize" in window and not isinstance(window["normalize"], bool):
        errors.append(f"{name}.normalize must be a boolean, got {window['normalize']!r}")
    return errors


def _valid_limits(spec: Mapping[str, Any]) -> Tuple[float, float]:
    limits = spec.get("limits")
    if isinstance(limits, (list, tuple)) and len(limits) == 2 and all(_is_number(v) for v in limits):
        return float(limits[0]), float(limits[1])
    return -np.inf, np.inf


def _validate_value_in_limits(name: str, value: Any, spec: Mapping[str, Any]) -> List[str]:
    if value is None:
        return []
    low, high = _valid_limits(spec)
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or not all(_is_number(v) for v in value) or value[0] > value[1]:
            return [f"{name} must be a number or a range [low, high] with low <= high, got {value!r}"]
        values = list(value)
    elif _is_number(value):
        values = [value]
    else:
        return [f"{name} must be a number or a range [low, high], got {value!r}"]
    if any(not low <= v <= high for v in values):
        return [f"{name}={value!r} is outside of the channel limits [{low}, {high}]"]
    return []


def desired_goal_errors(desired_goal: Any, process_spec: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Check a `desired_goal` configuration (value or schedule `[[step, value], ...]` per target channel).

    Args:
        desired_goal: Goal configuration to check, None is valid.
        process_spec: Channel specifications, the target channels are used for checking names and limits.

    Returns:
        List of detected problems, empty if the goal configuration is valid.
    """
    if desired_goal is None:
        return []
    targets = {k: v for k, v in process_spec.items() if isinstance(v, Mapping) and v.get("semantics") == "target"}
    if not isinstance(desired_goal, Mapping):
        return [f"desired_goal must be a mapping from target channel names to values, got {desired_goal!r}"]
    errors = []
    for target, goal in desired_goal.items():
        name = f"desired_goal.{target}"
        if target not in targets:
            errors.append(f"{name}: {target!r} is not a target channel (targets: {sorted(targets)})")
            continue
        if _is_number(goal):
            errors.extend(_validate_value_in_limits(name, goal, targets[target]))
            continue
        if not isinstance(goal, (list, tuple)) or len(goal) == 0:
            errors.append(f"{name} must be a number or a non-empty schedule [[step, value], ...], got {goal!r}")
            continue
        steps = []
        for entry in goal:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or not _is_int(entry[0])
                or not _is_number(entry[1])
            ):
                errors.append(
                    f"{name} schedule entries must be [step, value] with integer step and numeric value, got {entry!r}"
                )
                break
            steps.append(entry[0])
            errors.extend(_validate_value_in_limits(f"{name}[step {entry[0]}]", entry[1], targets[target]))
        else:
            if steps[0] != 0:
                errors.append(f"{name} schedule must start at step 0, got {steps[0]}")
            if any(b <= a for a, b in zip(steps, steps[1:])):
                errors.append(f"{name} schedule steps must be strictly increasing, got {steps}")
    return errors
