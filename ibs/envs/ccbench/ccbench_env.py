"""
Generation of artificial multivariate time-series data, suitable for use as OpenAI Gym environment.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import copy
import datetime as dt
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import gymnasium as gym
import numpy as np
import pandas as pd

from ibs.common.config import load_config
from ibs.common.goal_env import GoalEnv
from ibs.common.sequences import is_sequence

from . import generators
from .config_utils import resolve_callbacks

DEFAULT_START_TIMESTAMP = "2000-01-01T00:00:00Z"


class CCBenchEnv(GoalEnv):
    """Goal-based synthetic time-series environment driven by a channel specification.

    All random numbers are drawn from the environment's `np_random` generator, so `reset(seed=...)` makes the
    following episode reproducible. A seeded reset starts the episode at the configured `start_timestamp`, an unseeded
    reset continues the random stream and starts the episode `episode_step` after the end of the previous one.
    """

    def __init__(
        self,
        config_file: Optional[Path | str] = None,
        config_dict: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """Initialize the environment from a channel specification.

        Args:
            config_file: Optional YAML file providing configuration values.
            config_dict: Optional mapping of configuration values.
        """
        super().__init__(*args, **kwargs)
        if config_file is None and config_dict is None:
            config_file = Path(__file__).resolve().parents[3] / "resources/envs/ccbench/config/ccbench_default.yaml"
        config = load_config(config_file=config_file, config_dict=config_dict)

        process_spec = config.get("process_spec")
        if process_spec is None:
            raise ValueError("process_spec must be provided directly or via config")

        start_timestamp = config.get("start_timestamp")
        freq = config.get("freq")
        episode_step = config.get("episode_step")
        render_file = config.get("render_file")
        keep_intermediate_series = config.get("keep_intermediate_series", True)
        max_episode_steps = config.get("episode_length", None)

        if "action_spec" in config:
            process_spec = _apply_action_limits(process_spec, config["action_spec"])
        process_spec = resolve_callbacks(copy.deepcopy(dict(process_spec)))

        self.process_spec: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in process_spec.items()}
        self.start_timestamp = _coerce_timestamp(
            start_timestamp if start_timestamp is not None else DEFAULT_START_TIMESTAMP
        )
        self.cur_timestep: Optional[pd.Timestamp] = None
        self.next_timestep: Optional[pd.Timestamp] = None
        self._episode_start: Optional[pd.Timestamp] = None
        self.timestep_size = freq or pd.Timedelta(seconds=1)
        self.episode_step_size = episode_step or pd.DateOffset(days=1, normalize=True)
        self.keep_intermediate_series = keep_intermediate_series
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.action_control_map: Dict[str, Optional[str]] = {
            k: None for k, spec in self.process_spec.items() if spec["semantics"] == "action"
        }
        for k, spec in self.process_spec.items():
            if spec["semantics"] == "control":
                self.action_control_map[spec["assoc_action_channel"]] = k
        self.target_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "target"]
        self.state_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "state"]
        self.context_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "context"]
        self.temp_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "temp"]
        self.target_obj_map = {k: f"{k}_obj" for k, spec in self.process_spec.items() if spec["semantics"] == "target"}
        self._complete_process_spec()

        self.metadata["render.modes"] = ["human"]
        self.render_file = render_file
        if self.render_file is not None:
            self.metadata["render.modes"].append("file")
        self.reward_range = (-float("inf"), 0.0)

        def get_box_space(channel):
            if channel in self.process_spec and "limits" in self.process_spec[channel]:
                return gym.spaces.Box(
                    low=np.array(self.process_spec[channel]["limits"][0]),
                    high=np.array(self.process_spec[channel]["limits"][1]),
                    shape=None,
                    dtype=np.float32,
                )
            else:
                return gym.spaces.Box(low=np.array(-np.inf), high=np.array(np.inf), shape=None, dtype=np.float32)

        state_space: Dict[str, gym.spaces.Space] = {
            k: get_box_space(k)
            for k in self.process_spec.keys()
            if (k not in self.action_control_map or self.action_control_map[k] is None)
            and (k not in self.target_obj_map.values())
        }
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(state_space),
                "achieved_goal": gym.spaces.Dict({k: get_box_space(k) for k in self.target_channels}),
                "desired_goal": gym.spaces.Dict({k: get_box_space(k) for k in self.target_channels}),
            }
        )
        self.action_space = gym.spaces.Dict({k: get_box_space(k) for k, c in self.action_control_map.items()})

        # names of intermediate series only depend on generator parameters, not on random draws
        self.intermediate_map = {
            k: [f"{k}_{gen.info_str(verbosity=2)}" for gen in gens[:-1]]
            for k, gens in self._build_generators(np.random.default_rng(0)).items()
        }

        self.index: List[pd.Timestamp] = []
        self.desired_goal: Optional[Dict[str, float]] = None
        self.history: Dict[str, pd.Series] = {}
        self.replay: Optional[pd.DataFrame] = None
        self._generators: Dict[str, List[generators.ChannelGenerator]] = {}
        self.reset(perform_step=False)

    def _complete_process_spec(self) -> None:
        """Normalize generator types to lists and add channel entries for actions without spec and objectives."""
        for k in self.action_control_map:
            if k not in self.process_spec:
                self.process_spec[k] = dict(type=[], semantics="action")
        for k, o in self.target_obj_map.items():
            if o not in self.process_spec:
                self.process_spec[o] = dict(
                    type=[],
                    semantics="objective",
                    assoc_target_channel=k,
                    limits=self.process_spec[k].get("limits", [-np.inf, np.inf]),
                )
        for k, spec in self.process_spec.items():
            if not is_sequence(spec["type"]):
                spec["type"] = [spec["type"]]
            non_available = [typ for typ in spec["type"] if typ not in generators.__dict__]
            if len(non_available) > 0:
                raise ValueError(
                    f"The following generators are not available (key 'type' of specification for '{k}'): {non_available}"
                )

    def _build_generators(
        self, rng: np.random.Generator, replay: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[generators.ChannelGenerator]]:
        """Create fresh channel generator instances for all channels, drawing random numbers from `rng`."""
        return {
            k: [
                generators.__dict__[typ](
                    channel_name=k,
                    spec=spec,
                    replay=(
                        replay[k]
                        if ((i == len(spec["type"]) - 1) and replay is not None and k in replay.columns)
                        else None
                    ),
                    rng=rng,
                )
                for i, typ in enumerate(spec["type"])
            ]
            for k, spec in self.process_spec.items()
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
        episode_start: Optional[dt.datetime] = None,
        perform_step: bool = True,
        replay: Optional[pd.DataFrame] = None,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment state and optionally perform the first step.

        Args:
            seed: RNG seed for the environment's `np_random` generator, which drives all random channel generators.
                With a seed, the episode starts at `start_timestamp` (unless specified otherwise) and is reproducible.
            options: Optional reset options. Supported key: `episode_start` (timestamp, same as the `episode_start`
                argument, taking precedence over it).
            episode_start: Explicit start timestamp. If omitted, seeded resets and the first episode start at
                `start_timestamp`, unseeded resets start `episode_step` after the last timestep of the previous
                episode. Naive timestamps are interpreted as UTC.
            perform_step: If True, performs one step with zero actions to return a valid initial observation.
            replay: Optional DataFrame indexed by timestamps to replay channels.
                If provided, `episode_start` is derived from its index.
        """
        super().reset(seed=seed, options=dict(options) if options is not None else None)
        options = dict(options) if options is not None else {}

        # find specified or suitable episode start timestamp
        if replay is not None:
            start_ts = replay.index.min()
        elif options.get("episode_start") is not None:
            start_ts = options["episode_start"]
        elif episode_start is not None:
            start_ts = episode_start
        elif seed is not None or self._episode_start is None:
            start_ts = self.start_timestamp
        else:
            last_ts = self.cur_timestep if self.cur_timestep is not None else self._episode_start
            start_ts = last_ts + self.episode_step_size
        self._episode_start = _coerce_timestamp(start_ts)
        self.replay = replay
        self.next_timestep = self._episode_start
        self.cur_timestep = None
        self.index = []
        self.step_count = 0
        self.desired_goal = None

        # setup empty history and new generator instances for all channels
        self.history = {
            k: pd.Series([], dtype=np.float64, name=k, index=pd.DatetimeIndex([], tz="UTC")) for k in self.process_spec
        }
        self._generators = self._build_generators(self.np_random, replay)

        if perform_step:
            # observation, _, terminated, truncated, info = self.step({k: np.nan for k in self.action_control_map.keys()})
            observation, _, terminated, truncated, info = self.step({k: 0.0 for k in self.action_control_map.keys()})
            if terminated or truncated:
                info.setdefault("terminal_reason", "reset_step")
            return observation, info
        return {}, {}

    def step(
        self,
        action: Union[Mapping[str, float], np.ndarray],
        desired_goal: Optional[Mapping[str, float]] = None,
    ) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one timestep using the provided action.

        Args:
            action: Either a mapping of action channels to values or a numpy array
                aligned with `action_control_map` order.
            desired_goal: Optional goal override for target channels. If omitted,
                goals may be derived or replayed.
        """
        observation = {}
        terminated = False
        truncated = False
        info = {}

        self.step_count += 1
        self.cur_timestep = self.next_timestep
        if self.cur_timestep is None:
            raise RuntimeError("cur_timestep is None at step entry")
        cur_ts = self.cur_timestep
        self.index.append(cur_ts)

        def get_replay_goal():
            if self.replay is None:
                return None
            if cur_ts not in self.replay.index:
                return None
            goal = {}
            for t, o in self.target_obj_map.items():
                if o not in self.replay.columns:
                    return None
                goal[t] = float(self.replay.at[cur_ts, o])  # type: ignore[arg-type]
            return goal

        self.desired_goal = desired_goal or get_replay_goal() or self.desired_goal or self.derive_desired_goal()
        action_is_dict = not isinstance(action, (np.ndarray,))
        sample_values_dict: Dict[str, float] = {}
        sample_values_list: List[float] = []
        goal_values: Dict[str, float] = {}
        # iterate generation of current channel values for each channel
        for k, spec in self.process_spec.items():
            new_val = None
            if spec["semantics"] == "action":
                new_val = action[k] if action_is_dict else action[list(self.action_control_map.keys()).index(k)]
            if spec["semantics"] == "control":
                new_val = (
                    action[spec["assoc_action_channel"]]
                    if action_is_dict
                    else action[list(self.action_control_map.keys()).index(spec["assoc_action_channel"])]
                )
            if spec["semantics"] == "objective":
                new_val = self.desired_goal[spec["assoc_target_channel"]]
            # compute chain of channel generators for the current channel
            for j, gen in enumerate(self._generators[k]):
                old_val = new_val
                new_val = gen.step(timestep=self.cur_timestep, history=self.history, cur_val=new_val)
                if not isinstance(new_val, (float, int)):
                    raise ValueError(
                        f"{gen.__class__.__name__}.step() (spec {k}, module {j}) did not return a scalar "
                        f"number, but {str(new_val)} when called at timestep {self.cur_timestep} "
                        f"with cur_val={str(old_val)}"
                    )
                last = j == len(self._generators[k]) - 1
                if not last and self.keep_intermediate_series:
                    # key = f"{k}_{gen.info_str()}"
                    key = self.intermediate_map[k][j]
                    if key not in self.history:
                        self.history[key] = pd.Series(
                            [], dtype=np.float64, name=key, index=pd.DatetimeIndex([], tz="UTC")
                        )
                    self.history[key].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            if "limits" in spec:
                low, high = spec["limits"]
                new_val = float(np.clip(float(new_val), low, high))  # type: ignore[arg-type]
            if spec["semantics"] in ("action", "control"):
                # actions and control values are "input" and belong to the previous timestep
                # if last_timestep is not None: self.history[k][last_timestep] = new_val
                # self.history[k][self.cur_timestep] = np.nan
                self.history[k].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            else:
                self.history[k].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            if spec["semantics"] not in ("objective",) and (
                spec["semantics"] not in ("action",) or self.action_control_map[k] is None
            ):
                if action_is_dict:
                    sample_values_dict[k] = cast(float, new_val)
                else:
                    sample_values_list.append(cast(float, new_val))
            if spec["semantics"] == "target":
                goal_values[k] = cast(float, new_val)

        observation["observation"] = sample_values_dict if action_is_dict else np.array(sample_values_list)
        observation["achieved_goal"] = goal_values
        observation["desired_goal"] = self.desired_goal
        reward = self.compute_reward(achieved_goal=goal_values, desired_goal=self.desired_goal, info=info)

        terminated = self.step_count >= self.max_episode_steps if self.max_episode_steps is not None else False

        cur_ts = self.cur_timestep
        self.next_timestep = cur_ts + self.timestep_size
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment (currently a no-op placeholder)."""
        pass

    def compute_reward(
        self, achieved_goal: Mapping[str, float], desired_goal: Mapping[str, float], info: Mapping
    ) -> float:
        """Compute reward from achieved vs desired goals using per-target `vop_type`."""
        reward = 0.0
        for target, objective in self.target_obj_map.items():
            vop_type = self.process_spec[target]["vop_type"]
            if vop_type == "setpoint":
                reward -= np.abs(achieved_goal[target] - desired_goal[target])
            elif vop_type == "upper_limit":
                reward += np.minimum(desired_goal[target] - achieved_goal[target], 0.0)
            elif vop_type == "lower_limit":
                reward += np.minimum(achieved_goal[target] - desired_goal[target], 0.0)
            else:
                raise ValueError(f"unknown vop_type specified for channel {target}: {vop_type}")
        return reward

    def derive_desired_goal(self):
        """Derive a default goal (midpoint of limits or 0.0) for target channels."""
        return {
            k: spec["limits"][0] + 0.5 * (spec["limits"][1] - spec["limits"][0]) if has_limits(spec) else 0.0
            for k, spec in self.process_spec.items()
            if spec["semantics"] == "target"
        }

    def get_episode_data(self, include_temporary=False, include_intermediate=False) -> pd.DataFrame:
        """Return the current episode history as a DataFrame.

        Args:
            include_temporary: Include channels tagged as `temp`.
            include_intermediate: Include intermediate generator outputs.
        """
        intermediates = [
            intermediate for sens, intermediates in self.intermediate_map.items() for intermediate in intermediates
        ]
        res = {
            sens: (
                series
                if sens not in self.process_spec or self.process_spec[sens]["semantics"] not in ("control", "action")
                else series.shift(-1)
            )
            for sens, series in self.history.items()
            if (include_temporary or sens not in self.temp_channels)
            and (include_intermediate or sens not in intermediates)
        }
        return pd.DataFrame(res, index=pd.DatetimeIndex(self.index))


def has_limits(d):
    """Return True if `limits` exist and are finite at both ends."""
    return "limits" in d and (not np.isinf(d["limits"][0])) and (not np.isinf(d["limits"][1]))


def _apply_action_limits(
    process_spec: Mapping[str, Mapping[str, Any]], action_spec: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    merged = {k: dict(v) for k, v in process_spec.items()}
    for action_name, action_cfg in action_spec.items():
        if action_name in merged:
            if "limits" not in merged[action_name] and "limits" in action_cfg:
                merged[action_name]["limits"] = action_cfg["limits"]
            continue
        merged[action_name] = {
            "type": [],
            "semantics": "action",
        }
        if "limits" in action_cfg:
            merged[action_name]["limits"] = action_cfg["limits"]
    return merged


def _coerce_timestamp(value: Any) -> pd.Timestamp:
    """Convert a timestamp-like value to a UTC `pd.Timestamp` (naive values are interpreted as UTC)."""
    if not isinstance(value, (str, dt.datetime, dt.date, pd.Timestamp, np.datetime64)):
        raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def create_ccbench_env(**kwargs) -> CCBenchEnv:
    """Factory function to create real cartpole environment with specified config."""

    return CCBenchEnv(
        config_file=Path(__file__).resolve().parents[3]
        / "resources"
        / "envs"
        / "ccbench"
        / "config"
        / "ccbench_default.yaml",
        **kwargs,
    )
