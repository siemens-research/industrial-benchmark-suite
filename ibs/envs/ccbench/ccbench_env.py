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
from typing import Sequence
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
from .config_utils import channel_metadata
from .config_utils import parse_freq
from .config_utils import resolve_callbacks

DEFAULT_START_TIMESTAMP = "2000-01-01T00:00:00Z"
# maximum number of hidden warm-up steps for `warmup_steps="auto"`
MAX_AUTO_WARMUP_STEPS = 64


class CCBenchEnv(GoalEnv):
    """Goal-based synthetic time-series environment driven by a channel specification.

    All random numbers are drawn from the environment's `np_random` generator, so `reset(seed=...)` makes the
    following episode reproducible. A seeded reset starts the episode at the configured `start_timestamp`, an unseeded
    reset continues the random stream and starts the episode `episode_step` after the end of the previous one.

    Each reset performs `warmup_steps` hidden steps with zero actions (until all observations are finite for "auto"),
    followed by one step with zero actions providing the initial observation. None of these is counted in
    `step_count`, and the hidden warm-up steps are excluded from `get_episode_data()` by default. Reaching
    `episode_length` agent steps is reported as `truncated` (or as `terminated` if `truncate_on_time_limit` is False).
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config_file: Optional[Path | str] = None,
        config_dict: Optional[Mapping[str, Any]] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the environment from a channel specification.

        Args:
            config_file: Optional YAML file providing configuration values.
            config_dict: Optional mapping of configuration values.
            render_mode: Must be None, the environment does not support rendering.
        """
        super().__init__()
        if render_mode is not None:
            raise ValueError(f"Unsupported render_mode {render_mode!r}, CCBenchEnv does not support rendering")
        self.render_mode = render_mode
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
        self.truncate_on_time_limit = bool(config.get("truncate_on_time_limit", True))
        self.warmup_steps: Union[int, str] = config.get("warmup_steps", "auto")
        self.expose_temp_channels = bool(config.get("expose_temp_channels", False))

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
        self.timestep_size = parse_freq(freq)
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

        self.render_file = render_file
        self.reward_range = (-float("inf"), 0.0)

        def get_box_space(channel):
            low, high = self.process_spec.get(channel, {}).get("limits", [-np.inf, np.inf])
            return gym.spaces.Box(low=np.array(float(low)), high=np.array(float(high)), shape=None, dtype=np.float64)

        # channels returned in observation["observation"]
        self.observation_channels = [
            k
            for k, spec in self.process_spec.items()
            if spec["semantics"] != "objective"
            and (spec["semantics"] != "action" or self.action_control_map[k] is None)
            and (spec["semantics"] != "temp" or self.expose_temp_channels)
        ]
        state_space: Dict[str, gym.spaces.Space] = {k: get_box_space(k) for k in self.observation_channels}
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
        self._explicit_goal: Optional[Dict[str, float]] = None
        self._warmup_rows = 0
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
        """Reset the environment state and optionally perform the warm-up and initial steps.

        Args:
            seed: RNG seed for the environment's `np_random` generator, which drives all random channel generators.
                With a seed, the episode starts at `start_timestamp` (unless specified otherwise) and is reproducible.
            options: Optional reset options. Supported key: `episode_start` (timestamp, same as the `episode_start`
                argument, taking precedence over it).
            episode_start: Explicit start timestamp. If omitted, seeded resets and the first episode start at
                `start_timestamp`, unseeded resets start `episode_step` after the last timestep of the previous
                episode. Naive timestamps are interpreted as UTC.
            perform_step: If True, performs the hidden warm-up steps and one step with zero actions to return a valid
                initial observation.
            replay: Optional DataFrame indexed by timestamps to replay channels.
                If provided, `episode_start` is derived from its index.

        Returns:
            Initial observation and info dict with `step`, `timestamp`, `warmup_steps` and `channels` (see
            `ibs.envs.ccbench.config_utils.channel_metadata`). Both are empty if `perform_step` is False.
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
        self._explicit_goal = None
        self._warmup_rows = 0

        # setup empty history and new generator instances for all channels
        self.history = {
            k: pd.Series([], dtype=np.float64, name=k, index=pd.DatetimeIndex([], tz="UTC")) for k in self.process_spec
        }
        self._generators = self._build_generators(self.np_random, replay)

        if not perform_step:
            return {}, {}
        zero_action = {k: 0.0 for k in self.action_control_map}
        while True:
            values, goal_values = self._advance(zero_action)
            observation = self._make_observation(values, goal_values)
            if self.warmup_steps == "auto":
                if _is_finite_observation(observation):
                    break
                if self._warmup_rows >= MAX_AUTO_WARMUP_STEPS:
                    raise RuntimeError(
                        f"Observation is not finite after {MAX_AUTO_WARMUP_STEPS} warm-up steps, "
                        "configure warmup_steps explicitly"
                    )
            elif self._warmup_rows >= int(self.warmup_steps):
                break
            self._warmup_rows += 1
        info = {
            "step": self.step_count,
            "timestamp": self.cur_timestep,
            "warmup_steps": self._warmup_rows,
            "channels": self.channel_metadata,
        }
        return observation, info

    @property
    def channel_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Semantics, limits, unit and step duration of all exposed channels, including the actions."""
        return channel_metadata(self.process_spec, freq=self.timestep_size, include_temp=self.expose_temp_channels)

    def step(
        self,
        action: Union[Mapping[str, Any], np.ndarray, Sequence[float]],
        desired_goal: Optional[Mapping[str, float]] = None,
    ) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one timestep using the provided action.

        Args:
            action: Either a mapping of action channels to (scalar or shape () array) values, or a numpy array
                aligned with `action_control_map` order.
            desired_goal: Optional goal for (some of) the target channels, kept for the remaining episode. If omitted,
                goals are replayed or derived.

        Returns:
            Observation (dict with float values), reward, terminated, truncated and info dict with `step`, `timestamp`
            and `reward_terms` (reward contribution per target channel).
        """
        if self.next_timestep is None:
            raise RuntimeError("reset() must be called before step()")
        if desired_goal is not None:
            unknown = sorted(set(desired_goal) - set(self.target_channels))
            if unknown:
                raise ValueError(f"desired_goal contains unknown target channels: {unknown}")
            self._explicit_goal = {**(self._explicit_goal or {}), **{k: float(v) for k, v in desired_goal.items()}}
        self.step_count += 1
        values, goal_values = self._advance(self._action_dict(action))
        observation = self._make_observation(values, goal_values)
        reward_terms = self.compute_reward_terms(observation["achieved_goal"], observation["desired_goal"])
        info: Dict[str, Any] = {
            "step": self.step_count,
            "timestamp": self.cur_timestep,
            "reward_terms": reward_terms,
        }
        reward = self.compute_reward(observation["achieved_goal"], observation["desired_goal"], info)
        terminated = self.compute_terminated(observation["achieved_goal"], observation["desired_goal"], info)
        truncated = self.compute_truncated(observation["achieved_goal"], observation["desired_goal"], info)
        return observation, reward, terminated, truncated, info

    def _action_dict(self, action: Union[Mapping[str, Any], np.ndarray, Sequence[float]]) -> Dict[str, float]:
        """Convert an action (mapping or array in `action_control_map` order) to a dict of floats."""
        if isinstance(action, Mapping):
            missing = [k for k in self.action_control_map if k not in action]
            if missing:
                raise ValueError(f"action is missing the channels {missing}")
            return {k: float(np.asarray(action[k]).item()) for k in self.action_control_map}
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        if values.shape[0] != len(self.action_control_map):
            raise ValueError(f"action must have {len(self.action_control_map)} entries, got {values.shape[0]}")
        return {k: float(v) for k, v in zip(self.action_control_map, values)}

    def _current_goal(self, cur_ts: pd.Timestamp) -> Dict[str, float]:
        """Goal in effect at `cur_ts`: replayed, else explicitly set via `step()`, else derived."""
        if self.replay is not None and cur_ts in self.replay.index:
            if all(o in self.replay.columns for o in self.target_obj_map.values()):
                return {t: float(self.replay.at[cur_ts, o]) for t, o in self.target_obj_map.items()}  # type: ignore
        goal = {k: float(v) for k, v in self.derive_desired_goal().items()}
        if self._explicit_goal is not None:
            goal.update(self._explicit_goal)
        return goal

    def _advance(self, action: Mapping[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute all channel values for the next timestep without counting it as agent step.

        Args:
            action: Mapping of all action channels to float values.

        Returns:
            Values of all non-objective channels (including hidden temp channels) and of the target channels.
        """
        self.cur_timestep = self.next_timestep
        if self.cur_timestep is None:
            raise RuntimeError("cur_timestep is None at step entry")
        cur_ts = self.cur_timestep
        self.index.append(cur_ts)
        self.desired_goal = self._current_goal(cur_ts)

        values: Dict[str, float] = {}
        goal_values: Dict[str, float] = {}
        # iterate generation of current channel values for each channel
        for k, spec in self.process_spec.items():
            new_val = None
            if spec["semantics"] == "action":
                new_val = action[k]
            if spec["semantics"] == "control":
                new_val = action[spec["assoc_action_channel"]]
            if spec["semantics"] == "objective":
                new_val = self.desired_goal[spec["assoc_target_channel"]]
            # compute chain of channel generators for the current channel
            for j, gen in enumerate(self._generators[k]):
                old_val = new_val
                new_val = gen.step(timestep=cur_ts, history=self.history, cur_val=new_val)
                if not isinstance(new_val, (float, int)):
                    raise ValueError(
                        f"{gen.__class__.__name__}.step() (spec {k}, module {j}) did not return a scalar "
                        f"number, but {str(new_val)} when called at timestep {cur_ts} "
                        f"with cur_val={str(old_val)}"
                    )
                last = j == len(self._generators[k]) - 1
                if not last and self.keep_intermediate_series:
                    key = self.intermediate_map[k][j]
                    if key not in self.history:
                        self.history[key] = pd.Series(
                            [], dtype=np.float64, name=key, index=pd.DatetimeIndex([], tz="UTC")
                        )
                    self.history[key].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            new_val = float(new_val)  # type: ignore[arg-type]
            if "limits" in spec:
                low, high = spec["limits"]
                new_val = float(np.clip(new_val, low, high))
            # actions and control values are "input" and belong to the previous timestep, see get_episode_data()
            self.history[k].loc[cur_ts] = new_val  # type: ignore[index]
            if spec["semantics"] != "objective":
                values[k] = new_val
            if spec["semantics"] == "target":
                goal_values[k] = new_val

        self.next_timestep = cur_ts + self.timestep_size
        return values, goal_values

    def _make_observation(self, values: Mapping[str, float], goal_values: Mapping[str, float]) -> Dict[str, Any]:
        """Assemble the goal-env observation dict from computed channel values."""
        return {
            "observation": {k: values[k] for k in self.observation_channels},
            "achieved_goal": dict(goal_values),
            "desired_goal": dict(self.desired_goal or {}),
        }

    def render(self):
        """Render the environment (not supported, `render_mode` is always None)."""
        return None

    def compute_reward_terms(
        self, achieved_goal: Mapping[str, float], desired_goal: Mapping[str, float]
    ) -> Dict[str, float]:
        """Compute the reward contribution per target channel according to its `vop_type`.

        Args:
            achieved_goal: Current values of the target channels.
            desired_goal: Goal values of the target channels.

        Returns:
            Mapping of target channel to its (non-positive) reward term.
        """
        terms: Dict[str, float] = {}
        for target in self.target_obj_map:
            vop_type = self.process_spec[target]["vop_type"]
            if vop_type == "setpoint":
                terms[target] = -float(np.abs(achieved_goal[target] - desired_goal[target]))
            elif vop_type == "upper_limit":
                terms[target] = float(np.minimum(desired_goal[target] - achieved_goal[target], 0.0))
            elif vop_type == "lower_limit":
                terms[target] = float(np.minimum(achieved_goal[target] - desired_goal[target], 0.0))
            else:
                raise ValueError(f"unknown vop_type specified for channel {target}: {vop_type}")
        return terms

    def compute_reward(
        self, achieved_goal: Mapping[str, float], desired_goal: Mapping[str, float], info: Mapping
    ) -> float:
        """Compute reward from achieved vs desired goals as sum of `compute_reward_terms`."""
        reward = 0.0
        for term in self.compute_reward_terms(achieved_goal, desired_goal).values():
            reward += term
        return reward

    def _time_limit_reached(self, info: Mapping[str, Any]) -> bool:
        step = info.get("step", self.step_count)
        return self.max_episode_steps is not None and step >= self.max_episode_steps

    def compute_terminated(
        self, achieved_goal: Mapping[str, float], desired_goal: Mapping[str, float], info: Mapping
    ) -> bool:
        """The process has no terminal states, only the legacy time limit (`truncate_on_time_limit=False`)."""
        return not self.truncate_on_time_limit and self._time_limit_reached(info)

    def compute_truncated(
        self, achieved_goal: Mapping[str, float], desired_goal: Mapping[str, float], info: Mapping
    ) -> bool:
        """Truncate after `episode_length` agent steps (if `truncate_on_time_limit`)."""
        return self.truncate_on_time_limit and self._time_limit_reached(info)

    def derive_desired_goal(self):
        """Derive a default goal (midpoint of limits or 0.0) for target channels."""
        return {
            k: spec["limits"][0] + 0.5 * (spec["limits"][1] - spec["limits"][0]) if has_limits(spec) else 0.0
            for k, spec in self.process_spec.items()
            if spec["semantics"] == "target"
        }

    def get_episode_data(
        self, include_temporary=False, include_intermediate=False, include_warmup=False
    ) -> pd.DataFrame:
        """Return the current episode history as a DataFrame.

        Action and control values are shifted such that each row contains the values applied after its observation.

        Args:
            include_temporary: Include channels tagged as `temp`.
            include_intermediate: Include intermediate generator outputs.
            include_warmup: Include the rows of the hidden warm-up steps performed by `reset()`.
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
        data = pd.DataFrame(res, index=pd.DatetimeIndex(self.index))
        return data if include_warmup else data.iloc[self._warmup_rows :]


def has_limits(d):
    """Return True if `limits` exist and are finite at both ends."""
    return "limits" in d and (not np.isinf(d["limits"][0])) and (not np.isinf(d["limits"][1]))


def _is_finite_observation(observation: Mapping[str, Mapping[str, float]]) -> bool:
    """Return True if all values of a goal-env observation are finite."""
    return all(np.isfinite(v) for part in observation.values() for v in part.values())


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
    """Factory function to create the CCBench environment from the default config file."""

    return CCBenchEnv(
        config_file=Path(__file__).resolve().parents[3]
        / "resources"
        / "envs"
        / "ccbench"
        / "config"
        / "ccbench_default.yaml",
        **kwargs,
    )
