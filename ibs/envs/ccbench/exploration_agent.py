"""
Implementation of an RL agent performing configurable random actions to explore the available action space.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import datetime as dt
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from ibs.envs.ccbench import generators
from ibs.common.sequences import is_sequence
from ibs.envs.ccbench.config_utils import resolve_callbacks


class ExplorationAgent(object):
    """
    Simple non-trainable agent which executes a configured random policy to achieve exploration of (e.g.) an OpenAI gym
    environment for obtaining a meaningful offline training data set.
    """

    def __init__(self, action_spec: Mapping[str, Mapping[str, Any]], *args, seed: Optional[int] = None, **kwargs):
        """
        Initialize a new exploration agent, given an environment (potentially allowing access to its fields/methods),
        and a specification of the actions to generate.

        Args:
            action_spec: this dictionary should contain an entry for each action channel to generate actions for,
                mapping the channel name to the channel specification. Each channel specification needs to / can
                have these entries:
                  * type - required - name of a `ibs.envs.ccbench.generators` generator, or a list of such. Useful
                    generators for actions are e.g. `spike_action_gen`, `controls_exponential_action_scaler`,
                    `controls_limiter`.
                  * semantics - required - should be `action` for the `action_spec`, but `temp` might also be useful,
                    if temporary series are needed for the actual action series generation.
                  * limits - optional - if the size of the actions should be clipped to some interval after all modules
                    specified in `type`have been evaluated
                Additionally, parameters for the individual generator modules can be included in this dict. Callback
                strings (`fc_callback`, `af_kernel`) are resolved via `ibs.envs.ccbench.config_utils.resolve_callbacks`.
            seed: seed for the agent's random number generator, which drives all random action generators. Use the
                `seed` argument of `reset` to re-seed the agent for individual episodes.
        """
        self.action_spec = resolve_callbacks(action_spec)
        for spec in self.action_spec.values():
            if not is_sequence(spec["type"]):
                spec["type"] = [spec["type"]]
        self.rng = np.random.default_rng(seed)
        self.cur_timestep = None
        self.generators: Dict[str, List[generators.ChannelGenerator]] = {}

        self.action_channels = [k for k, spec in self.action_spec.items() if spec["semantics"] == "action"]

        self.index = []
        self.history = {}
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the state of this agent to be ready for a new episode.

        This method has to be called before each new episode.

        Args:
            seed: if given, the agent's random number generator is re-seeded, making the following episode
                reproducible; otherwise the random stream of the previous episode is continued.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.cur_timestep = None
        self.index = []
        self.history = {}

        # create individual channel generator instances
        self.generators = {
            k: [generators.__dict__[typ](channel_name=k, spec=spec, rng=self.rng) for typ in spec["type"]]
            for k, spec in self.action_spec.items()
        }

    def act(
        self,
        observation: Mapping[str, float],
        desired_goal: Optional[Mapping[str, float]] = None,
        timestep: Optional[dt.datetime] = None,
    ) -> Mapping[str, float]:
        """
        Compute actions suggested by the agent's policy, given the current observations/state and desired goals.

        Args:
            observation: the state/observation as returned from the environment (for `gym.GoalEnv` this should be the
                `observation` value from the returned observation).
            desired_goal: in general, policies should try to reach the values specified in `desired_goal` for the
                target states. This exploration agent ignores the specified goal and chooses a random action.
            timestep: the current timestep; this might be relevant in case hte policy is time-dependent and needs to
                know how much time has passed since the last action.

        Returns:
            A dict containing the suggested actions.
        """
        actions = {}
        self.cur_timestep = timestep
        self.index.append(timestep)
        for k, v in observation.items():
            if k not in self.history:
                self.history[k] = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=np.float64, name=k)
            self.history[k].loc[timestep] = v  # type: ignore[index]

        # iterate generation of current channel values for each channel
        for k, spec in self.action_spec.items():
            if k not in self.history:
                self.history[k] = pd.Series([], index=pd.DatetimeIndex([], tz="UTC"), dtype=np.float64, name=k)
            cur_action = None
            # compute chain of channel generators for the current channel
            for gen in self.generators[k]:
                cur_action = gen.step(timestep=timestep, history=self.history, cur_val=cur_action)
            if "limits" in spec:
                cur_action = np.clip(cur_action, spec["limits"][0], spec["limits"][1])  # type: ignore[arg-type]
            self.history[k].loc[timestep] = cur_action  # type: ignore[index]
            actions[k] = cur_action

        return actions

    def observe(self, reward: float) -> None:
        """
        Calling this method provides feedback on the reward obtained for the last provided actions (from `act`).

        This method should be called after each `act`, when the reward is available.

        Args:
            reward: the reward achieved by executing the actions suggested by the last `act` call.

        Returns:

        """
        pass
