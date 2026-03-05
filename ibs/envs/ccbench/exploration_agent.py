"""
Implementation of an RL agent performing configurable random actions to explore the available action space.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import datetime as dt
from typing import Any
from typing import Mapping
from typing import Optional

import numpy as np
import pandas as pd

from ibs.envs.ccbench import generators
from ibs.common.sequences import is_sequence


class ExplorationAgent(object):
    """
    Simple non-trainable agent which executes a configured random policy to achieve exploration of (e.g.) an OpenAI gym
    environment for obtaining a meaningful offline training data set.
    """

    def __init__(self, action_spec: Mapping[str, Mapping[str, Any]], *args, **kwargs):
        """
        Initialize a new exploration agent, given an environment (potentially allowing access to its fields/methods),
        and a specification of the actions to generate.

        Args:
            action_spec: this dictionary should contain an entry for each action channel to generate actions for,
                mapping the channel name to the channel specification. Each channel specification needs to / can
                have these entries:
                  * type - required - name of a armor.data_generation.generators generator, or a list of such. Useful
                    generators for actions are e.g. `spike_action_gen`, `controls_exponential_action_scaler`,
                    `controls_decay`.
                  * semantics - required - should be `action` for the `action_spec`, but `temp` might also be useful,
                    if temporary series are needed for the actual action series generation.
                  * limits - optional - if the size of the actions should be clipped to some interval after all modules
                    specified in `type`have been evaluated
                Additionally, parameters for the individual generator modules can be included in this dict.
            *args:
            **kwargs:
        """
        self.action_spec = {k: dict(v) for k, v in action_spec.items()}
        self.cur_timestep = None

        self.action_channels = [k for k, spec in self.action_spec.items() if spec["semantics"] == "action"]

        self.index = []
        self.history = {}
        self.reset()

    def reset(self) -> None:
        """
        Reset the state of this agent to be ready for a new episode.

        This method has to be called before each new episode.
        """
        self.cur_timestep = None
        self.index = []
        self.history = {}

        # create individual channel generator instances
        for k, spec in self.action_spec.items():
            if not is_sequence(spec["type"]):
                spec["type"] = [spec["type"]]
            spec["generator"] = [generators.__dict__[typ](channel_name=k, spec=spec) for typ in spec["type"]]

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
                self.history[k] = pd.Series([], index=pd.DatetimeIndex([], tz="utc"), dtype=np.float32, name=k)
            self.history[k].loc[timestep] = v  # type: ignore[index]

        # iterate generation of current channel values for each channel
        for k, spec in self.action_spec.items():
            if k not in self.history:
                self.history[k] = pd.Series([], index=pd.DatetimeIndex([], tz="utc"), dtype=np.float32, name=k)
            if not is_sequence(spec["type"]):
                spec["type"] = [spec["type"]]
            cur_action = None
            # compute chain of channel generators for the current channel
            for j, gen in enumerate(spec["generator"]):
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
