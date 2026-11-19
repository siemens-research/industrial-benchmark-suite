"""
(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT

Authors:
    Stefan Depeweg
    Phillip Swazinna
    Ludwig Winkler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from ibs.envs.industrial_benchmark.goldstone import Domain, GoldstoneEnvironment, SystemResponse


# ---------------------------------------------------------------------------
# EffectiveAction
# ---------------------------------------------------------------------------


class EffectiveAction:
    def __init__(self, velocity: float, gain: float, setpoint: float) -> None:
        self.setpoint = setpoint
        self.effective_velocity = self._calc_effective_velocity(velocity, gain, setpoint)
        self.effective_gain = self._calc_effective_gain(gain, setpoint)

    def _calc_effective_velocity(self, a: float, b: float, setpoint: float) -> float:
        min_unscaled = self._calc_velocity_unscaled(self._effective_a(100, setpoint), self._effective_b(0, setpoint))
        max_unscaled = self._calc_velocity_unscaled(self._effective_a(0, setpoint), self._effective_b(100, setpoint))
        unscaled = self._calc_velocity_unscaled(self._effective_a(a, setpoint), self._effective_b(b, setpoint))
        return (unscaled - min_unscaled) / (max_unscaled - min_unscaled)

    def _calc_effective_gain(self, b: float, setpoint: float) -> float:
        min_unscaled = self._calc_gain_unscaled(self._effective_b(100, setpoint))
        max_unscaled = self._calc_gain_unscaled(self._effective_b(0, setpoint))
        unscaled = self._calc_gain_unscaled(self._effective_b(b, setpoint))
        return (unscaled - min_unscaled) / (max_unscaled - min_unscaled)

    @staticmethod
    def _effective_a(a: float, setpoint: float) -> float:
        return a + 101.0 - setpoint

    @staticmethod
    def _effective_b(b: float, setpoint: float) -> float:
        return b + 1.0 + setpoint

    @staticmethod
    def _calc_velocity_unscaled(effective_a: float, effective_b: float) -> float:
        return (effective_b + 1.0) / effective_a

    @staticmethod
    def _calc_gain_unscaled(effective_b: float) -> float:
        return 1.0 / effective_b

    def getEffectiveVelocity(self) -> float:
        return self.effective_velocity

    def getEffectiveGain(self) -> float:
        return self.effective_gain


# ---------------------------------------------------------------------------
# BenchmarkState
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkState:
    """Typed state representation for the industrial benchmark.

    Replaces the former ``OrderedDict`` with cryptic single-letter keys
    (e.g. ``'o'``, ``'coc'``, ``'fb'``) with clearly named, typed fields.
    """

    # Internal buffers & costs
    operational_cost_buffer: NDArray[np.float64] = field(default_factory=lambda: np.zeros(10))
    current_operational_cost: float = 0.0
    fatigue_base: float = 0.0
    operational_cost_conv: float = 0.0
    hidden_gain: float = 0.0
    hidden_velocity: float = 0.0
    hidden_shift: float = 0.0

    # Goldstone variables
    gs_domain: int = 0
    gs_sys_response: int = 0
    gs_phi_idx: int = 0
    effective_gain: float = 0.0
    effective_velocity: float = 0.0
    miscalibration: float = 0.0

    # Observables
    setpoint: float = 50.0
    velocity: float = 50.0
    gain: float = 50.0
    shift: float = 50.0
    fatigue: float = 0.0
    consumption: float = 0.0
    cost: float = 0.0
    reward: float = 0.0

    # Ordered list of observable keys for external access
    OBSERVABLE_KEYS: list[str] = field(
        default_factory=lambda: [
            "setpoint",
            "velocity",
            "gain",
            "shift",
            "fatigue",
            "consumption",
            "cost",
            "reward",
        ],
        repr=False,
    )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-style assignment for backward compatibility."""
        setattr(self, key, value)

    def observable_array(self) -> NDArray[np.float64]:
        """Return observable state as a flat numpy array."""
        return np.concatenate([np.array(getattr(self, k)).ravel() for k in self.OBSERVABLE_KEYS])

    def full_array(self) -> NDArray[np.float64]:
        """Return the full Markov state as a flat numpy array."""
        all_keys = [
            "operational_cost_buffer",
            "current_operational_cost",
            "fatigue_base",
            "operational_cost_conv",
            "hidden_gain",
            "hidden_velocity",
            "hidden_shift",
            "gs_domain",
            "gs_sys_response",
            "gs_phi_idx",
            "effective_gain",
            "effective_velocity",
            "miscalibration",
            "setpoint",
            "velocity",
            "gain",
            "shift",
            "fatigue",
            "consumption",
            "cost",
            "reward",
        ]
        return np.concatenate([np.array(getattr(self, k)).ravel() for k in all_keys])


# ---------------------------------------------------------------------------
# IDS – core industrial dynamics system
# ---------------------------------------------------------------------------


class IDS:
    """Lightweight Python implementation of the industrial benchmark.

    Uses the same standard settings as in
    ``src/main/resources/simTest.properties`` of the Java implementation.
    """

    def __init__(
        self,
        p: float = 50,
        stationary_p: bool = True,
        initial_seed: int | None = None,
    ) -> None:
        np.random.seed(initial_seed)

        self.max_required_step: float = np.sin(15.0 / 180.0 * np.pi)
        self.gs_bound: float = 1.5
        self.gs_setpoint_dependency: float = 0.02
        self.gs_scale: float = 2.0 * self.gs_bound + 100.0 * self.gs_setpoint_dependency

        self.CRF: float = 3.0
        self.CRC: float = 1.0
        self.CRGS: float = 25.0

        self.stationary_p = stationary_p
        self.gs_environment = GoldstoneEnvironment(
            24,
            self.max_required_step,
            self.max_required_step / 2.0,
        )

        self.state = BenchmarkState()
        self.state.setpoint = p
        self.state.gs_domain = Domain.positive.value
        self.state.gs_sys_response = SystemResponse.advantageous.value

        self._init = True
        self._define_new_sequence()
        self.step(np.zeros(3))

    def visibleState(self) -> NDArray[np.float64]:
        return self.state.observable_array()

    def markovState(self) -> NDArray[np.float64]:
        return self.state.full_array()

    def step(self, delta: NDArray[np.float64]) -> None:
        self._update_setpoint()
        self._add_action(delta)
        self._update_fatigue()
        self._update_current_operational_cost()
        self._update_operational_cost_convolution()
        self._update_gs()
        self._update_operational_costs()
        self._update_cost()

    def _update_setpoint(self) -> None:
        if self.stationary_p:
            return

        if self._p_step == self._p_steps:
            self._define_new_sequence()

        new_p = self.state.setpoint + self._p_ch
        if new_p > 100 or new_p < 0:
            if np.random.rand() > 0.5:
                self._p_ch *= -1

        self.state.setpoint = np.clip(new_p, 0, 100)
        self._p_step += 1

    def _add_action(self, delta: NDArray[np.float64]) -> None:
        delta = np.clip(delta, -1, 1)
        self.state.velocity = np.clip(self.state.velocity + delta[0], 0.0, 100.0)
        self.state.gain = np.clip(self.state.gain + 10 * delta[1], 0.0, 100.0)
        self.state.shift = np.clip(
            self.state.shift + ((self.max_required_step / 0.9) * 100.0 / self.gs_scale) * delta[2],
            0.0,
            100.0,
        )
        self.state.hidden_shift = np.clip(
            self.gs_scale * self.state.shift / 100.0
            - self.gs_setpoint_dependency * self.state.setpoint
            - self.gs_bound,
            -self.gs_bound,
            self.gs_bound,
        )

    def _update_fatigue(self) -> None:
        exp_lambda = 0.1
        action_tolerance = 0.05
        fatigue_amplification = 1.1
        fatigue_amplification_max = 5.0
        fatigue_amplification_start = 1.2

        velocity = self.state.velocity
        gain = self.state.gain
        setpoint = self.state.setpoint

        hidden_gain = self.state.hidden_gain
        hidden_velocity = self.state.hidden_velocity

        eff_act = EffectiveAction(velocity, gain, setpoint)
        eff_act_velocity = eff_act.getEffectiveVelocity()
        eff_act_gain = eff_act.getEffectiveGain()

        self.state.effective_gain = eff_act_gain
        self.state.effective_velocity = eff_act_velocity

        noise_e_g = np.random.exponential(exp_lambda)
        noise_e_v = np.random.exponential(exp_lambda)
        noise_u_g = np.random.rand()
        noise_u_v = np.random.rand()

        noise_b_g = float(np.random.binomial(1, np.clip(eff_act_gain, 0.001, 0.999)))
        noise_b_v = float(np.random.binomial(1, np.clip(eff_act_velocity, 0.001, 0.999)))

        noise_gain = 2.0 * (1.0 / (1.0 + np.exp(-noise_e_g)) - 0.5)
        noise_velocity = 2.0 * (1.0 / (1.0 + np.exp(-noise_e_v)) - 0.5)

        noise_gain += (1 - noise_gain) * noise_u_g * noise_b_g * eff_act_gain
        noise_velocity += (1 - noise_velocity) * noise_u_v * noise_b_v * eff_act_velocity

        if eff_act_gain <= action_tolerance:
            hidden_gain = eff_act_gain
        elif hidden_gain >= fatigue_amplification_start:
            hidden_gain = np.minimum(fatigue_amplification_max, fatigue_amplification * hidden_gain)
        else:
            hidden_gain = 0.9 * hidden_gain + noise_gain / 3.0

        if eff_act_velocity <= action_tolerance:
            hidden_velocity = eff_act_velocity
        elif hidden_velocity >= fatigue_amplification_start:
            hidden_velocity = np.minimum(fatigue_amplification_max, fatigue_amplification * hidden_velocity)
        else:
            hidden_velocity = 0.9 * hidden_velocity + noise_velocity / 3.0

        if np.maximum(hidden_velocity, hidden_gain) == fatigue_amplification_max:
            alpha = 1.0 / (1.0 + np.exp(-np.random.normal(2.4, 0.4)))
        else:
            alpha = np.maximum(noise_velocity, noise_gain)

        fb = np.maximum(0, ((30000.0 / ((5 * velocity) + 100)) - 0.01 * (gain**2)))
        self.state.hidden_velocity = hidden_velocity
        self.state.hidden_gain = hidden_gain
        self.state.fatigue = (fb * (1 + 2 * alpha)) / 3.0
        self.state.fatigue_base = fb

    def _update_current_operational_cost(self) -> None:
        cost_setpoint = 2.0
        cost_velocity = 4.0
        cost_gain = 2.5

        gain = self.state.gain
        velocity = self.state.velocity
        setpoint = self.state.setpoint

        costs = cost_setpoint * setpoint + cost_gain * gain + cost_velocity * velocity
        o = np.exp(costs / 100.0)
        self.state.current_operational_cost = o

        if self._init:
            self.state.operational_cost_buffer += o
            self._init = False
        else:
            self.state.operational_cost_buffer[:-1] = self.state.operational_cost_buffer[1:]
            self.state.operational_cost_buffer[-1] = o

    def _update_operational_cost_convolution(self) -> None:
        conv_array = np.array([0.11111, 0.22222, 0.33333, 0.22222, 0.11111, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.state.operational_cost_conv = np.dot(self.state.operational_cost_buffer, conv_array)

    def _update_gs(self) -> None:
        effective_shift = self.state.hidden_shift

        domain = self.state.gs_domain
        phi_idx = self.state.gs_phi_idx
        system_response = self.state.gs_sys_response

        reward, domain, phi_idx, system_response = self.gs_environment.state_transition(
            Domain(domain),
            phi_idx,
            SystemResponse(system_response),
            effective_shift,
        )
        self.state.miscalibration = -reward
        self.state.gs_domain = domain.value
        self.state.gs_sys_response = system_response.value
        self.state.gs_phi_idx = phi_idx

    def _update_operational_costs(self) -> None:
        rGS = self.state.miscalibration
        e_new_hidden = self.state.operational_cost_conv - (self.CRGS * (rGS - 1.0))
        operational_costs = e_new_hidden - np.random.randn() * (1 + 0.005 * e_new_hidden)
        self.state.consumption = operational_costs

    def _update_cost(self) -> None:
        fatigue = self.state.fatigue
        consumption = self.state.consumption
        cost = self.CRF * fatigue + self.CRC * consumption

        self.state.cost = cost
        self.state.reward = -cost

    def _define_new_sequence(self) -> None:
        length = np.random.randint(1, 100)
        self._p_steps = length
        self._p_step = 0
        p_ch = 2 * np.random.rand() - 1
        if np.random.rand() < 0.1:
            p_ch *= 0.0
        self._p_ch = p_ch


# ---------------------------------------------------------------------------
# IBGym – Gymnasium wrapper
# ---------------------------------------------------------------------------


class IBGym(gym.Env):
    """Gymnasium wrapper for the industrial benchmark."""

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        setpoint: float,
        reward_type: str = "classic",
        action_type: str = "continuous",
        observation_type: str = "classic",
        reset_after_timesteps: int = 1000,
        init_seed: int | None = None,
        n_past_timesteps: int = 30,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode

        self.setpoint = setpoint
        self.init_seed = init_seed
        np.random.seed(self.init_seed)

        self.reward_function = reward_type
        self.reset_after_timesteps = reset_after_timesteps
        self.action_type = action_type
        self.observation_type = observation_type
        self.n_past_timesteps = n_past_timesteps

        self.IB: IDS | None = None
        self.info: dict[str, Any] = {}
        self.reward: float = 0.0
        self.delta_reward: float = 0.0
        self.smoothed_reward: float = 0.0
        self.env_steps: int = 0
        self.truncated: bool = False
        self.last_action: Any = None
        self.observation: Any = None

        if self.action_type == "discrete":
            self.action_space = gym.spaces.Discrete(27)
            self.env_action: list[list[int]] = [[v, g, s] for v in [-1, 0, 1] for g in [-1, 0, 1] for s in [-1, 0, 1]]
        elif self.action_type == "continuous":
            self.action_space = gym.spaces.Box(
                np.array([-1, -1, -1], dtype=np.float32),
                np.array([+1, +1, +1], dtype=np.float32),
            )
        else:
            raise ValueError('action_type must be "discrete" or "continuous"')

        single_low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        single_high = np.array([100, 100, 100, 100, 1000, 1000], dtype=np.float32)

        if self.observation_type == "classic":
            self.observation_space = gym.spaces.Box(low=single_low, high=single_high)
        elif self.observation_type == "include_past":
            low = np.tile(single_low, self.n_past_timesteps)
            high = np.tile(single_high, self.n_past_timesteps)
            self.observation_space = gym.spaces.Box(low=low, high=high)
        else:
            raise ValueError('observation_type must be "classic" or "include_past"')

        self.reset()

    def step(
        self,
        action: Any,
    ) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        if self.truncated:
            self.reset()

        self.last_action = action

        if self.action_type == "discrete":
            self.IB.step(self.env_action[action])
        elif self.action_type == "continuous":
            self.IB.step(action)

        return_observation = self._update_observation()

        new_reward = -self.IB.state.cost
        self.delta_reward = new_reward - self.reward
        self.reward = new_reward
        self.smoothed_reward = 0.9 * self.smoothed_reward + 0.1 * self.reward

        self.env_steps += 1
        if self.env_steps >= self.reset_after_timesteps:
            self.truncated = True

        if self.reward_function == "classic":
            return_reward = self.reward
        elif self.reward_function == "delta":
            return_reward = self.delta_reward
        else:
            raise ValueError('reward_function must be "classic" or "delta"')

        self.info = self._markovian_state()
        return return_observation, return_reward, False, self.truncated, self.info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.IB = IDS(self.setpoint, initial_seed=self.init_seed)
        self.init_seed = np.random.randint(0, 100000)

        if self.observation_type == "include_past":
            self.observation = []

        return_observation = self._update_observation()

        self.info = self._markovian_state()
        self.reward = -self.IB.state.cost
        self.delta_reward = 0.0
        self.smoothed_reward = self.reward
        self.env_steps = 0
        self.truncated = False

        return return_observation, self.info

    def render(self) -> None:
        if self.render_mode == "human":
            print(
                "Reward:",
                self.reward,
                "State (v,g,s):",
                self.IB.visibleState()[1:4],
                "Action:",
                self.last_action,
            )

    def _update_observation(self) -> NDArray[np.float64]:
        if self.observation_type == "classic":
            self.observation = self.IB.visibleState()[:-2]
            return self.observation

        if self.observation_type == "include_past":
            single_state = self.IB.visibleState()[:-2]

            if len(self.observation) > 0:
                self.observation.pop()

            self.observation.insert(0, single_state)

            if len(self.observation) < self.n_past_timesteps:
                oldest_obs = self.observation[-1]
                oldest_repeated = [oldest_obs] * (self.n_past_timesteps - len(self.observation))
                self.observation = self.observation + oldest_repeated

            return np.hstack(self.observation)

        raise ValueError('observation_type must be "classic" or "include_past"')

    def _markovian_state(self) -> dict[str, float]:
        s = self.IB.state
        keys = [
            "setpoint",
            "velocity",
            "gain",
            "shift",
            "fatigue",
            "consumption",
            "op_cost_t0",
            "op_cost_t1",
            "op_cost_t2",
            "op_cost_t3",
            "op_cost_t4",
            "op_cost_t5",
            "op_cost_t6",
            "op_cost_t7",
            "op_cost_t8",
            "op_cost_t9",
            "ml1",
            "ml2",
            "ml3",
            "hv",
            "hg",
        ]
        values = [
            s.setpoint,
            s.velocity,
            s.gain,
            s.shift,
            s.fatigue,
            s.consumption,
            s.operational_cost_buffer[0],
            s.operational_cost_buffer[1],
            s.operational_cost_buffer[2],
            s.operational_cost_buffer[3],
            s.operational_cost_buffer[4],
            s.operational_cost_buffer[5],
            s.operational_cost_buffer[6],
            s.operational_cost_buffer[7],
            s.operational_cost_buffer[8],
            s.operational_cost_buffer[9],
            s.gs_domain,
            s.gs_sys_response,
            s.gs_phi_idx,
            s.hidden_velocity,
            s.hidden_gain,
        ]
        return dict(zip(keys, values))
