"""
The MIT License (MIT)

Copyright 2017 Siemens AG

Authors: Alexander Hentschel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy import arccos, cbrt, cos, pi, sign, sin, sqrt


# ---------------------------------------------------------------------------
# Normalized Linearly-biased Goldstone Potential (NLGP)
# ---------------------------------------------------------------------------

class NormalizedGoldstonePotential:

    def __init__(self) -> None:
        u0 = cbrt(1 + sqrt(2)) / sqrt(3)
        r0 = u0 + 1 / (3 * u0)

        lmbd = 2 * r0**2 - r0**4 + 8 * sqrt(2 / 27.0) * r0

        self._norm_alpha = 2 / lmbd
        self._norm_beta = 1 / lmbd
        self._norm_kappa = -8 * sqrt(2 / 27.0) / lmbd
        self._qh_b = -sqrt(1 / 27.0)

    def polar_value(self, r: float, phi: float) -> float:
        rsq = np.square(r)
        return -self._norm_alpha * rsq + self._norm_beta * np.square(rsq) + self._norm_kappa * sin(phi) * r

    def global_minimum_radius(self, phi: float) -> float:
        f = np.vectorize(self._global_minimum_radius_scalar)
        return f(phi)

    def _global_minimum_radius_scalar(self, phi: float) -> float:
        qh = self._norm_kappa * abs(sin(phi)) / (8 * self._norm_beta)
        signum_phi = np.sign(sin(phi))
        if signum_phi == 0:
            signum_phi = 1

        if qh <= self._qh_b:
            u = cbrt(-signum_phi * qh + sqrt(qh * qh - 1 / 27))
            r0 = u + 1 / (3 * u)
        else:
            r0 = signum_phi * sqrt(4 / 3) * cos(1 / 3 * arccos(-qh * sqrt(27)))
        return r0


# ---------------------------------------------------------------------------
# Reward function for a fixed angle phi
# ---------------------------------------------------------------------------

class GoldstoneRewardFunction:

    def __init__(self, phi: float, max_required_step: float) -> None:
        if max_required_step <= 0:
            raise ValueError("Value for argument max_required_step must be positive")

        self.phi = phi
        self.max_required_step = max_required_step
        self._reward_fn = self._build_reward_function(phi, max_required_step)
        self.optimum_radius = self._compute_optimal_radius(phi, max_required_step)
        self.optimum_value = self._reward_fn(self.optimum_radius)

    def reward(self, r: float) -> float:
        return self._reward_fn(r)

    @staticmethod
    def _compute_optimal_radius(phi: float, max_required_step: float) -> float:
        phi = np.mod(phi, 2 * pi)
        opt = max(abs(sin(phi)), max_required_step)
        if phi >= pi:
            opt *= -1
        return opt

    @staticmethod
    def _build_radius_transform(opt_rad: float, min_rad: float):
        def transform(x: float) -> float:
            if abs(x) <= abs(opt_rad):
                return x * abs(min_rad) / abs(opt_rad)
            else:
                exponent = (2 - abs(opt_rad)) / (2 - abs(min_rad))
                scaling = (2 - abs(min_rad)) / (2 - abs(opt_rad)) ** exponent
                return np.sign(x) * (abs(min_rad) + scaling * (abs(x) - abs(opt_rad)) ** exponent)
        return transform

    def _build_reward_function(self, phi: float, max_required_step: float):
        nlgp = NormalizedGoldstonePotential()
        phi = np.mod(phi, 2 * pi)
        opt_rad = self._compute_optimal_radius(phi, max_required_step)
        min_rad = nlgp.global_minimum_radius(phi)
        tr = np.vectorize(self._build_radius_transform(opt_rad, min_rad))
        return lambda r: nlgp.polar_value(tr(r), phi)


# ---------------------------------------------------------------------------
# Goldstone dynamics
# ---------------------------------------------------------------------------

class Domain(Enum):
    negative = -1
    positive = +1


class SystemResponse(Enum):
    advantageous = +1
    disadvantageous = -1


class GoldstoneDynamics:

    def __init__(self, number_steps: int, max_required_step: float, safe_zone: float) -> None:
        if safe_zone < 0:
            raise ValueError("safe_zone must be non-negative")
        if (number_steps < 1) or (number_steps % 4 != 0):
            raise ValueError("number_steps must be positive and integer multiple of 4")

        self._safe_zone = safe_zone
        self._strongest_penalty_abs_idx = number_steps // 4
        self._penalty_functions = self._build_reward_functions(number_steps, max_required_step)

    def reset(self) -> tuple[Domain, int, SystemResponse]:
        return Domain.positive, 0, SystemResponse.advantageous

    def reward(self, phi_idx: int, position: float) -> float:
        return self._get_penalty_function(phi_idx).reward(position)

    def state_transition(
        self,
        domain: Domain,
        phi_idx: int,
        system_response: SystemResponse,
        position: float,
    ) -> tuple[Domain, int, SystemResponse]:
        old_domain = domain

        domain = self._compute_domain(old_domain, position)

        if domain != old_domain:
            system_response = SystemResponse.advantageous

        phi_idx += self._compute_angular_step(domain, phi_idx, system_response, position)

        system_response = self._updated_system_response(phi_idx, system_response)

        phi_idx = self._apply_symmetry(phi_idx)

        if (phi_idx == 0) and (abs(position) <= self._safe_zone):
            domain, phi_idx, system_response = self.reset()

        return domain, phi_idx, system_response

    def _compute_domain(self, domain: Domain, position: float) -> Domain:
        if abs(position) <= self._safe_zone:
            return domain
        return Domain(sign(position))

    def _compute_angular_step(
        self,
        domain: Domain,
        phi_idx: int,
        system_response: SystemResponse,
        position: float,
    ) -> int:
        if abs(position) <= self._safe_zone:
            return -sign(phi_idx)

        if phi_idx == -domain.value * self._strongest_penalty_abs_idx:
            return 0
        return system_response.value * sign(position)

    def _updated_system_response(self, phi_idx: int, system_response: SystemResponse) -> SystemResponse:
        if abs(phi_idx) >= self._strongest_penalty_abs_idx:
            return SystemResponse.disadvantageous
        return system_response

    def _apply_symmetry(self, phi_idx: int) -> int:
        if abs(phi_idx) < self._strongest_penalty_abs_idx:
            return phi_idx

        phi_idx = (phi_idx + (4 * self._strongest_penalty_abs_idx)) % (4 * self._strongest_penalty_abs_idx)
        phi_idx = 2 * self._strongest_penalty_abs_idx - phi_idx
        return phi_idx

    def _get_penalty_function(self, phi_idx: int) -> GoldstoneRewardFunction:
        idx = int(self._strongest_penalty_abs_idx + phi_idx)
        if idx < 0:
            idx = idx + len(self._penalty_functions)
        return self._penalty_functions[idx]

    def _build_reward_functions(self, number_steps: int, max_required_step: float) -> np.ndarray:
        k = self._strongest_penalty_abs_idx
        angle_grid = np.arange(-k, k + 1) * 2 * pi / number_steps
        return np.array([GoldstoneRewardFunction(phi, max_required_step) for phi in angle_grid])


# ---------------------------------------------------------------------------
# GoldstoneEnvironment – public interface used by IDS
# ---------------------------------------------------------------------------

class GoldstoneEnvironment:

    def __init__(self, number_steps: int, max_required_step: float, safe_zone: float) -> None:
        self._dynamics = GoldstoneDynamics(number_steps, max_required_step, safe_zone)

    def reward(self, phi_idx: int, position: float) -> float:
        return self._dynamics.reward(phi_idx, position)

    def state_transition(
        self,
        domain: Domain,
        phi_idx: int,
        system_response: SystemResponse,
        position: float,
    ) -> tuple[float, Domain, int, SystemResponse]:
        domain, phi_idx, system_response = self._dynamics.state_transition(
            domain, phi_idx, system_response, position,
        )
        return self.reward(phi_idx, position), domain, phi_idx, system_response