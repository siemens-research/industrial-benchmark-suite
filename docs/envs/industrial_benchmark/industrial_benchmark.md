# Industrial Benchmark

## Gymnasium Environment for the Industrial Benchmark (IB)

The Industrial Benchmark (IB) is a lightweight, fully analytical reinforcement learning environment that models key challenges of real-world industrial control: heteroscedastic noise, delayed and convoluted reward signals, and a complex dependency between actions, fatigue, and operational cost.

Originally introduced in [Hein et al., 2017](https://arxiv.org/abs/1709.09480), this implementation is a modernised Python port integrated into the Industrial Benchmark Suite.

## Environment Description

### State Space

The observable state consists of 8 variables, of which 6 are exposed to the agent by default:

| Num | Observation  | Min  | Max    | Description                          |
|-----|------------- |------|--------|--------------------------------------|
| 0   | Setpoint     | 0    | 100    | External setpoint (may be stationary or dynamic) |
| 1   | Velocity     | 0    | 100    | Current velocity control variable    |
| 2   | Gain         | 0    | 100    | Current gain control variable        |
| 3   | Shift        | 0    | 100    | Current shift control variable       |
| 4   | Fatigue      | 0    | ~1000  | Wear-induced cost component          |
| 5   | Consumption  | 0    | ~1000  | Operational cost component           |
| 6   | Cost         | —    | —      | Total cost = CRF × fatigue + CRC × consumption |
| 7   | Reward       | —    | —      | Negative of total cost               |

### Action Space

The agent controls 3 continuous action deltas, each clipped to [-1, +1]:

| Num | Action          | Min  | Max  | Description                      |
|-----|-----------------|------|------|----------------------------------|
| 0   | Delta Velocity  | -1.0 | +1.0 | Additive change to velocity      |
| 1   | Delta Gain      | -1.0 | +1.0 | Additive change to gain (×10)    |
| 2   | Delta Shift     | -1.0 | +1.0 | Additive change to shift (scaled)|

A discrete action mode (27 actions) is also available via `action_type="discrete"`.

### Reward Function

The reward is the negative total cost:

    reward = -(CRF × fatigue + CRC × consumption)

where:
- **Fatigue** depends non-linearly on the effective velocity and gain, amplified by a hidden wear process with stochastic noise.
- **Consumption** is derived from a convolution over recent operational costs, perturbed by a Goldstone-potential-based miscalibration penalty and additive noise.

The reward function `"classic"` returns the instantaneous reward; `"delta"` returns the step-over-step change.

### Starting State

At each `reset()`, the IDS core is re-initialised with `setpoint=p`, velocity/gain/shift at 50, and a fresh random seed.

### Episode End

Episodes are truncated (`truncated=True`, `terminated=False`) after `reset_after_timesteps` steps (default: 1000).

## References

- D. Hein, S. Depeweg, M. Tokic, S. Udluft, A. Hentschel, T. A. Runkler, V. Sterzing. "A benchmark environment motivated by industrial control problems." *IEEE Symposium Series on Computational Intelligence (SSCI)*, 2017.
