# Four Tank Benchmark

## Gymnasium Environment for a 4-Tank Process Demonstrator

In the 4-tank system, water can be pumped in between four interconnected tanks. The goal is to control the water levels in the tanks by applying appropriate valve opening levels. This benchmark environment provides a realistic simulation of the system dynamics based on pre-trained dynamic models derived from real-world measurements.

## Environment Description

### State Space

The state space of the 4-tank system is defined as follows. The first 6 variables correspond to the current valve opening levels, the next 3 variables correspond to the current water levels in the main tanks (as measured by level meters) and the last 3 variables are their corresponding objectives (external inputs).

| Num | Observation                | Min   | Max   |
|-----|----------------------------|-------|-------|
| 0   | Inlet Valve Tank 1         | 0.0   | 700.0 |
| 1   | Inlet Valve Tank 2         | 0.0   | 700.0 |
| 2   | Inlet Valve Tank 3         | 0.0   | 700.0 |
| 3   | Connecting Valve Tank 1-2  | 0.0   | 100.0 |
| 4   | Connecting Valve Tank 3-4  | 0.0   | 100.0 |
| 5   | Connecting Valve Tank 2-3  | 0.0   | 100.0 |
| 6   | Level Meter Tank 1         | 0.0   | 21.0  |
| 7   | Level Meter Tank 2         | 0.0   | 21.0  |
| 8   | Level Meter Tank 3         | 0.0   | 21.0  |
| 9   | Objective Level Tank 1     | 0.0   | 21.0  |
| 10  | Objective Level Tank 2     | 0.0   | 21.0  |
| 11  | Objective Level Tank 3     | 0.0   | 21.0  |

### Action Space

The system is controlled by the following continuous (floating point) control actions. Each action corresponds to the setpoint of a valve opening level, which is applied at each step.

| Num | Action                     | Min   | Max   |
|-----|----------------------------|-------|-------|
| 0   | Control Valve 1            | 0.0   | 700.0 |
| 1   | Control Valve 2            | 0.0   | 700.0 |
| 2   | Control Valve 3            | 0.0   | 700.0 |
| 3   | Control Valve 1-2          | 0.0   | 100.0 |
| 4   | Control Valve 3-4          | 0.0   | 100.0 |

### Reward Function

The reward at each step is the sum of two components:

1. Setpoint adherence – the negative absolute deviation of each level meter from its corresponding objective, weighted by 1.0:

   ```
   r_setpoint = -1.0 * |Level Meter Tank 1 - Objective Level Tank 1|
                -1.0 * |Level Meter Tank 2 - Objective Level Tank 2|
                -1.0 * |Level Meter Tank 3 - Objective Level Tank 3|
   ```

2. Action regularization – a penalty on the 1-step change of each valve observation (weights: 0.001 for inlet valves, 0.01 for connecting valves):

   ```
   r_action = -0.001 * |Inlet Valve Tank k(t) - Inlet Valve Tank k(t-1)|    (k = 1, 2, 3)
              -0.01  * |Connecting Valve j(t) - Connecting Valve j(t-1)|     (j = 1-2, 3-4)
   ```

The total reward is `r = r_setpoint + r_action`. The maximum achievable reward is 0, attained when all water levels match their objectives and no valve changes are applied.

### Starting State

Depending on the config option "init_method", the starting state is either drawn from a set list
of random initial states ("from_data") or always using the same initial sample provided by the
"initial_observation" option ("fixed").

### Episode End

Depending on the config field "max_episode_length", the episode is either truncated when the
maximum step limit is reached (default: 100) or is run infinitely (None).
