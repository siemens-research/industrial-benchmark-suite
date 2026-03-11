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

The reward is computed as the sum over the negative absolute deviations of the current water levels in the 3 main tanks from their corresponding objectives (demand levels) minus a regularization term on the action changes (sum of negative absolute value of the 1-step differences over all 3 actions). The maximum reward is 0, which is achieved when the water levels in all 3 main tanks perfectly match their objectives and no more action changes needed to be applied.

### Starting State

Depending on the config field 'init_method', the starting state is either drawn as a 'random' sample or set by a 'fixed' sample (as defined by the 'fixed_init_observation' field).

### Episode End

Depending on the config field 'max_episode_steps', the episode is either terminated when the maximum step limit is reached or is run infinitely.
