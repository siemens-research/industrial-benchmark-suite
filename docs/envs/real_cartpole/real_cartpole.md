# Real Cartpole Benchmark

![real_cartpole_banner.PNG](real_cartpole_banner.PNG)

## Gym(nasium) Environment for the Siemens Hardware Cartpole

The environment provides a simulation of the Siemens hardware cartpole system
currently located at the Siemens Technology site in Munich/Neuperlach ("Real Cartpole")
based on pre-trained dynamic models. It includes a realistic representation
("digital twin") of the actual physical system, including friction and solid body
dynamics, amongst others.

## Environment Description

### State Space

The state space of the real cartpole system is defined as follows:

| Num | Observation           | Min                | Max               |
|-----|-----------------------|--------------------|-------------------|
| 0   | Cart Position         | 0 mm               | 600 mm            |
| 1   | Cart Velocity         | -Inf               | Inf               |
| 2   | Pole Angle            | -$\pi$ rad (-180°) | +$\pi$ rad (180°) |
| 3   | Pole Angular Velocity | -Inf               | Inf               |

The angular coordinates are defined such that the upright pole position is given by 0 rad and
downward position by +/- pi rad. Be aware that position space is not well explored
and modelled near the outer edges (Cart Position < 150mm and  Cart Position > 450mm).

### Action Space

The system is controlled by a 1-dimensional, continuous (floating point) action in
[-1, 1] which is equivalent to the motor current (force signal) applied to the cart.

### Rewards

The continuous reward signal is composed of 3 components that are linearly weighted:

```python
reward = 0.01 * (
    - min(abs(CartPosition - 325.0) / 10.0, 30.0)
    - min(abs(PoleAngle) / 5.0, 36.0)
    - min(abs(PoleAngularVelocity) / 10.0, 3.0) * (-15.0 < PoleAngle < 15.0)
)
```

The maximum achievable reward is 0 (if all three criteria are perfectly met).
Any deviation from the individual targets result in a negative reward signal.
If the episode is terminated before the maximum steps number is reached,
a negative reward of -1.*(steps_missed) is returned, where steps_missed corresponds
to the remaining number of steps.

### Starting State

Depending on the config field 'init_method', the starting state is either drawn as a 'random'
sample or set by a 'fixed' sample (as defined by the 'initial_observation' field).

### Episode End

Depending on the config field 'max_episode_length', the episode is either truncated
(`truncated=True`) when the maximum step limit is reached (default: 500) or is run infinitely
(None). Independent of the step limit, the episode is terminated (`terminated=True`) as a failure
if the state leaves the observation space, e.g. if the cart moves outside the allowed position
interval between 0 and 600mm.
