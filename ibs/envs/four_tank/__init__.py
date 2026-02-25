"""Register pre-configured environements as benchmarks."""

from gymnasium.envs.registration import register

register(
    id="FourTank-v0",
    entry_point="ibs.envs.four_tank.four_tank_env:create_four_tank_env",
)
