"""Register pre-configured environements as benchmarks."""

from gymnasium.envs.registration import register

register(
    id="RealCartPole-v0",
    entry_point="ibs.envs.real_cartpole.real_cartpole_env:create_real_cartpole_env",
)
