"""Register pre-configured environements as benchmarks."""

from gymnasium.envs.registration import register

register(
    id="Distillation-v0",
    entry_point="ibs.envs.distillation.distillation_env:create_distillation_env",
)
