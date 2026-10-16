"""Register pre-configured environments as benchmarks."""

from gymnasium.envs.registration import register

from ibs.envs.real_cartpole.real_cartpole_env import RealCartPoleEnv
from ibs.envs.four_tank.four_tank_env import FourTankEnv
from ibs.envs.distillation.distillation_env import DistillationEnv
from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.generate_data import generate_data
from ibs.envs.industrial_benchmark.industrial_benchmark_env import IBGym

__all__ = ["RealCartPoleEnv", "FourTankEnv", "DistillationEnv", "CCBenchEnv", "IBGym", "generate_data"]

register(
    id="RealCartPole-v0",
    entry_point="ibs.envs.real_cartpole.real_cartpole_env:create_real_cartpole_env_v0",
)

register(
    id="FourTank-v0",
    entry_point="ibs.envs.four_tank.four_tank_env:create_four_tank_env",
)

register(
    id="Distillation-v0",
    entry_point="ibs.envs.distillation.distillation_env:create_distillation_env",
)

register(
    id="CCBench-v0",
    entry_point="ibs.envs.ccbench.ccbench_env:create_ccbench_env_v0",
)

register(
    id="CCBench-v1",
    entry_point="ibs.envs.ccbench.ccbench_env:create_ccbench_env",
)

register(
    id="IndustrialBenchmark-v0",
    entry_point="ibs.envs.industrial_benchmark.industrial_benchmark_env:IBGym",
    kwargs={"setpoint": 50},
)
