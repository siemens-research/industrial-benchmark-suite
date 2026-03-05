"""
Continuous Casting Benchmark (CCBench) environment.

A synthetic continuous casting process control environment for RL benchmarking.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

from gymnasium.envs.registration import register

from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.generate_data import generate_data

__all__ = ["CCBenchEnv", "generate_data"]

register(
    id="CCBench-v0",
    entry_point="ibs.envs.ccbench.ccbench_env:create_ccbench_env",
)
