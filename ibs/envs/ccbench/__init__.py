"""
Continuous Casting Benchmark (CCBench) environment.

A synthetic continuous casting process control environment for RL benchmarking.
"""

from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.generate_data import generate_data

__all__ = ["CCBenchEnv", "generate_data"]
