# Industrial Benchmark Suite (IBS)

## Environment Descriptions

- [Real Cartpole Environment](docs/envs/real_cartpole/real_cartpole.md)
- [Distillation Environment](docs/envs/distillation/distillation.md)
- [Four Tank Environment](docs/envs/four_tank/four_tank.md)


## Data generation script

After installation of the project, a command line tool is provided which allows to generate example rollout data from
some of the provided environments. An example call is the following, where the number of episodes (rollouts) and
duration of an episode are configured in the specified configuration file.

```bash
gen_industry_benchmarks ccbench --config-file resources/envs/ccbench/config/ccbench_default.yaml
```
