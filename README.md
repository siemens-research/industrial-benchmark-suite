# Industrial Benchmark Suite (IBS)

The Industrial Benchmark Suite (IBS) is a collection of simulated environments designed to facilitate research and development in the field of reinforcement learning (RL) for industrial control systems. The suite includes a variety of environments that mimic real-world industrial processes, allowing researchers to test, evaluate and improve RL algorithms in settings which pose special challenges w.r.t. system identification, task variability and noise conditions.


## Environments

The suite currently includes the following environments, which are described in more detail in the corresponding documentation pages:

- [Real Cartpole Environment](docs/envs/real_cartpole/real_cartpole.md)
- [Distillation Environment](docs/envs/distillation/distillation.md)
- [Four Tank Environment](docs/envs/four_tank/four_tank.md)
- [Continuous Casting Benchmark](docs/envs/ccbench/continuous_casting_concept.md)

## Datasets

In addition to the environments, the IBS also provides the original datasets of pre-recorded measurements from the corresponding hardware setups or industry-grade process simulations for some of the environments. These datasets can be used for pure offline RL research and development, or as a foundation for training and benchmarking of time-series modeling methods.


## Installation

Install the package directly from the repository using pip:

```bash
pip install git+https://code.siemens.com/ai/industrial-benchmark-suite.git
```

Or, if you have cloned the repository locally:

```bash
pip install .
```

## Contributing

To set up a development environment, first clone the repository and then install all dependencies (including dev/test extras) using either **Poetry** or **uv**:

### Using Poetry

```bash
poetry install
poetry run pytest
```

### Using uv

```bash
uv sync
uv run pytest
```

Before committing, make sure pre-commit hooks are active:

```bash
pre-commit install
```


## Data generation toolkit

After installation of the project, a command line tool is provided which allows to generate example rollout data from
some of the provided environments. An example call is the following, where the number of episodes (rollouts) and
duration of an episode are configured in the specified configuration file.

```bash
gen_industry_benchmarks ccbench --config-file resources/envs/ccbench/config/ccbench_default.yaml
```

## License

The code in this repository is licensed under the Siemens Inner Source License v1.4. See the [LICENSE](LICENSE) file for details.


## Contributors

- Marc Weber (marc-weber@siemens.com)
- Holger Schoener (holger.schoener@siemens.com)


## Acknowledgements

Parts of the work contained in this repository were funded by the German Federal Ministry of Education and Research (BMFTR) in the context of the project "RIESIQ" (grant number 16IS24087A). We thank our project partners for their collaboration and support.
