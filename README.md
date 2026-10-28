# Industrial Benchmark Suite (IBS)

The Industrial Benchmark Suite (IBS) is a collection of simulated environments designed to facilitate research and development in the field of reinforcement learning (RL) for industrial control systems. The suite includes a variety of environments that mimic real-world industrial processes, allowing researchers to test, evaluate and improve RL algorithms in settings which pose special challenges regarding system identification, task variability and noise conditions.


## Environments

The suite currently includes the following environments, which are described in more detail in the corresponding documentation pages:

- [Real Cartpole Environment](docs/envs/real_cartpole/real_cartpole.md)
- [Distillation Environment](docs/envs/distillation/distillation.md)
- [Four Tank Environment](docs/envs/four_tank/four_tank.md)
- [Industrial Benchmark](docs/envs/industrial_benchmark/industrial_benchmark.md)
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

To set up a development environment, first clone the repository and then install all dependencies (including the dev group) using **uv**:

```bash
uv sync
uv run pytest
```

The optional documentation dependencies can be installed with:

```bash
uv sync --group docs
```

Before committing, make sure pre-commit hooks are active:

```bash
uv run pre-commit install
```


## Data generation toolkit

After installation of the project, a command line tool is provided which allows to generate example rollout data from
some of the provided environments. An example call is the following, where the number of episodes (rollouts) and
duration of an episode are configured in the specified configuration file.

```bash
gen_industry_benchmarks ccbench --config-file resources/envs/ccbench/config/ccbench_default.yaml
```

## License

The code in this repository is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.


## Main contributors

- Marc Weber (marc-weber@siemens.com)
- Holger Schoener (holger.schoener@siemens.com)
- Daniel Hein (hein.daniel@siemens.com)

## Citation

A scientific paper describing the Industrial Benchmark Suite is currently in preparation. Until a dedicated
publication is available, please cite the software project directly. For example:

> Weber, M., Schoener, H., & Hein, D. (2026). *Industrial Benchmark Suite (IBS): A collection of industrial
> benchmark environments for reinforcement learning* (Version 0.1.0) [Computer software].
> Siemens AG. https://code.siemens.com/ai/industrial-benchmark-suite

Or, using BibTeX:

```bibtex
@software{industrial_benchmark_suite,
  author  = {Weber, Marc and Schoener, Holger and Hein, Daniel},
  title   = {{Industrial Benchmark Suite (IBS): A collection of industrial benchmark
             environments for reinforcement learning}},
  year    = {2026},
  version = {0.1.0},
  publisher = {Siemens AG},
  url     = {https://code.siemens.com/ai/industrial-benchmark-suite}
}
```

Please update the `version` and `year` fields to match the release you used, and watch this section for an
updated reference once an accompanying paper is published.

### Citing the Industrial Benchmark

If you explicitly use the **Industrial Benchmark** environment contained in this suite, please additionally
cite its dedicated publication:

> D. Hein, S. Depeweg, M. Tokic, S. Udluft, A. Hentschel, T. A. Runkler, and V. Sterzing. "A benchmark
> environment motivated by industrial control problems," in *2017 IEEE Symposium Series on Computational
> Intelligence (SSCI)*, 2017, pp. 1-8.

Or, using BibTeX:

```bibtex
@inproceedings{hein2017industrial,
  author    = {Hein, Daniel and Depeweg, Stefan and Tokic, Michel and Udluft, Steffen
               and Hentschel, Alexander and Runkler, Thomas A. and Sterzing, Volkmar},
  title     = {A benchmark environment motivated by industrial control problems},
  booktitle = {2017 IEEE Symposium Series on Computational Intelligence (SSCI)},
  year      = {2017},
  pages     = {1--8}
}
```

## Acknowledgements

Parts of the work contained in this repository were funded by the German Federal Ministry of Education and Research (BMFTR) in the context of the project "RIESIQ" (grant number 16IS24087A). We thank our project partners for their collaboration and support.
