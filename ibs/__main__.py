"""Command-line interface for Industry Benchmarks."""
from pathlib import Path
from typing import Optional

import typer


app = typer.Typer(
    help='Industry Benchmarks - Gym environments for industrial process control benchmarking',
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command('ccbench')
def ccbench(
    data_dir: Optional[Path] = typer.Option(
        None,
        '--data-dir',
        help='Path of the directory which receives generated data, default "./data" (but potentially overwritten by configuration file).',
    ),
    config_file: Path = typer.Option(
        Path('config') / 'ccbench_default.yaml',
        '--config-file',
        help='Filename of the YAML configuration file with specifications for the data to process/generate, '
             'default "./config/ccbench_default.yaml".',
    ),
):
    """Generate synthetic timeseries data from the Continuous Casting Benchmark environment."""
    from ibs.envs.ccbench.generate_data import generate_data

    return generate_data(
        config_file=str(config_file),
        config_dict={
            'data_dir': str(data_dir) if data_dir is not None else None,
            'config_file': str(config_file),
        },
    )


if __name__ == '__main__':
    app()
