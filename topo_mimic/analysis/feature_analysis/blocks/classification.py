"""
Run per-cluster classification metrics as an atomic block.

Classification auto-activates when ``sample_labels_seed_*.yaml`` is present,
regardless of ``run_*`` flags. Disabling every other flag via
``_block_runner.run_single_block`` still triggers the classification block
because ``main_transitions.run_analysis`` always checks for sample labels.
"""

from .._block_runner import cli, run_single_block

FLAG = "run_purity_analysis"


def main(config_path: str) -> None:
    run_single_block(config_path, FLAG)


if __name__ == "__main__":
    cli(FLAG)
