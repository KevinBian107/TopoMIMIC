"""Run MDS subspace projection as an atomic block."""

from .._block_runner import cli, run_single_block

FLAG = "run_mds_analysis"


def main(config_path: str) -> None:
    run_single_block(config_path, FLAG)


if __name__ == "__main__":
    cli(FLAG)
