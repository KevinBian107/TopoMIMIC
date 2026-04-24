"""Run per-cluster transition comparison plots as an atomic block."""

from .._block_runner import cli, run_single_block

FLAG = "run_transition_comparison"


def main(config_path: str) -> None:
    run_single_block(config_path, FLAG)


if __name__ == "__main__":
    cli(FLAG)
