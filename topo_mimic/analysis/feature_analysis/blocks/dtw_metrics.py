"""Run DTW metrics + internal consistency as an atomic block.

``run_dtw_metrics`` drives both the DTW sequence comparison and the internal
consistency analysis (paired in ``main_transitions.run_analysis``), plus the
DTW/consistency aggregation tables.
"""

from .._block_runner import cli, run_single_block

FLAG = "run_dtw_metrics"


def main(config_path: str) -> None:
    run_single_block(config_path, FLAG)


if __name__ == "__main__":
    cli(FLAG)
