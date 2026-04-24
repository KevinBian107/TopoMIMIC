"""CLI entry for ``python -m topo_mimic.analysis.tables``."""

from __future__ import annotations

import sys


_SUBCOMMANDS = ("all", "paper", "warping", "results")


def _run_paper() -> None:
    from .paper_tables import main as paper_main
    paper_main()


def _run_warping() -> None:
    from .warping_tables import main as warping_main
    warping_main()


def _run_results() -> None:
    from .results_table import main as results_main
    results_main(argv=[])


def main() -> None:
    sub = sys.argv[1] if len(sys.argv) > 1 else "all"
    if sub not in _SUBCOMMANDS:
        print(f"unknown subcommand {sub!r}; pick one of {_SUBCOMMANDS}")
        sys.exit(2)
    if sub in ("all", "results"):
        _run_results()
    if sub in ("all", "paper"):
        _run_paper()
    if sub in ("all", "warping"):
        _run_warping()


if __name__ == "__main__":
    main()
