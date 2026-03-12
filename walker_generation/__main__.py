"""CLI entry point: python3 -m walker_generation <config.yaml>"""

import argparse

from walker_generation.config import load_config
from walker_generation.generate import run_generation


def main():
    parser = argparse.ArgumentParser(
        description="Generate walker transition clips in TopoVNL format.",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to generation config YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_generation(config)


if __name__ == "__main__":
    main()
