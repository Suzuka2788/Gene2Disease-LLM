"""The single command-line entry point."""
from __future__ import annotations

import argparse

from . import experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="g2d")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the stage named in a configuration file")
    run.add_argument("--config", required=True)
    arguments = parser.parse_args(argv)
    print(experiments.run(arguments.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
