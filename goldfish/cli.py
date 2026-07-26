"""Goldfish command dispatcher installed as the ``goldfish`` console script."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import infer
import prepare
import train


Command = Callable[[Sequence[str] | None], int]
_COMMANDS: dict[str, Command] = {
    "prepare": prepare.main,
    "train": train.main,
    "infer": infer.main,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(_COMMANDS), help="Goldfish command to run.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments forwarded unchanged to the selected command.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args.arguments)
