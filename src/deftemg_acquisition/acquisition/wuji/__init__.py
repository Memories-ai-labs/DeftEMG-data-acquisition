"""Wuji Glove recording alongside the armband (``wuji-sdk``, optional dependency).

``WujiGloveRecorder`` is an ``AcquisitionSession`` source; ``add_arguments`` /
``sources_from_args`` give the ``acquire`` and ``prompter`` commands their ``--wuji`` options.
"""

from __future__ import annotations

import argparse

from .wuji_recorder import STREAMS, TOPICS, StreamBuffer, WujiGloveRecorder

__all__ = [
    "STREAMS",
    "TOPICS",
    "StreamBuffer",
    "WujiGloveRecorder",
    "add_arguments",
    "sources_from_args",
]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("Wuji Glove")
    g.add_argument("--wuji", action="store_true", help="record a Wuji Glove with every trial")
    g.add_argument(
        "--wuji-sn", nargs="+", metavar="SN",
        help="serial numbers of the gloves to record (default: every glove found)",
    )  # fmt: skip


def sources_from_args(args: argparse.Namespace) -> list[WujiGloveRecorder]:
    if not args.wuji:
        return []
    return [WujiGloveRecorder(sn=args.wuji_sn)]
