"""Wuji Glove recording alongside the armband (``wuji-sdk``, optional dependency).

``WujiGloveRecorder`` is an ``AcquisitionSession`` source; ``add_arguments`` /
``sources_from_args`` give the ``acquire`` and ``prompter`` commands their ``--wuji`` options.
"""

from __future__ import annotations

import argparse

from .wuji_recorder import DEFAULT_STREAMS, STREAMS, StreamBuffer, WujiGloveRecorder

__all__ = [
    "DEFAULT_STREAMS",
    "STREAMS",
    "StreamBuffer",
    "WujiGloveRecorder",
    "add_arguments",
    "sources_from_args",
]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("Wuji Glove")
    g.add_argument("--wuji", action="store_true", help="record a Wuji Glove with every trial")
    g.add_argument("--wuji-sn", help="glove serial number (default: the only glove found)")
    g.add_argument(
        "--wuji-calibrate", action="store_true",
        help="calibrate each glove's hand model before recording, under a new timestamped SDK user",
    )  # fmt: skip
    g.add_argument(
        "--wuji-user", metavar="NAME",
        help="name of that SDK user, before its timestamp (default: deftemg)",
    )  # fmt: skip
    g.add_argument(
        "--wuji-streams",
        nargs="+",
        default=list(DEFAULT_STREAMS),
        choices=sorted(STREAMS),
        metavar="STREAM",
        help=f"glove streams to record (default: {' '.join(DEFAULT_STREAMS)})",
    )


def sources_from_args(args: argparse.Namespace) -> list[WujiGloveRecorder]:
    if not args.wuji:
        return []
    live = not getattr(args, "no_stream", False)  # with the armband's live stream
    return [WujiGloveRecorder(sn=args.wuji_sn, streams=args.wuji_streams, live=live,
                              calibrate=args.wuji_calibrate, user=args.wuji_user)]  # fmt: skip
