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
        "--wuji-simulate", action="store_true", help="record a simulated glove (implies --wuji)"
    )
    g.add_argument(
        "--wuji-replay",
        metavar="MCAP",
        help="play a glove recording (the SDK's MCAP) as the glove (implies --wuji)",
    )
    g.add_argument(
        "--wuji-replay-speed", type=float, default=1.0, help="replay speed factor (default 1)"
    )
    g.add_argument(
        "--wuji-replay-loop", action="store_true", help="play the recording again and again"
    )
    g.add_argument(
        "--wuji-streams",
        nargs="+",
        default=list(DEFAULT_STREAMS),
        choices=sorted(STREAMS),
        metavar="STREAM",
        help=f"glove streams to record (default: {' '.join(DEFAULT_STREAMS)})",
    )


def sources_from_args(args: argparse.Namespace) -> list[WujiGloveRecorder]:
    if not (args.wuji or args.wuji_simulate or args.wuji_replay):
        return []
    if args.wuji_simulate and args.wuji_replay:
        raise SystemExit("--wuji-simulate and --wuji-replay: choose one")
    glove = None
    if args.wuji_replay:
        from .replay import ReplayGlove

        glove = ReplayGlove(
            args.wuji_replay, speed=args.wuji_replay_speed, loop=args.wuji_replay_loop
        )
    elif args.wuji_simulate:
        from .sim import SimulatedGlove

        glove = SimulatedGlove()
    live = not getattr(args, "no_stream", False)  # with the armband's live stream
    return [WujiGloveRecorder(sn=args.wuji_sn, streams=args.wuji_streams, glove=glove, live=live)]
