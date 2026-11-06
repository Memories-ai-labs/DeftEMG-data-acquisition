"""Stream the armband to other computers, without recording here.

    pixi run serve                              # the armband (auto-detected port)
    pixi run serve --port /dev/cu.usbmodem1101
    pixi run serve --simulate                   # no hardware

Opens the armband, sets its filters and publishes its streams over LSL until Ctrl-C: EMG, IMU
and Markers as ``acquire`` does, plus its serial bytes unchanged (``acquisition.remote``), from
which a session on another computer records and streams as if the armband were plugged into it:

    pixi run prompter --remote <source_id or this computer's host name> ...   # over there
    pixi run acquire --remote <...>
    pixi run viewer --source <...>              # only to watch

Prints the ``source_id`` to give ``--remote``, then once in a while how many bytes went out and
whether anyone is reading them. Nothing is saved on this computer (``acquire`` and ``prompter``
publish the same streams while they record, if both computers should keep a copy).
"""

from __future__ import annotations

import argparse
import logging
import socket
import time
from pathlib import Path

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--powerline-hz", type=int, default=60, choices=[50, 60],
                    help="mains frequency: the armband's notch with --device-filter")  # fmt: skip
    ap.add_argument("--highpass-hz", type=int, default=60, choices=[20, 40, 60],
                    help="the armband's high-pass, with --device-filter")  # fmt: skip
    ap.add_argument(
        "--device-filter", action=argparse.BooleanOptionalAction, default=False,
        help="the armband's built-in filters (off by default: raw 24-bit EMG, as acquire)",
    )  # fmt: skip
    ap.add_argument("--simulate", action="store_true", help="use a simulated armband")
    ap.add_argument("--duration", type=float, help="stop after this many s (tests)")
    ap.add_argument("--status-every", type=float, default=10.0, help="s between status lines")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    device = None
    if args.simulate:
        from deftemg_acquisition.acquisition.sim import SimulatedArmband

        device = Armband(args.port or "simulated", ser=SimulatedArmband())
    import tempfile

    config = SessionConfig(
        port=args.port,
        powerline_hz=args.powerline_hz,
        highpass_hz=args.highpass_hz,
        filter_enabled=args.device_filter,
        data_dir=Path(tempfile.mkdtemp(prefix="serve-")),  # session.json only: no trials
        live_stream=True,
        raw_stream=True,
    )
    with AcquisitionSession(config, device=device) as session:
        print(f"Streaming the armband: source_id {session.source_id} on {socket.gethostname()}")
        print(f"  on the other computer: --remote {session.source_id}   (Ctrl-C to stop)")
        start = last = time.monotonic()
        try:
            while args.duration is None or time.monotonic() - start < args.duration:
                time.sleep(0.2)
                if time.monotonic() - last >= args.status_every:
                    last = time.monotonic()
                    readers = "read" if session.raw.have_consumers() else "nobody reading"
                    print(f"  {session.raw.n_bytes / 1e6:.1f} MB sent ({readers})", flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
