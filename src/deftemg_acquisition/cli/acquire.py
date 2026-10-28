"""Record EMG trials from the keyboard.

    pixi run acquire --port /dev/cu.usbmodem1101 --subject S01 --viewer
    pixi run acquire --simulate --viewer        # no hardware needed

Keys: s = start trial, e = end trial, q = quit.

By default the armband streams for the whole session (live over LSL for the viewer) and
only the data between s and e is saved.

Asks for your user ID and, if you have recorded before, whether you re-donned the band since
your last session (a new donning id if yes; the same one if no). ``--subject`` and ``--redon`` /
``--same-donning`` answer in advance.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig, subjects

START_KEY, END_KEY, QUIT_KEY = "s", "e", "q"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument(
        "--powerline-hz", type=int, default=60, choices=[50, 60],
        help="mains frequency: the armband's notch with --device-filter, else the live notch",
    )  # fmt: skip
    ap.add_argument(
        "--highpass-hz", type=int, default=60, choices=[20, 40, 60],
        help="the armband's high-pass, with --device-filter",
    )  # fmt: skip
    ap.add_argument(
        "--device-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="the armband's built-in filters (power-line notch and high-pass go on and off "
        "together); off by default, recording raw 24-bit EMG (filter it offline with "
        "filter-emg)",
    )
    ap.add_argument(
        "--live-highpass", choices=["none", "20", "40", "60"], default="20",
        help="without --device-filter: causal 4th-order Butterworth high-pass (Hz) on the live "
             "stream and viewer; recordings stay raw (default: 20)",
    )  # fmt: skip
    ap.add_argument(
        "--live-notch",
        choices=["none", "50", "60", "comb50", "comb60", "comb50all", "comb60all"],
        help="without --device-filter: 2 Hz wide notch on the live stream and viewer at the "
             "mains frequency; comb50 / comb60 add every odd harmonic below 1 kHz, for a mains-"
             "powered device attached or high mains pickup; comb50all / comb60all add every "
             "harmonic, odd and even (default: comb60all, or comb50all with --powerline-hz 50, "
             "as filter-emg filters recordings)",
    )  # fmt: skip
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--session", help="session folder name (default: timestamp)")
    ap.add_argument("--label", help="label applied to every trial")
    ap.add_argument(
        "--arm", choices=["right", "left"], default="right", help="arm wearing the band"
    )
    ap.add_argument("--viewer", action="store_true", help="also open the live viewer")
    ap.add_argument("--no-stream", action="store_true", help="don't publish over LSL")
    ap.add_argument(
        "--per-trial-acq",
        action="store_true",
        help="send StartDataAcq/StopDataAcq around each trial instead of streaming throughout",
    )
    ap.add_argument("--simulate", action="store_true", help="use a simulated armband")
    ap.add_argument("-v", "--verbose", action="store_true")
    subjects.add_arguments(ap)
    args = ap.parse_args()
    donning = subjects.donning_from_args(args, args.data_dir)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deftemg_acquisition.acquisition.keyboard import cbreak_keys

    config = SessionConfig(
        port=args.port,
        powerline_hz=args.powerline_hz,
        highpass_hz=args.highpass_hz,
        filter_enabled=args.device_filter,
        live_highpass_hz=None if args.live_highpass == "none" else float(args.live_highpass),
        live_notch=args.live_notch,
        data_dir=args.data_dir,
        session_name=args.session,
        continuous=not args.per_trial_acq,
        live_stream=not args.no_stream,
        metadata={
            "arm": args.arm,
            **donning.metadata(),
            **({"simulated": True} if args.simulate else {}),
        },
    )
    device = None
    if args.simulate:
        from deftemg_acquisition.acquisition.sim import SimulatedArmband

        device = Armband("simulated", ser=SimulatedArmband())

    viewer = None
    if args.viewer:
        viewer = subprocess.Popen(
            [sys.executable, "-m", "deftemg_acquisition.viz.viewer", "--arm", args.arm]
        )
    try:
        with AcquisitionSession(config, device=device) as session:
            print(f"Saving to {session.session_dir}")
            print(f"[{START_KEY}] start trial   [{END_KEY}] end trial   [{QUIT_KEY}] quit")
            with cbreak_keys() as keys:
                for key in keys:
                    if key == START_KEY and not session.in_trial:
                        session.start_trial(label=args.label)
                    elif key == END_KEY and session.in_trial:
                        session.end_trial()
                    elif key == QUIT_KEY:
                        break
    finally:
        if viewer is not None:
            viewer.terminate()


if __name__ == "__main__":
    main()
