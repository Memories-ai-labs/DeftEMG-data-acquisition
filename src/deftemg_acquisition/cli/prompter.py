"""Run the scrolling finger-press prompter while recording EMG.

    pixi run prompter --port /dev/cu.usbmodem1101 --subject S01 --viewer protocols/demo
    pixi run prompter --simulate protocols/demo/d2_multi_b01.json

Sequences are JSON files from ``make-sequences``; directories are expanded (sorted by name).
Keys in the prompter window: Shift start block, Enter end early, Left/Right choose block,
Esc quit.

Asks for your user ID and, if you have recorded before, whether you re-donned the band since
your last session (a new donning id if yes; the same one if no). ``--subject`` and ``--redon`` /
``--same-donning`` answer in advance.

Each session starts with a calibration block (about 25 s): rest quietly, then one brief effort
as hard as is comfortable (``--calibration-reps``: how many), matching the task blocks: before
pinch or squeeze blocks, a pinch or squeeze of the thumb with each other finger (thumb implied,
as in those blocks); otherwise a press of each digit. It sets the session's noise and
maximum-effort levels (``deftemg_acquisition.data.calibration``: e.g. ``press_fraction`` gives
low / high presses as a fraction of the maximum of the same digit or pinch).
Another calibration block comes up ~10 minutes later, then every 20 minutes (``--recalibrate``:
minutes to the second block, then between later ones), before the next block once it is due,
so every trial has a recent calibration before it. ``--no-calibration`` skips them
all.

``--effort-meter`` adds an effort bar under the target row once a calibration block is done:
overall EMG effort relative to the calibration maximum of what is prompted (a digit, a pinch,
or a chord's digits together), with a zone per level (``--effort-zones``, default low 20-40 %,
high 65-85 %). Keep the bar in the outlined zone while holding a press. Its trace is saved with
each block, and the trial's metadata records that the meter was on.

``--viewer`` opens the live viewer in its own window (and process, so it can't slow the
prompter), the two side by side at full height: the prompter on the left, the viewer on the
right. ``--no-stream`` opens no viewer (it reads the LSL stream).

``--replay PATH`` replays recorded sessions or trials instead of the armband
(``acquisition.replay``), for trying a setup without hardware: it plays once, from when the
prompter opens, and the replayed EMG doesn't follow the prompts on screen.

Other code can add options, task window plugins and viewer options (``main``'s ``extension``:
:class:`PrompterExtension`).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pyqtgraph as pg

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig, subjects
from deftemg_acquisition.tasks.app import TaskPlugin, TaskWindow
from deftemg_acquisition.tasks.effort import ZONES
from deftemg_acquisition.tasks.prompter import DEFAULT_LOOKAHEAD_S
from deftemg_acquisition.tasks.sequence import CALIBRATION_REPS, Sequence, calibration_for


def expand(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(p.glob("*.json")) if p.is_dir() else [p]
    return out


class PrompterExtension:
    """Adds to the prompter from other code (``main``'s ``extension``)."""

    viewer_module = "deftemg_acquisition.viz.viewer"  # the live viewer (run with python -m)

    def add_arguments(self, ap: argparse.ArgumentParser) -> None:
        """More options (and help: ``ap.description``)."""

    def plugins(self, args: argparse.Namespace, config: SessionConfig) -> list[TaskPlugin]:
        """Task window plugins, once the session's config is set (before it opens); may exit
        on bad options."""
        return []

    def viewer_args(self, args: argparse.Namespace) -> list[str] | None:
        """More options for the live viewer; a list opens it even without ``--viewer``."""
        return None


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("sequences", type=Path, nargs="+", help="sequence files or directories")
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--simulate", action="store_true", help="use a simulated armband")
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument(
        "--lookahead", type=float, default=DEFAULT_LOOKAHEAD_S, help="s of prompts visible ahead"
    )
    ap.add_argument("--session", help="session folder name (default: timestamp)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
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
    ap.add_argument(
        "--sync-patch",
        action="store_true",
        help="flash a corner square at each prompt onset (photodiode / slow-motion video check)",
    )
    ap.add_argument(
        "--no-calibration", action="store_true",
        help="skip the calibration blocks (rest, then a maximum press per digit)",
    )  # fmt: skip
    ap.add_argument(
        "--calibration-reps", type=int, default=CALIBRATION_REPS, metavar="N",
        help=f"maximum presses per digit in each calibration block (default {CALIBRATION_REPS}; "
             "2 were too tiring)",
    )  # fmt: skip
    ap.add_argument(
        "--recalibrate", type=float, nargs="+", default=[10.0, 20.0], metavar="MIN",
        help="minutes from one calibration block to the next: the first value after the "
             "starting block, the last one repeated (default: 10 20)",
    )  # fmt: skip
    ap.add_argument(
        "--effort-meter", action="store_true",
        help="show an effort bar under the target row: overall EMG relative to the calibration "
             "maximum of what is prompted, with a target zone per level (from the first "
             "calibration block on; saved with each block)",
    )  # fmt: skip
    ap.add_argument(
        "--effort-zones", type=float, nargs=4, default=[*ZONES["low"], *ZONES["high"]],
        metavar=("LOW_MIN", "LOW_MAX", "HIGH_MIN", "HIGH_MAX"),
        help="target zones as fractions of the maximum (default: 0.2 0.4 0.65 0.85)",
    )  # fmt: skip
    ap.add_argument(
        "--replay", type=Path, nargs="+", metavar="PATH",
        help="replay recorded sessions / trials instead of the armband (no hardware)",
    )  # fmt: skip
    ap.add_argument(
        "--replay-speed", type=float, default=1.0,
        help="with --replay: 1 at the pace it was recorded, 2 twice as fast, ...",
    )  # fmt: skip
    ap.add_argument(
        "--viewer", action="store_true",
        help="also open the live viewer, on the right of the screen (the prompter on the left)",
    )  # fmt: skip
    ap.add_argument("--no-stream", action="store_true", help="don't publish over LSL")
    ap.add_argument("-v", "--verbose", action="store_true")
    subjects.add_arguments(ap)
    return ap


def main(argv: list[str] | None = None, extension: PrompterExtension | None = None) -> None:
    ext = extension or PrompterExtension()
    ap = parser()
    ext.add_arguments(ap)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    lo_min, lo_max, hi_min, hi_max = args.effort_zones
    if not 0 <= lo_min < lo_max <= hi_min < hi_max:
        sys.exit("--effort-zones: need 0 <= LOW_MIN < LOW_MAX <= HIGH_MIN < HIGH_MAX")
    if args.effort_meter and args.no_calibration:
        sys.exit("--effort-meter needs calibration blocks (its 100 %): drop --no-calibration")
    sequences = []
    for path in expand(args.sequences):
        try:
            seq = Sequence.load(path)
            seq.level_names  # noqa: B018 (raises for a sequence mixing level codings)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"{path}: not a usable sequence ({e})") from None
        sequences.append((path, seq))
    if not sequences:
        sys.exit("no sequence files found")
    if not args.no_calibration:  # per-session calibration first
        # like the first task block: maximum pinches before pinch blocks, squeezes before
        # squeeze blocks, presses otherwise (later ones match the block they come before)
        block = calibration_for(sequences[0][1].task_set, args.calibration_reps)
        sequences.insert(0, (Path(f"built-in/{block.name}.json"), block))
    simulated = args.simulate or bool(args.replay)  # a replay is no new donning either
    donning = subjects.donning_from_args(
        argparse.Namespace(**{**vars(args), "simulate": simulated}), args.data_dir
    )

    metadata = {"hand": args.hand, "arm": args.hand}  # band worn on the task hand's arm
    if simulated:
        metadata["simulated"] = True
    metadata.update(donning.metadata())
    config = SessionConfig(
        port=args.port,
        powerline_hz=args.powerline_hz,
        highpass_hz=args.highpass_hz,
        filter_enabled=args.device_filter,
        live_highpass_hz=None if args.live_highpass == "none" else float(args.live_highpass),
        live_notch=args.live_notch,
        data_dir=args.data_dir,
        session_name=args.session,
        live_stream=not args.no_stream,
        metadata=metadata,
    )
    device = None
    if args.simulate:
        from deftemg_acquisition.acquisition.sim import SimulatedArmband

        device = Armband("simulated", ser=SimulatedArmband())
    elif args.replay:
        from deftemg_acquisition.acquisition.replay import RecordedStream, ReplayArmband

        stream = RecordedStream.load(args.replay)
        device = ReplayArmband(stream, speed=args.replay_speed)
        config.filter_enabled = stream.filter_status.enabled
        config.metadata["replay"] = [str(p) for p in args.replay]
    plugins = ext.plugins(args, config)

    viewer = left = None
    viewer_args = ext.viewer_args(args)
    if viewer_args is None and args.viewer:
        viewer_args = []
    if viewer_args is not None and args.no_stream:
        print("--no-stream: no viewer (it reads the LSL stream)")
    elif viewer_args is not None:  # side by side: the prompter on the left, the viewer right
        from deftemg_acquisition.viz.screen import geometry_args, snap, split_screen

        pg.mkQApp("Prompter")
        left, right = split_screen()
        viewer = subprocess.Popen(
            [sys.executable, "-m", ext.viewer_module, "--arm", args.hand, *viewer_args,
             "--beside", *geometry_args(right)]
        )  # fmt: skip
    try:
        with AcquisitionSession(config, device=device) as session:
            print(f"Saving to {session.session_dir}")
            app = pg.mkQApp("Prompter")
            win = TaskWindow(
                session,
                sequences,
                hand=args.hand,
                lookahead_s=args.lookahead,
                sync_patch=args.sync_patch,
                recalibrate_s=None if args.no_calibration else [60 * m for m in args.recalibrate],
                calibration_reps=args.calibration_reps,
                effort_meter=args.effort_meter,
                effort_zones={"low": (lo_min, lo_max), "high": (hi_min, hi_max)},
                plugins=plugins,
            )
            win.show()
            if left is not None:
                snap(win, left)
            win.raise_()
            win.activateWindow()
            app.exec()
    finally:
        if viewer is not None:
            viewer.terminate()


if __name__ == "__main__":
    main()
