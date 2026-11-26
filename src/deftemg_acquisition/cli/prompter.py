"""Run the scrolling finger-press prompter, or text-prompt blocks, while recording EMG.

    pixi run prompter --port /dev/cu.usbmodem1101 --subject S01 --viewer protocols/demo
    pixi run prompter --simulate protocols/demo/d2_multi_b01.json
    pixi run prompter --remote <source_id or host> --viewer protocols/demo   # see below
    pixi run prompter --wuji protocols/G01/S01   # grasp and natural-task blocks, with the glove

Blocks are JSON files: lane sequences from ``make-sequences``, text blocks from
``make-task-sequences``; directories are expanded (sorted by name; a set's ``plan.json`` is
skipped). Lane and text blocks can be mixed. Keys in the prompter window: Shift start block,
Enter end early, Left/Right choose block, Esc quit; in a text block, Space starts a prompt's
countdown and marks its end, and r redoes the last prompt. Text blocks that carry terms open
with the subject's terms slide (Shift goes on); T shows it again between blocks. Each block is
saved when it ends; the window says "Saving..." until it is (with a glove, about 2 s per block
minute).

Asks for your user ID and, if you have recorded before, whether you re-donned the band since
your last session (a new donning id if yes; the same one if no). ``--subject`` and ``--redon`` /
``--same-donning`` answer in advance.

Each session starts with a calibration block (about 25 s): rest quietly, then one brief effort
as hard as is comfortable (``--calibration-reps``: how many), matching the task blocks: before
pinch or squeeze blocks, a pinch or squeeze of the thumb with each other finger (thumb implied,
as in those blocks); otherwise a press of each digit. It sets the session's noise and
maximum-effort levels (``deftemg_acquisition.data.calibration``: e.g. ``press_fraction`` gives
low / high presses as a fraction of the maximum of the same digit or pinch). Text blocks have
their own calibration, a brief maximum of each grasp, kept on its own schedule: a session with
both kinds of block has a calibration of each kind before the first block of that kind.
Another calibration block comes up ~10 minutes later, then every 20 minutes (``--recalibrate``:
minutes to the second block, then between later ones), before the next block once it is due,
so every trial has a recent calibration before it. ``--no-calibration`` skips them
all.

``--effort-meter`` adds an effort bar under the target row once a calibration block is done
(lane blocks only): overall EMG effort relative to the calibration maximum of what is prompted
(a digit, a pinch, or a chord's digits together), with a zone per level (``--effort-zones``,
default low 20-40 %, high 65-85 %). Keep the bar in the outlined zone while holding a press. Its
trace is saved with each block, and the trial's metadata records that the meter was on.

Session conditions (e.g. whether a water bottle is filled): text blocks name the ones they
need, and the prompter asks for them at the start (``--condition NAME=VALUE`` answers in
advance; without a terminal every one must be given). They are saved in the session's and those
blocks' metadata, and hold for the whole session: to change one, start a new session.

``--viewer`` opens the live viewer in its own window (and process, so it can't slow the
prompter), the two side by side at full height: the prompter on the left, the viewer on the
right. ``--no-stream`` opens no viewer (it reads the LSL stream).

``--replay PATH`` replays recorded sessions or trials instead of the armband
(``acquisition.armband.replay``), for trying a setup without hardware: it plays once, from when the
prompter opens, and the replayed EMG doesn't follow the prompts on screen.

``--remote SOURCE`` takes the armband plugged into another computer (``pixi run serve`` there;
``acquisition.remote``): its stream's ``source_id`` or that computer's host name. The session is
recorded here as with the armband plugged in, with the other computer's filter setting.

Other code can add options, task window plugins and viewer options (``main``'s ``extension``:
:class:`PrompterExtension`).
"""

from __future__ import annotations

import argparse
import difflib
import logging
import subprocess
import sys
from pathlib import Path

import pyqtgraph as pg

from deftemg_acquisition.acquisition import (
    AcquisitionSession,
    Armband,
    SessionConfig,
    subjects,
    wuji,
)
from deftemg_acquisition.acquisition.session import source_id_for
from deftemg_acquisition.cli.make_task_sequences import PLAN
from deftemg_acquisition.tasks.app import (
    TaskPlugin,
    TaskWindow,
    block_conditions,
    calibration_family,
)
from deftemg_acquisition.tasks.effort import ZONES
from deftemg_acquisition.tasks.prompter import DEFAULT_LOOKAHEAD_S
from deftemg_acquisition.tasks.sequence import CALIBRATION_REPS, Sequence
from deftemg_acquisition.tasks.textblock import TIMED, TextBlock, calibration_for_block, load_block
from deftemg_acquisition.viz.screen import fit_font

Blocks = list[tuple[Path, Sequence | TextBlock]]


def expand(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        # a generated set's plan (make-task-sequences) sits beside its blocks
        out += sorted(f for f in p.glob("*.json") if f.name != PLAN) if p.is_dir() else [p]
    return out


def with_calibrations(blocks: Blocks, reps: int = CALIBRATION_REPS) -> Blocks:
    """``blocks`` with a calibration before the first block of each kind (lane, text): like
    that block (``calibration_for_block``), so lane-only sessions start as they always have."""
    out: Blocks = []
    seen = set()
    for path, block in blocks:
        if calibration_family(block) not in seen:
            seen.add(calibration_family(block))
            cal = calibration_for_block(block, reps)
            out.append((Path(f"built-in/{cal.name}.json"), cal))
        out.append((path, block))
    return out


def condition_questions(blocks: Blocks) -> dict[str, dict | None]:
    """The session conditions the blocks use: {name: {"ask", "choices"[, "label"]}} from the
    blocks' ``params["condition_questions"]``, or None for one no block describes (any
    non-empty value is then accepted). Exits if one is described badly (``condition_questions``
    must be a table of conditions) or two blocks describe the same condition differently
    (their labels included)."""
    out: dict[str, dict | None] = {}
    source: dict[str, Path] = {}
    for path, block in blocks:
        described = block.params.get("condition_questions", {})
        if not isinstance(described, dict):
            sys.exit(f"{path}: condition_questions must be a table of conditions")
        for name in block_conditions(block):
            out.setdefault(name, None)
            q = described.get(name)
            if q is None:
                continue
            ok = isinstance(q, dict) and isinstance(q.get("ask"), str)
            choices = q.get("choices") if ok else None
            if not (isinstance(choices, list) and choices
                    and all(isinstance(c, str) and c for c in choices)):  # fmt: skip
                sys.exit(f"{path}: condition_questions.{name}: need an ask and choices")
            label = q.get("label")
            if label is not None and not (isinstance(label, str) and label.strip()):
                sys.exit(f"{path}: condition_questions.{name}.label: expected a non-empty string")
            # the label too: the screens show it, so blocks must agree on it
            q = {"ask": q["ask"], "choices": choices, **({"label": label} if label else {})}
            if out[name] is None:
                out[name], source[name] = q, path
            elif out[name] != q:
                sys.exit(f"condition {name}: {source[name]} and {path} ask it differently")
    return out


def ask_conditions(
    questions: dict[str, dict | None],
    given: dict[str, str],
    interactive: bool,
    ask=input,
    say=print,
) -> dict[str, str]:
    """The session's condition values: those ``given`` (``--condition``) for the conditions in
    ``questions``, the rest asked on the terminal. A given name no block uses is ignored, with
    a note (naming a close one in case of a typo, unless that one was given too); a value not
    among the choices exits, and so does a condition left unanswered without a terminal."""

    def bad(name: str, value: str) -> str | None:
        q = questions[name]
        if not value:
            return f"{name}: give a value"
        if q is not None and value not in q["choices"]:
            return f"{name}: {value!r} is not one of {', '.join(q['choices'])}"
        return None

    for name in given:
        if name not in questions:
            close = difflib.get_close_matches(name, list(questions), n=1, cutoff=0.85)
            hint = f" (did you mean {close[0]}?)" if close and close[0] not in given else ""
            say(f"--condition {name}: no block here uses it; ignored{hint}")
        elif why := bad(name, given[name]):
            sys.exit(f"--condition {why}")
    values = {k: v for k, v in given.items() if k in questions}
    missing = [k for k in questions if k not in values]
    if missing and not interactive:
        sys.exit("no terminal to ask: pass --condition NAME=VALUE for " + ", ".join(missing))
    try:
        for name in missing:
            q = questions[name]
            text = f"{q['ask']} [{'/'.join(q['choices'])}]: " if q else f"Condition {name}: "
            while (why := bad(name, value := ask(text).strip())) is not None:
                say(why)
            values[name] = value
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nno answer: stopping") from None
    return {k: values[k] for k in questions}


def _condition_arg(text: str) -> tuple[str, str]:
    name, eq, value = text.partition("=")
    if not eq or not name.strip():
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {text!r}")
    return name.strip(), value.strip()


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
    ap.add_argument(
        "sequences", type=Path, nargs="+", help="block files (lane or text) or directories"
    )
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--simulate", action="store_true", help="use a simulated armband")
    ap.add_argument(
        "--remote", metavar="SOURCE",
        help="the armband on another computer (pixi run serve there): its source_id or host "
             "name; its filter setting is used",
    )  # fmt: skip
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
    ap.add_argument(
        "--condition", type=_condition_arg, action="append", default=[], metavar="NAME=VALUE",
        help="a session condition the text blocks need (asked for if not given); repeatable",
    )  # fmt: skip
    ap.add_argument("--no-stream", action="store_true", help="don't publish over LSL")
    ap.add_argument("-v", "--verbose", action="store_true")
    subjects.add_arguments(ap)
    wuji.add_arguments(ap)
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
            seq = load_block(path)
            if isinstance(seq, Sequence):
                seq.level_names  # noqa: B018 (raises for a sequence mixing level codings)
            block_conditions(seq)  # raises for a malformed list
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"{path}: not a usable sequence ({e})") from None
        if isinstance(seq, TextBlock) and seq.mode == TIMED:  # before any device or folder opens
            raise SystemExit(f"{path}: a timed block can only be replayed, not run")
        sequences.append((path, seq))
    if not sequences:
        sys.exit("no sequence files found")
    if args.effort_meter and all(isinstance(b, TextBlock) for _, b in sequences):
        print("--effort-meter: ignored (it is for lane blocks; these are all text blocks)")
    given: dict[str, str] = {}
    for name, value in args.condition:
        if given.setdefault(name, value) != value:
            sys.exit(f"--condition {name}: given twice ({given[name]!r}, {value!r})")
    # checked and asked before the session opens: a missing answer records nothing
    conditions = ask_conditions(condition_questions(sequences), given, sys.stdin.isatty())
    if not args.no_calibration:  # per-session calibration first
        # like the first task block of its kind: maximum pinches before pinch blocks, squeezes
        # before squeeze blocks, presses otherwise, grasps before text blocks (later ones match
        # the block they come before)
        sequences = with_calibrations(sequences, args.calibration_reps)
    simulated = args.simulate or bool(args.replay)  # a replay is no new donning either
    donning = subjects.donning_from_args(
        argparse.Namespace(**{**vars(args), "simulate": simulated}), args.data_dir
    )

    metadata = {"hand": args.hand, "arm": args.hand}  # band worn on the task hand's arm
    if simulated:
        metadata["simulated"] = True
    metadata.update(donning.metadata())
    if conditions:
        metadata["conditions"] = conditions
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
        from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband

        device = Armband("simulated", ser=SimulatedArmband())
    elif args.replay:
        from deftemg_acquisition.acquisition.armband.replay import RecordedStream, ReplayArmband

        stream = RecordedStream.load(args.replay)
        device = ReplayArmband(stream, speed=args.replay_speed)
        config.filter_enabled = stream.filter_status.enabled
        config.metadata["replay"] = [str(p) for p in args.replay]
    elif args.remote:
        from deftemg_acquisition.acquisition.remote import RemoteArmband

        device = RemoteArmband(args.remote)
        config.filter_enabled = device.resolve().enabled  # before the plugins see the config
    sources = wuji.sources_from_args(args)
    plugins = ext.plugins(args, config)

    viewer = left = None
    viewer_args = ext.viewer_args(args)
    if viewer_args is None and args.viewer:
        viewer_args = []
    if viewer_args is not None and args.no_stream:
        print("--no-stream: no viewer (it reads the LSL stream)")
    elif viewer_args is not None:  # side by side: the prompter on the left, the viewer right
        if args.wuji:
            viewer_args = [*viewer_args, "--glove"]
        from deftemg_acquisition.viz.screen import geometry_args, snap, split_screen

        fit_font(pg.mkQApp("Prompter"))
        left, right = split_screen()
        # this session's streams, not another's of the same name (another computer's)
        source = ["--source", source_id_for(device.port)] if device is not None else []
        viewer = subprocess.Popen(
            [sys.executable, "-m", ext.viewer_module, "--arm", args.hand, *viewer_args,
             "--beside", *geometry_args(right), *source]
        )  # fmt: skip
    try:
        with AcquisitionSession(config, device=device, sources=sources) as session:
            print(f"Saving to {session.session_dir}")
            app = pg.mkQApp("Prompter")
            fit_font(app)  # before any window (once: kept if already done)
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
                conditions=conditions,
            )
            win.show()
            if left is not None:
                snap(win, left)
            win.raise_()
            win.activateWindow()
            app.exec()
    except KeyboardInterrupt:
        sys.exit(130)  # Ctrl+C during calibration: the session has already shut down
    finally:
        if viewer is not None:
            viewer.terminate()


if __name__ == "__main__":
    main()
