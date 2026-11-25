"""Task window: runs a list of prompt sequences as trials of an :class:`AcquisitionSession`.

Keys (the window must have focus):
    Shift        start the current block (records one trial)
    Enter        end the running block early (saved, marked incomplete; the block stays current)
    Left/Right   choose the previous/next block while idle
    Esc          quit (ends a running block first)
    Space, r     text blocks only: see below
The letters s / e / q do the same as Shift / Enter / Esc.

A block ends by itself ``tail_s`` after its last prompt and the next block becomes current.

Text blocks (``textblock.TextBlock``, run by ``selfpaced.SelfPacedRunner``) are drawn in the
prompter's text mode and saved with ``task="text_prompter"`` and ``text_*`` arrays (no
``prompt_*`` arrays, ``scroll_start_host_ns`` or effort trace). Space starts each prompt's
countdown (``countdown_s``) and marks its end (a prompt with a fixed duration, such as a grasp
calibration maximum, also ends by itself), and ``r`` redoes the last prompt; after the last
end, Space (or Enter or Esc, which then save the block as complete; Esc also quits) finishes
the block ``tail_s`` later. Closing the window then saves it as complete at once, with no tail
(Esc waits ``tail_s`` first). Enter before that ends the block early, as for a lane block.
Timed text blocks (the old grasp calibration) are only replayed: the window rejects them.
Markers: ``text|preview|i``, ``text|countdown|i``,
``text|go|i|<label>``, ``text|end|i``; the sync patch flashes at each Go.

Resume: Shift on a self-paced text block that was ended early starts a new trial at the prompt
after the last one with an "ok" attempt (prompt order is fixed, so those form a prefix), or where
that run started if it had none; the idle screen names that prompt and counts only the prompts
left. Each text block trial's metadata has ``text_resume_from`` (the first prompt's index, 0 for
a fresh run) and, for a resumed run, ``text_resumed_trial`` (the npz file name, without
extension, of the run just before; set even when that run had no "ok" attempt), so a block's
trials form a chain; ``text_prompt_index`` stays an index into the whole block. A block that
completes forgets where it stopped, so redoing a done block starts at its first prompt. The
grasp calibration always starts at its first prompt. Resume state lives in the window only:
after a restart a block starts at its first prompt.

Recalibration (``recalibrate_s``, e.g. [600, 1200]) keeps two clocks: ``lane`` for the press,
pinch and squeeze calibrations (before lane blocks) and ``grasp`` for the grasp calibration
(before text blocks); one kind never counts for the other. Once a calibration block of a kind
has been done, the next one of that kind is due ``recalibrate_s[0]`` s after it started (a
grasp calibration: at its first Go, since it may wait at its first preview), later ones every
``recalibrate_s[-1]`` s. A text block also needs a grasp calibration before it if
none has been done yet this session (a lane block does not: the session's first lane
calibration is put in by whoever builds the list, so a skipped or early-ended one forces
nothing). A due calibration block (``textblock.calibration_for_block``: like the block it comes
before) is put in front of the current block when the window opens, when a block finishes,
or when Shift is pressed after a pause (the screen then shows it; Shift again starts it).
Without ``recalibrate_s`` nothing is put in.

Effort meter (``effort_meter=True``, ``tasks.effort``): a bar under the target row shows the
overall EMG effort relative to the calibration maximum of what is prompted, from the first
completed lane calibration block of the session on (each later one replaces the reference; a
grasp calibration never does). Lane task blocks (not text blocks) save its trace (``effort``,
``effort_host_ns``, ``effort_max_uv``, ``effort_level``: -1 between presses, else the index
into ``level_names``) and the settings (``effort_meter`` in the trial's metadata).

A session source that reports a problem with a block (``ok: false`` in the trial's ``sources``,
e.g. a glove that sent nothing) is named on the screen between blocks, until the next block.
The sources are also checked every ``SOURCE_CHECK_MS``, between blocks and during them
(``AcquisitionSession.source_status``); a problem shows on a band under the header until it
clears.

Session conditions (``conditions``: {name: value}, e.g. how a bottle is filled): a text block
names the ones it needs in ``params["conditions"]``; those (and only those) go into its trial's
metadata as ``conditions`` and show on the screen before and during it. A block needing one
that is not given is an error when the window is made, before anything is recorded.

Plugins (``plugins``: :class:`TaskPlugin`) add to the window from other code, e.g. a live display
of a model's output: text between blocks, metadata and arrays saved with each block, a hook
each frame tick. Once attached they can't stop a recording: a hook that raises is logged and
skipped (``attach`` runs before anything is recorded, and its errors stop the window). Hooks get
the block as it is in the list: a ``Sequence`` or a ``TextBlock`` (no ``prompts`` with digits),
so a hook that assumes lane fields fails on a text block and is skipped like any other.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from deftemg_acquisition.acquisition import AcquisitionSession
from deftemg_acquisition.clock import host_ns

from .effort import ZONES, EffortMeter, prompt_level, reference_from_trial
from .prompter import DEFAULT_LOOKAHEAD_S, PrompterWidget
from .selfpaced import GO_DEBOUNCE_S, IDLE, SelfPacedRunner, TextView
from .sequence import (
    CALIBRATION,
    CALIBRATION_REPS,
    Sequence,
    active_label,
)
from .textblock import COUNTDOWN_S, TASK, TIMED, TextBlock, calibration_for_block

log = logging.getLogger(__name__)

# Tick well above the display rate so a repaint is always pending at the next screen refresh;
# Qt coalesces update() calls, so this does not add paints. Prompt positions are computed at
# paint time (see PrompterWidget.t0_perf), so ticks only request repaints and send markers.
FRAME_MS = 4
SOURCE_CHECK_MS = 1000
# The calibration clocks: lane calibrations (press, pinch, squeeze) and the grasp calibration
LANE, GRASP = "lane", "grasp"


def calibration_family(block: Sequence | TextBlock) -> str:
    """Which calibration clock a block (task or calibration) belongs to."""
    return GRASP if isinstance(block, TextBlock) else LANE


def block_conditions(block: Sequence | TextBlock) -> list[str]:
    """The session conditions a block needs: a text block's ``params["conditions"]``."""
    if not isinstance(block, TextBlock):
        return []
    names = block.params.get("conditions") or []
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{block.name}: params.conditions must be a list of names, got {names!r}")
    return names


@dataclass(frozen=True)
class ResumePoint:
    """Where an early-ended self-paced text block resumes: ``resume_index`` (the first prompt of
    the next run) and ``last_trial`` (the npz file's stem of the run that ended early). Kept
    with the block object itself, so its key (``id(block)``) can't point at another block."""

    block: TextBlock
    resume_index: int
    last_trial: str


def source_note(sources: dict) -> str:
    """Text for the screen between blocks naming the sources that failed in a trial (its
    ``TrialResult.sources``); empty if none did."""
    failed = [f"{name}: {meta.get('error', 'failed')}"
              for name, meta in sources.items() if meta.get("ok") is False]  # fmt: skip
    if not failed:
        return ""
    return "Last block saved without:\n" + "\n".join(failed) + "\n\n"


class TaskPlugin:
    """Adds to a :class:`TaskWindow` (its ``plugins``). Every hook is optional. ``sequence`` is
    the block: a ``Sequence``, or a ``TextBlock`` for a text block (check before using lane
    fields such as ``prompts[i].digits``)."""

    def attach(self, window: TaskWindow) -> None:
        """Once, at the end of the window's set-up (the session is open, nothing shown yet);
        unlike the other hooks, an error here is raised."""

    def instructions(self, sequence: Sequence | TextBlock) -> str:
        """More text for the screen between blocks, after the effort bar's note."""
        return ""

    def block_started(self, sequence: Sequence | TextBlock, metadata: dict) -> None:
        """A block is about to start recording; ``metadata`` goes into its trial's metadata."""

    def block_ending(self, sequence: Sequence | TextBlock, arrays: dict) -> None:
        """The running block is about to be saved; ``arrays`` go into its file."""

    def block_ended(self, sequence: Sequence | TextBlock, completed: bool) -> None:
        """The block has been saved (``completed``: it ran to its end, not ended early); the
        window then moves on and shows the screen between blocks."""

    def tick(self, t: float) -> None:
        """While a block runs, every ``FRAME_MS`` (``t``: s since it started), before the
        prompter is asked to repaint."""

    def close(self) -> None:
        """The window is closing (any running block has been saved)."""


class TaskWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        session: AcquisitionSession,
        sequences: list[tuple[Path, Sequence | TextBlock]],
        hand: str = "right",
        lookahead_s: float = DEFAULT_LOOKAHEAD_S,
        tail_s: float = 1.5,
        sync_patch: bool = False,
        recalibrate_s: list[float] | None = None,
        calibration_reps: int = CALIBRATION_REPS,
        effort_meter: bool = False,
        effort_zones: dict[str, tuple[float, float]] | None = None,
        plugins: list[TaskPlugin] | tuple[TaskPlugin, ...] = (),
        countdown_s: float = COUNTDOWN_S,
        go_debounce_s: float = GO_DEBOUNCE_S,
        conditions: dict[str, str] | None = None,
    ):
        super().__init__()
        if not sequences:
            raise ValueError("no sequences to run")
        self.conditions = dict(conditions or {})
        for _, block in sequences:  # fail now, not when the block comes up mid-session
            if isinstance(block, TextBlock) and block.mode == TIMED:
                raise ValueError(f"{block.name}: a timed text block can only be replayed, not run")
            missing = [c for c in block_conditions(block) if c not in self.conditions]
            if missing:
                given = ", ".join(self.conditions) or "none"
                raise ValueError(f"{block.name}: needs the session condition(s) "
                                 f"{', '.join(missing)} (given: {given})")  # fmt: skip
        self.session = session
        self.sequences = sequences
        self.hand = hand
        self.lookahead_s = lookahead_s
        self.tail_s = tail_s
        self.current = 0
        self.completed: set[int] = set()
        self._t0_perf: float | None = None
        self._t0_ns = 0
        self._last_label: str | None = None
        self.recalibrate_s = list(recalibrate_s or [])
        self.calibration_reps = calibration_reps
        self._now = time.monotonic  # replaceable in tests
        self._block_start = 0.0
        # per clock (LANE, GRASP): start of the last completed calibration, and how many
        self._last_calibration: dict[str, float] = {}
        self._n_calibrations = {LANE: 0, GRASP: 0}
        self._due_note = ""
        self._due_family = LANE  # the kind of calibration the note is about
        self._source_note = ""  # the sources that failed in the last block
        # Text blocks: the countdown before Go, Space's debounce, and the running block
        self.countdown_s = countdown_s
        self.go_debounce_s = go_debounce_s
        self._host_ns = host_ns  # the text runner's clock (int ns); replaceable in tests
        self._runner: SelfPacedRunner | None = None
        self._finish_at_ns: int | None = None  # a finished text block is saved at this time
        self._close_after = False  # Esc finished a text block: quit once it is saved
        self._saving = False  # a block is being saved: keys are dropped
        # Early-ended self-paced text blocks, keyed by the block's identity (not its index:
        # calibration blocks are put in and taken out, shifting the indices)
        self._resume: dict[int, ResumePoint] = {}

        self.setWindowTitle("Finger-press prompter")
        self.sync_patch = sync_patch
        self.prompter = PrompterWidget(hand, lookahead_s, sync_patch)
        self.setCentralWidget(self.prompter)
        self.effort_meter: EffortMeter | None = None
        self._effort_log: list[tuple[int, float, float, int]] = []
        if effort_meter:
            status = session.filter_status
            self.effort_meter = EffortMeter(status.enabled if status else True,
                                            session.live_emg_filter())  # fmt: skip
            self.effort_meter.start()
            session.device.attach_sink("effort", self.effort_meter)
            self.prompter.show_effort = True
            self.prompter.effort_zones = dict(effort_zones or ZONES)
        self.plugins = list(plugins)
        self._tick_failed: set[int] = set()  # plugins whose tick failed (logged once)
        for plugin in self.plugins:
            plugin.attach(self)
        # Tall, narrow window: long lanes show several upcoming prompts at once.
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        height = int(screen.height() * 0.95)
        self.resize(min(int(height * 0.62), screen.width()), height)
        self._timer = QtCore.QTimer(self, timeout=self._tick, interval=FRAME_MS)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._source_timer = QtCore.QTimer(
            self, timeout=self.check_sources, interval=SOURCE_CHECK_MS
        )
        self._source_timer.start()
        self._schedule_calibration()  # a first text block: its grasp calibration first
        self._show_idle()

    # --- state -------------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._t0_perf is not None

    @property
    def sequence(self) -> Sequence | TextBlock:
        return self.sequences[self.current][1]

    @property
    def t0_ns(self) -> int:
        """Host time (ns) of the running (or last) block's t = 0, the EMG timestamps' clock."""
        return self._t0_ns

    def _plugins(self, hook: str, *args) -> list:
        """Call ``hook`` on each plugin; one that raises is logged and skipped (``tick``, every
        frame: logged the first time only)."""
        out = []
        for plugin in self.plugins:
            try:
                out.append(getattr(plugin, hook)(*args))
            except Exception:  # an add-on must never stop a recording
                if hook != "tick" or id(plugin) not in self._tick_failed:
                    log.exception("task window plugin %r: %s failed", plugin, hook)
                if hook == "tick":
                    self._tick_failed.add(id(plugin))
        return out

    def check_sources(self) -> None:
        """Show the sources' current problems on the warning band (cleared once they clear)."""
        problems = self.session.source_status()
        warning = "   ".join(f"{name}: {problem}" for name, problem in problems.items())
        if warning == self.prompter.warning:
            return
        if warning:
            log.warning("source problem: %s", warning)
        else:
            log.info("sources fine again")
        self.prompter.warning = warning
        self.prompter.update()

    def _show_idle(self) -> None:
        if self._due_note and calibration_family(self.sequence) != self._due_family:
            self._due_note = ""  # moved to a block of the other kind: not its calibration
        self.prompter.sequence = self.sequence
        self.prompter.text_view = None  # never left over from a text block
        self.prompter.effort = None
        self.prompter.end_run()
        n = len(self.sequences)
        done = f"{len(self.completed)}/{n} blocks done"
        self.prompter.header = f"Block {self.current + 1}/{n}: {self.sequence.name}   ({done})"
        if len(self.completed) == n:
            # a done block that was redone and ended early: say so, Shift goes on with it
            line = self._resume_line(self.sequence)
            redo = f"{self.sequence.name}: {line}, Shift to go on.\n" if line else ""
            self.prompter.message = self._source_note + (
                "All blocks done.\n" + redo
                + "Esc to quit, or Left/Right then Shift to redo a block."
            )  # fmt: skip
        else:
            resumes = self.resume_point(self.sequence) is not None  # the summary says so
            again = "  (already done)" if self.current in self.completed and not resumes else ""
            how = self.sequence.params.get("instructions")
            what = (self._text_summary() if isinstance(self.sequence, TextBlock) else
                    f"{len(self.sequence.prompts)} presses, "
                    f"{self.sequence.duration_s + self.tail_s:.0f} s\n")  # fmt: skip
            self.prompter.message = (
                self._source_note
                + self._due_note
                + f"{self.sequence.name}{again}\n{what}"
                + (f"\n{how}\n" if how else "")
                + self._effort_note()
                + "".join(t for t in self._plugins("instructions", self.sequence) if t)
                + "\nPress Shift to start"
            )
        self.prompter.update()

    def _text_summary(self) -> str:
        """The idle screen's lines for a text block: the object to place, the prompts and the
        estimated length."""
        block = self.sequence
        place = block.params.get("object_name")
        n = len(block.prompts)
        resume = self.resume_point(block)
        if resume is None:
            length = f"{n} prompts, about {max(1, round(block.estimate_s() / 60))} min"
        else:  # only what is left
            k = resume.resume_index
            left = n - k
            length = (f"{self._resume_line(block)}\n{left} prompt{'s' if left != 1 else ''} "
                      f"left, about {max(1, round(block.estimate_s(k) / 60))} min")  # fmt: skip
        conds = self._conditions_line(block)
        return ((f"Place: {place}\n" if place else "") + length + "\n"
                + (f"Conditions: {conds}\n" if conds else ""))  # fmt: skip

    def _resume_line(self, block: Sequence | TextBlock) -> str:
        """Where the current ``block`` resumes, for the idle screen; empty if it doesn't."""
        resume = self.resume_point(block)
        if resume is None:
            return ""
        where = f"prompt {resume.resume_index + 1} of {len(block.prompts)}"
        done = self.current in self.completed
        return f"Done before; redo resumes at {where}" if done else f"Resumes at {where}"

    def resume_point(self, block: Sequence | TextBlock) -> ResumePoint | None:
        """Where ``block`` resumes if it was ended early (self-paced text blocks only), else
        None (it starts at its first prompt)."""
        point = self._resume.get(id(block))
        return point if point is not None and point.block is block else None

    def _note_resume(self, block: TextBlock, completed: bool, resume_index: int,
                     npz_path: str) -> None:  # fmt: skip
        """A text block's run was saved: remember where it resumes, or forget it once done."""
        if completed or block.is_calibration:
            self._resume.pop(id(block), None)
        else:
            self._resume[id(block)] = ResumePoint(block, resume_index, Path(npz_path).stem)

    def _block_conditions(self, block: Sequence | TextBlock) -> dict[str, str]:
        """The session conditions ``block`` uses, with their values."""
        return {name: self.conditions[name] for name in block_conditions(block)}

    def _conditions_line(self, block: Sequence | TextBlock) -> str:
        return ", ".join(f"{k} = {v}" for k, v in self._block_conditions(block).items())

    def _effort_note(self) -> str:
        if isinstance(self.sequence, TextBlock):  # no effort bar on text blocks
            return ""
        if not self.prompter.show_effort or self.sequence.task_set == CALIBRATION:
            return ""
        if self.effort_meter.reference is None:
            return "\nThe effort bar comes on after a calibration block.\n"
        return ("\nKeep the effort bar in the outlined zone (low or high) while you hold each "
                "press.\n")  # fmt: skip

    def _effort_settings(self) -> dict:
        m = self.effort_meter
        return {
            "zones": {lv: list(z) for lv, z in self.prompter.effort_zones.items()},
            "window_s": m.window_s,
            "reference": m.reference.calibration_id if m.reference else None,
            "filter": m.emg_filter.describe() if m.emg_filter else "the armband's",
        }

    def _calibration_due(self, family: str) -> str | None:
        """Why a calibration of ``family`` (LANE or GRASP) is due, or None if none is."""
        if not self.recalibrate_s:
            return None
        last = self._last_calibration.get(family)
        if last is None:  # lane: none until the first is done (as before); grasp: one first
            return "no calibration of this kind yet" if family == GRASP else None
        gap = self.recalibrate_s[min(self._n_calibrations[family] - 1, len(self.recalibrate_s) - 1)]
        since = self._now() - last
        return f"{since / 60:.0f} min since the last one" if since >= gap else None

    def _schedule_calibration(self) -> bool:
        """Put a calibration block in front of the current block if one of its kind is due."""
        if self.running or self.sequence.is_calibration:
            return False
        family = calibration_family(self.sequence)
        why = self._calibration_due(family)
        if why is None:
            return False
        if family == GRASP:  # skipped or ended-early grasp calibrations: replaced, not added to
            self._drop_pending_grasp_calibrations()
        at = self.current
        # like the block it comes before: pinches before pinch blocks, grasps before text blocks
        block = calibration_for_block(self.sequences[at][1], self.calibration_reps)
        self.sequences.insert(at, (Path(f"built-in/{block.name}.json"), block))
        self.completed = {i + 1 if i >= at else i for i in self.completed}
        self._due_note = f"Calibration due ({why})\n\n"
        self._due_family = family
        log.info("calibration block %s due: %s", block.name, why)
        return True

    def _drop_pending_grasp_calibrations(self) -> None:
        """Take out the grasp calibration blocks not yet completed (skipped, or ended early),
        so a due one is put in once, not again on each Shift; ``current`` keeps its block."""
        drop = {
            i
            for i, (_, b) in enumerate(self.sequences)
            if i not in self.completed and isinstance(b, TextBlock) and b.is_calibration
        }
        if not drop:
            return
        keep = [i for i in range(len(self.sequences)) if i not in drop]
        new = {old: k for k, old in enumerate(keep)}
        self.sequences[:] = [self.sequences[i] for i in keep]
        self.completed = {new[i] for i in self.completed}
        self.current = new[self.current]

    def start_block(self) -> None:
        if self.running:
            return
        if self._schedule_calibration():  # after a pause: show it first
            self._show_idle()
            return
        self._due_note = ""
        self._block_start = self._now()
        path, seq = self.sequences[self.current]
        text = isinstance(seq, TextBlock)
        extra = {}
        if self.prompter.show_effort and not text and seq.task_set != CALIBRATION:
            extra["effort_meter"] = self._effort_settings()  # (off after a failure)
        if block_conditions(seq):
            extra["conditions"] = self._block_conditions(seq)
        self._plugins("block_started", seq, extra)
        self._effort_log = []
        if text:
            self._start_text_block(path, seq, extra)
            return
        self.session.start_trial(
            label=seq.name,
            task="scroll_prompter",
            task_set=seq.task_set,
            sequence_file=str(path),
            sequence=seq.to_dict(),
            hand=self.hand,
            lookahead_s=self.lookahead_s,
            sync_patch=self.sync_patch,
            **extra,
        )
        # Prompt times are relative to this instant (host clock, same as the EMG timestamps).
        self._t0_ns = host_ns()
        self._t0_perf = time.perf_counter()
        self.prompter.begin_run(self._t0_perf)
        self._last_label = None
        self.prompter.message = ""
        self.prompter.header = f"Block {self.current + 1}/{len(self.sequences)}: {seq.name}"
        self._timer.start()
        self._tick()

    def _start_text_block(self, path: Path, block: TextBlock, extra: dict) -> None:
        resume = self.resume_point(block)
        start = resume.resume_index if resume is not None else 0
        if resume is not None:
            extra = {**extra, "text_resumed_trial": resume.last_trial}

        def runner() -> SelfPacedRunner:
            return SelfPacedRunner(block, self.countdown_s, self.go_debounce_s,
                                   start_index=start, conditions=self._conditions_line(block),
                                   host_ns=self._host_ns)  # fmt: skip

        # Built once before the trial starts, so a bad start index or setting raises with no
        # trial open, and again after it, so the block starts (its first preview) once
        # recording has begun
        runner()
        self.session.start_trial(
            label=block.name,
            task=TASK,
            task_set=block.task_set,
            mode=block.mode,
            sequence_file=str(path),
            sequence=block.to_dict(),
            hand=self.hand,
            tail_s=self.tail_s,
            countdown_s=self.countdown_s,
            go_debounce_s=self.go_debounce_s,
            sync_patch=self.sync_patch,
            text_resume_from=start,
            **extra,
        )
        self._runner = runner()
        self._t0_ns = self._runner.start_ns  # block_start_host_ns
        self._finish_at_ns = None
        self._close_after = False
        self._t0_perf = time.perf_counter()
        self.prompter.text_view = self._runner.view()
        self.prompter.begin_run(self._t0_perf)
        self.prompter.message = ""
        self.prompter.header = f"Block {self.current + 1}/{len(self.sequences)}: {block.name}"
        self._timer.start()
        self._tick()

    def _text_arrays(self) -> tuple[dict, bool]:
        """The running text block's arrays, and whether it is complete (its runner decides:
        ended early, or finished)."""
        completed = self._runner.end_block()  # stops an attempt under way; no-op once finished
        for marker in self._runner.markers():  # its end, and any event since the last tick
            self.session.push_marker(marker)
        arrays = self._runner.arrays(self.prompter.go_paint_ns)
        if self.sync_patch:
            arrays["sync_flash_paint_ns"] = np.array(self.prompter.flash_paint_ns, np.int64)
        self._runner = None
        self._finish_at_ns = None
        return arrays, completed

    def end_block(self, completed: bool) -> None:
        """Save the running block (``completed``: it ran to its end). A text block's runner
        decides instead: complete once finished, else ended early."""
        if not self.running:
            return
        self._timer.stop()
        elapsed = time.perf_counter() - self._t0_perf
        self._t0_perf = None
        if self._runner is not None:
            self._end_text_block(elapsed)
            return
        arrays = self.sequence.prompt_arrays()
        arrays["scroll_start_host_ns"] = np.int64(self._t0_ns)
        # When each prompt was actually drawn reaching the target / gone (-1: never drawn,
        # e.g. the block was ended early). Display delay comes on top of these.
        arrays["prompt_onset_paint_ns"] = self.prompter.onset_paint_ns.copy()
        arrays["prompt_offset_paint_ns"] = self.prompter.offset_paint_ns.copy()
        if self.sync_patch:
            arrays["sync_flash_paint_ns"] = np.array(self.prompter.flash_paint_ns, np.int64)
        arrays["prompt_onset_host_ns"] = self._t0_ns + np.round(
            arrays["prompt_onset_s"] * 1e9
        ).astype(np.int64)
        arrays["prompt_offset_host_ns"] = self._t0_ns + np.round(
            arrays["prompt_offset_s"] * 1e9
        ).astype(np.int64)
        if self._effort_log:
            t_ns, value, top, level = zip(*self._effort_log, strict=True)
            arrays["effort_host_ns"] = np.array(t_ns, np.int64)
            arrays["effort"] = np.array(value, np.float32)
            arrays["effort_max_uv"] = np.array(top, np.float32)
            arrays["effort_level"] = np.array(level, np.int8)
        seq = self.sequence
        self._plugins("block_ending", seq, arrays)
        result = self._save(
            arrays=arrays,
            completed=completed,
            scroll_start_host_ns=self._t0_ns,
            elapsed_s=round(elapsed, 3),
        )
        self._after_block(result, seq, completed)

    def _end_text_block(self, elapsed: float) -> None:
        runner = self._runner
        go_ns = next((a.go_ns for a in runner.attempts if a.go_ns >= 0), None)
        if go_ns is not None:  # the recalibration clock starts at the first Go, not at Shift
            self._block_start += (go_ns - runner.start_ns) / 1e9
        arrays, completed = self._text_arrays()
        block = self.sequence
        self._plugins("block_ending", block, arrays)
        result = self._save(arrays=arrays, completed=completed, elapsed_s=round(elapsed, 3))
        self._note_resume(block, completed, runner.resume_index(), result.npz_path)
        self._after_block(result, block, completed)
        if self._close_after:
            self._close_after = False
            self.close()

    def _save(self, **trial):
        """Save the block, showing "Saving..." until the save returns. Keys pressed meanwhile
        are dropped: they are delivered once the save is done, while still marked as saving,
        so a Shift or Esc pressed at the saving screen starts or quits nothing."""
        self._saving = True
        try:
            self._show_saving()
            return self.session.end_trial(**trial)
        finally:
            QtWidgets.QApplication.processEvents()  # the keys queued during the save
            self._saving = False

    def _show_saving(self) -> None:
        """Paint "Saving..." now (a glove's arrays take a while: about 2 s per block minute),
        and process the events it needs, since the save blocks the event loop. Key presses
        stay queued until the save is done."""
        self.prompter.end_run()  # the block's paint times are in its arrays already
        if self.prompter.text_mode:
            self.prompter.text_view = TextView(IDLE)
        self.prompter.message = "Saving..."
        self.prompter.repaint()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    def _after_block(self, result, seq: Sequence | TextBlock, completed: bool) -> None:
        """A block has been saved: note failed sources, move on, show the idle screen."""
        self._source_note = source_note(result.sources)
        self._plugins("block_ended", seq, completed)
        if completed:
            if self.sequence.is_calibration:  # restarts its own kind's clock only
                family = calibration_family(self.sequence)
                self._last_calibration[family] = self._block_start
                self._n_calibrations[family] += 1
                if family == LANE and self.effort_meter is not None:
                    self._set_effort_reference()
            self.completed.add(self.current)
            self.current = min(self.current + 1, len(self.sequences) - 1)
            pending = [i for i in range(len(self.sequences)) if i not in self.completed]
            if pending and self.current in self.completed:
                self.current = pending[0]
            if pending:
                self._schedule_calibration()
        self._show_idle()

    def _set_effort_reference(self) -> None:
        """The just-saved lane calibration block becomes the effort meter's 100 %."""
        last = self.session.trials[-1]
        ref = reference_from_trial(Path(last.npz_path), self.effort_meter.emg_filter)
        if ref is not None:
            self.effort_meter.reference = ref
            log.info("effort meter: maxima from %s: %s", ref.calibration_id,
                     {k: round(v, 1) for k, v in ref.efforts_uv.items()})  # fmt: skip

    def _update_effort(self, t: float) -> None:
        """Effort for the prompt being held, or the next one between presses (rest reads ~0)."""
        meter, seq = self.effort_meter, self.sequence
        held = [pr for pr in seq.prompts if pr.onset_s <= t < pr.offset_s]
        later = [pr for pr in seq.prompts if pr.onset_s > t]
        pr = held[0] if held else min(later, key=lambda q: q.onset_s) if later else (
            seq.prompts[-1] if seq.prompts else None)  # fmt: skip
        level = prompt_level(pr.levels) if held else None
        if meter.reference is None or pr is None:
            self.prompter.effort = None
            return
        value = meter.effort(pr.digits)
        self.prompter.effort = (value, level)
        top = meter.reference.maximum_uv(pr.digits)
        code = seq.level_names.index(level) if level in seq.level_names else -1
        self._effort_log.append((host_ns(), np.nan if value is None else value,
                                 np.nan if top is None else top, code))  # fmt: skip

    def _tick(self) -> None:
        if self._runner is not None:
            self._plugins("tick", (self._host_ns() - self._t0_ns) / 1e9)
            self._text_step()
            return
        t = time.perf_counter() - self._t0_perf
        if self.prompter.show_effort and self.sequence.task_set != CALIBRATION:
            try:
                self._update_effort(t)
            except Exception:  # an optional display must never stop a recording
                log.exception("effort meter failed; it stays off for this session")
                self.prompter.effort = None
                self.prompter.show_effort = False
        self._plugins("tick", t)
        self.prompter.update()
        self._send_prompt_markers(t)
        if t > self.sequence.duration_s + self.tail_s:
            self.end_block(completed=True)

    def _text_step(self) -> None:
        """Advance the running text block: apply its due events, show it, send its markers, and
        save it ``tail_s`` after it finished."""
        runner = self._runner
        runner.update()
        now = self._host_ns()
        if runner.finished and runner.completed and self._finish_at_ns is None:
            self._finish_at_ns = now + round(self.tail_s * 1e9)
        # A finished block shows nothing while its tail is recorded
        self.prompter.text_view = TextView(IDLE) if runner.finished else runner.view()
        for marker in runner.markers():
            self.session.push_marker(marker)
        self.prompter.update()
        if self._finish_at_ns is not None and now >= self._finish_at_ns:
            self.end_block(completed=True)

    def _text_key(self, key) -> bool:
        """Space, r, Enter and Esc while a text block runs; returns whether it was handled."""
        Qt = QtCore.Qt
        runner = self._runner
        if key == Qt.Key_Space:
            runner.space()
        elif key == Qt.Key_R:
            runner.redo()
        elif key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_E, Qt.Key_Escape, Qt.Key_Q):
            quit_ = key in (Qt.Key_Escape, Qt.Key_Q)
            if not runner.end_block():  # ended early: saved now (Esc: and quit)
                if quit_:
                    self.close()
                else:
                    self.end_block(completed=False)
                return True
            self._close_after |= quit_  # after the last end: saved complete after the tail
        else:
            return False
        self._text_step()
        return True

    def _send_prompt_markers(self, t: float) -> None:
        label = active_label(self.sequence, t)
        if label != self._last_label:
            self.session.push_marker(f"prompt|{label or 'rest'}")
            self._last_label = label

    # --- input -------------------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        Qt = QtCore.Qt
        if event.isAutoRepeat() or self._saving:
            return
        if self._runner is not None and self._text_key(key):
            return
        if key in (Qt.Key_Shift, Qt.Key_S):
            self.start_block()
        elif key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_E):
            self.end_block(completed=False)
        elif key in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        elif not self.running and key in (Qt.Key_Left, Qt.Key_Right):
            step = 1 if key == Qt.Key_Right else -1
            self.current = (self.current + step) % len(self.sequences)
            self._show_idle()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._source_timer.stop()
        self._close_after = False  # closing already
        self.end_block(completed=False)
        if self.effort_meter is not None:
            self.session.device.detach_sink("effort")
            self.effort_meter.stop()
        self._plugins("close")
        super().closeEvent(event)
