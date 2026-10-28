"""Task window: runs a list of prompt sequences as trials of an :class:`AcquisitionSession`.

Keys (the window must have focus):
    Shift        start the current block (records one trial)
    Enter        end the running block early (saved, marked incomplete; the block stays current)
    Left/Right   choose the previous/next block while idle
    Esc          quit (ends a running block first)
The letters s / e / q do the same as Shift / Enter / Esc.

A block ends by itself ``tail_s`` after its last prompt and the next block becomes current.

Recalibration (``recalibrate_s``, e.g. [600, 1200]): once a calibration block has been done, the
next one is due ``recalibrate_s[0]`` s after it started, later ones every ``recalibrate_s[-1]``
s. A due calibration block is put in front of the current block when a block finishes, or
when Shift is pressed after a pause (the screen then shows it; Shift again starts it).

Effort meter (``effort_meter=True``, ``tasks.effort``): a bar under the target row shows the
overall EMG effort relative to the calibration maximum of what is prompted, from the first
completed calibration block of the session on (each later one replaces the reference). Task
blocks save its trace (``effort``, ``effort_host_ns``, ``effort_max_uv``, ``effort_level``:
-1 between presses, else the index into ``level_names``) and the settings (``effort_meter`` in
the trial's metadata).

Plugins (``plugins``: :class:`TaskPlugin`) add to the window from other code, e.g. a live display
of a model's output: text between blocks, metadata and arrays saved with each block, a hook
each frame tick. Once attached they can't stop a recording: a hook that raises is logged and
skipped (``attach`` runs before anything is recorded, and its errors stop the window).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from deftemg_acquisition.acquisition import AcquisitionSession
from deftemg_acquisition.clock import host_ns

from .effort import ZONES, EffortMeter, prompt_level, reference_from_trial
from .prompter import DEFAULT_LOOKAHEAD_S, PrompterWidget
from .sequence import (
    CALIBRATION,
    CALIBRATION_REPS,
    Sequence,
    active_label,
    calibration_for,
)

log = logging.getLogger(__name__)

# Tick well above the display rate so a repaint is always pending at the next screen refresh;
# Qt coalesces update() calls, so this does not add paints. Prompt positions are computed at
# paint time (see PrompterWidget.t0_perf), so ticks only request repaints and send markers.
FRAME_MS = 4


class TaskPlugin:
    """Adds to a :class:`TaskWindow` (its ``plugins``). Every hook is optional."""

    def attach(self, window: TaskWindow) -> None:
        """Once, at the end of the window's set-up (the session is open, nothing shown yet);
        unlike the other hooks, an error here is raised."""

    def instructions(self, sequence: Sequence) -> str:
        """More text for the screen between blocks, after the effort bar's note."""
        return ""

    def block_started(self, sequence: Sequence, metadata: dict) -> None:
        """A block is about to start recording; ``metadata`` goes into its trial's metadata."""

    def block_ending(self, sequence: Sequence, arrays: dict) -> None:
        """The running block is about to be saved; ``arrays`` go into its file."""

    def block_ended(self, sequence: Sequence, completed: bool) -> None:
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
        sequences: list[tuple[Path, Sequence]],
        hand: str = "right",
        lookahead_s: float = DEFAULT_LOOKAHEAD_S,
        tail_s: float = 1.5,
        sync_patch: bool = False,
        recalibrate_s: list[float] | None = None,
        calibration_reps: int = CALIBRATION_REPS,
        effort_meter: bool = False,
        effort_zones: dict[str, tuple[float, float]] | None = None,
        plugins: list[TaskPlugin] | tuple[TaskPlugin, ...] = (),
    ):
        super().__init__()
        if not sequences:
            raise ValueError("no sequences to run")
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
        self._last_calibration: float | None = None  # start of the last completed one
        self._n_calibrations = 0
        self._due_note = ""

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
        self._show_idle()

    # --- state -------------------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._t0_perf is not None

    @property
    def sequence(self) -> Sequence:
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

    def _show_idle(self) -> None:
        self.prompter.sequence = self.sequence
        self.prompter.effort = None
        self.prompter.end_run()
        n = len(self.sequences)
        done = f"{len(self.completed)}/{n} blocks done"
        self.prompter.header = f"Block {self.current + 1}/{n}: {self.sequence.name}   ({done})"
        if len(self.completed) == n:
            self.prompter.message = (
                "All blocks done.\nEsc to quit, or Left/Right then Shift to redo a block."
            )
        else:
            again = "  (already done)" if self.current in self.completed else ""
            how = self.sequence.params.get("instructions")
            self.prompter.message = (
                self._due_note
                + f"{self.sequence.name}{again}\n{len(self.sequence.prompts)} presses, "
                f"{self.sequence.duration_s + self.tail_s:.0f} s\n"
                + (f"\n{how}\n" if how else "")
                + self._effort_note()
                + "".join(t for t in self._plugins("instructions", self.sequence) if t)
                + "\nPress Shift to start"
            )
        self.prompter.update()

    def _effort_note(self) -> str:
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

    def _calibration_due(self) -> float | None:
        """Minutes since the last calibration started, if another one is due."""
        if not self.recalibrate_s or self._last_calibration is None:
            return None
        gap = self.recalibrate_s[min(self._n_calibrations - 1, len(self.recalibrate_s) - 1)]
        since = self._now() - self._last_calibration
        return since / 60 if since >= gap else None

    def _schedule_calibration(self) -> bool:
        """Put a calibration block in front of the current block if one is due."""
        if self.running or self.sequence.task_set == CALIBRATION:
            return False
        since_min = self._calibration_due()
        if since_min is None:
            return False
        at = self.current
        # like the block it comes before: pinches before pinch blocks, and so on
        block = calibration_for(self.sequences[at][1].task_set, self.calibration_reps)
        self.sequences.insert(at, (Path(f"built-in/{block.name}.json"), block))
        self.completed = {i + 1 if i >= at else i for i in self.completed}
        self._due_note = f"Calibration due ({since_min:.0f} min since the last one)\n\n"
        log.info("calibration block due: %.0f min since the last one", since_min)
        return True

    def start_block(self) -> None:
        if self.running:
            return
        if self._schedule_calibration():  # after a pause: show it first
            self._show_idle()
            return
        self._due_note = ""
        self._block_start = self._now()
        path, seq = self.sequences[self.current]
        extra = {}
        if self.prompter.show_effort and seq.task_set != CALIBRATION:  # off after a failure
            extra["effort_meter"] = self._effort_settings()
        self._plugins("block_started", seq, extra)
        self._effort_log = []
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

    def end_block(self, completed: bool) -> None:
        if not self.running:
            return
        self._timer.stop()
        elapsed = time.perf_counter() - self._t0_perf
        self._t0_perf = None
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
        self.session.end_trial(
            arrays=arrays,
            completed=completed,
            scroll_start_host_ns=self._t0_ns,
            elapsed_s=round(elapsed, 3),
        )
        self._plugins("block_ended", seq, completed)
        if completed:
            if self.sequence.task_set == CALIBRATION:
                self._last_calibration = self._block_start
                self._n_calibrations += 1
                if self.effort_meter is not None:
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
        """The just-saved calibration block becomes the effort meter's 100 %."""
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

    def _send_prompt_markers(self, t: float) -> None:
        label = active_label(self.sequence, t)
        if label != self._last_label:
            self.session.push_marker(f"prompt|{label or 'rest'}")
            self._last_label = label

    # --- input -------------------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        key = event.key()
        Qt = QtCore.Qt
        if event.isAutoRepeat():
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
        self.end_block(completed=False)
        if self.effort_meter is not None:
            self.session.device.detach_sink("effort")
            self.effort_meter.stop()
        self._plugins("close")
        super().closeEvent(event)
