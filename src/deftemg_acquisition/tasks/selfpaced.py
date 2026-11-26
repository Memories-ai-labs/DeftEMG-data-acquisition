"""Self-paced text blocks, without Qt: the runner's state machine, the saved attempts, and what
the prompter shows at a moment (``TextView``).

The prompter (``prompter.PrompterWidget.text_view``) draws a ``TextView``. Live and replay
build it with the same pure function, ``text_view_at(block, attempts, t_ns)``, from the
attempts' host times, so a replayed block shows what the subject saw.

Self-paced flow (``SelfPacedRunner``): the block starts on a preview of prompt
``start_index`` (0, or where an early-ended run stopped); Space starts a countdown, Go is marked
``countdown_s`` later; Space marks the end and the next prompt is previewed. After the last end
the runner waits: Space (``space()``) or ``end_block()`` finish the block as complete, ``r``
(``redo()``) redoes the last prompt. Space is ignored during the countdown, and for
``go_debounce_s`` after Go and after each end, so a double press can't end a trial at once. A
prompt with a fixed duration ends by itself (``end_reason`` "timer") unless Space ends it first.
``redo()`` voids the most recent attempt (under way, or just finished) and previews that prompt
again; in the countdown it cancels. ``end_block()`` elsewhere ends the block early (incomplete).

Timed blocks (the grasp calibration of the first recordings, before it became self-paced) are
not run any more, only replayed: ``text_view_at`` previews each prompt from the previous
prompt's end (with the time left to its onset), then shows Go at its onset and the end at its
offset. ``SelfPacedRunner`` rejects a timed block.

Attempts: one per preview, in order; a redo or a cancel adds one. The attempt of a preview left
by a ``redo()`` of the previous attempt is dropped (the new preview starts at that instant), so
every saved attempt reached its countdown, except an attempt ended by ``end_block()`` in its
preview. ``-1`` = did not happen.

One clock, the injected ``host_ns`` (the monotonic host clock in use), drives the state machine
and stamps the saved times, in integer ns, so the runner and ``text_view_at`` agree to the ns.
Scheduled events (Go, a timer end) get their scheduled host time, not the
time the runner noticed them. A deadline is due at its time exactly: a key pressed at the same
ns as a fixed duration's end finds the task already ended by the timer, as the view shows it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from deftemg_acquisition.clock import host_ns as _host_ns

from .textblock import COUNTDOWN_S, TIMED, TextBlock, TextPrompt

IDLE = "idle"  # no prompt (before the block starts, once it has finished, a timed block's tail)
PREVIEW = "preview"  # rest; the next prompt is shown dimmed, with its grasp highlighted
COUNTDOWN = "countdown"  # 3 2 1 before Go
GO = "go"  # the task is under way
DONE = "done"  # after the last end: Space finishes the block, r redoes the last prompt
PHASES = (IDLE, PREVIEW, COUNTDOWN, GO, DONE)

# Why an attempt stopped (``text_end_reason``): the operator (Space, or ``r``), a fixed
# duration or a timed prompt's offset, or the block ending early.
OPERATOR, TIMER, BLOCK_END = "operator", "timer", "block_end"
END_REASONS = (OPERATOR, TIMER, BLOCK_END)
# An attempt's outcome (``text_status``): done; voided by ``r``; ``r`` in its countdown; the
# block ended during it. "" while it is under way.
OK, REDO, CANCELLED, INCOMPLETE = "ok", "redo", "cancelled", "incomplete"
STATUSES = (OK, REDO, CANCELLED, INCOMPLETE)
GO_DEBOUNCE_S = 1.0


@dataclass(frozen=True)
class TextView:
    phase: str
    prompt: TextPrompt | None = None
    index: int = -1  # the prompt's index in the block (-1: none)
    total: int = 0  # prompts in the block
    # The attempt's row in the saved arrays (-1: none). Meaningful for a Go view, which must
    # have one: it keys go_paint_ns (the first painted Go frame). Preview and countdown views
    # set it too, but a preview's may name a row that later holds another prompt's attempt (a
    # redo drops the next prompt's preview row), so don't rely on it there.
    attempt: int = -1
    countdown_left: float | None = None  # s until Go (countdown; preview of a timed block)
    time_left: float | None = None  # s until a fixed-duration task ends by itself
    conditions: str = ""  # the session's conditions this block uses, as one line
    # Whether ``r`` (``SelfPacedRunner.redo()``) would act now: always in the countdown, Go
    # and DONE; in a preview, when the previous attempt is "ok" (not at a resumed block's first
    # prompt, nor in the preview a redo just started). Never in a timed block (only replayed).
    can_redo: bool = False

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"text view phase must be one of {PHASES}, got {self.phase!r}")
        if self.phase == GO and self.attempt < 0:  # its Go paint time would not be logged
            raise ValueError(f"a {GO!r} text view needs its attempt row, got {self.attempt}")


@dataclass
class Attempt:
    """One saved row: an attempt at prompt ``index`` (host times in ns, -1 = did not happen)."""

    index: int  # absolute index into the block's prompts
    preview_ns: int
    countdown_ns: int = -1
    go_ns: int = -1
    end_ns: int = -1  # when it stopped (end, timer, redo, cancel, early block end)
    end_reason: str = ""  # one of END_REASONS once stopped
    status: str = ""  # one of STATUSES once stopped


def attempts_from_arrays(arrays: Mapping[str, np.ndarray]) -> list[Attempt]:
    """The attempts back from a trial's ``text_*`` arrays (``TextBlock.block_arrays``), for
    ``text_view_at`` in replay."""
    cols = ("text_prompt_index", "text_preview_host_ns", "text_countdown_host_ns",
            "text_go_host_ns", "text_end_host_ns", "text_end_reason", "text_status")  # fmt: skip
    rows = zip(*(np.asarray(arrays[k]).tolist() for k in cols), strict=True)
    return [
        Attempt(int(i), int(p), int(c), int(g), int(e), str(r), str(s))
        for i, p, c, g, e, r, s in rows
    ]


def _ns(s: float) -> int:
    return round(s * 1e9)


def text_view_at(
    block: TextBlock,
    attempts: list[Attempt],
    t_ns: int,
    *,
    start_ns: int | None = None,
    countdown_s: float = COUNTDOWN_S,
    conditions: str = "",
    finish_ns: int = -1,
) -> TextView:
    """What the prompter shows at host time ``t_ns``, from the block's attempts so far.

    ``start_ns`` is the block's start (``block_start_host_ns``; needed for a timed block, whose
    view follows its schedule). ``countdown_s`` is the trial's countdown. ``finish_ns`` is when
    the block finished (``text_finish_host_ns``; -1: not yet, or a file from before it was
    saved); from then on the view is idle, as the block's tail was shown. An attempt still under
    way is followed by its schedule (Go ``countdown_s`` after the countdown starts, a fixed
    duration's end), so the live view does not wait for the runner's next update and agrees
    with the replayed view at every time. ``can_redo`` is decided as ``redo()`` would at that
    time: the next prompt's preview after an end counts its attempt as "ok" even before the
    runner has set it (a timer end not yet applied), or after a later redo voided it (replay).
    """
    n = len(block.prompts)

    def view(phase: str, i: int = -1, attempt: int = -1, **kw) -> TextView:
        prompt = block.prompts[i] if phase in (PREVIEW, COUNTDOWN, GO) else None
        return TextView(phase, prompt, i, n, attempt, conditions=conditions, **kw)

    if finish_ns != -1 and t_ns >= finish_ns:
        return view(IDLE)
    if block.mode == TIMED:
        if start_ns is None:
            raise ValueError(f"{block.name}: a timed block's view needs start_ns")
        if t_ns < start_ns or any(a.end_reason == BLOCK_END and t_ns >= a.end_ns for a in attempts):
            return view(IDLE)
        dt = t_ns - start_ns
        for i, p in enumerate(block.prompts):  # row i is prompt i: timed prompts run once
            if dt < _ns(p.onset_s):
                return view(PREVIEW, i, i, countdown_left=(_ns(p.onset_s) - dt) / 1e9)
            if dt < _ns(p.offset_s):
                return view(GO, i, i, time_left=(_ns(p.offset_s) - dt) / 1e9)
        return view(IDLE)

    k = next((k for k in range(len(attempts) - 1, -1, -1) if attempts[k].preview_ns <= t_ns), -1)
    if k < 0:
        return view(IDLE)
    a = attempts[k]
    duration = block.prompts[a.index].duration_s
    go, end, reason = a.go_ns, a.end_ns, a.end_reason
    if a.countdown_ns != -1 and go == -1 and end == -1:  # countdown under way: Go is due then
        go = a.countdown_ns + _ns(countdown_s)
    if go != -1 and end == -1 and duration is not None:  # fixed duration under way
        end, reason = go + _ns(duration), TIMER
    if end != -1 and t_ns >= end:  # stopped, and no later preview yet
        if reason == BLOCK_END:
            return view(IDLE)
        if a.index + 1 < n:  # the next preview (its attempt may be dropped later by a redo)
            return view(PREVIEW, a.index + 1, k + 1, can_redo=True)
        return view(DONE, n - 1, can_redo=True)
    if a.countdown_ns == -1 or t_ns < a.countdown_ns:
        # the previous attempt's status was final when this preview began
        return view(PREVIEW, a.index, k, can_redo=k > 0 and attempts[k - 1].status == OK)
    if go == -1 or t_ns < go:  # countdown_s only for a countdown cancelled before Go
        left = ((go if go != -1 else a.countdown_ns + _ns(countdown_s)) - t_ns) / 1e9
        return view(COUNTDOWN, a.index, k, countdown_left=left, can_redo=True)
    left = None if duration is None else (go + _ns(duration) - t_ns) / 1e9
    return view(GO, a.index, k, time_left=left, can_redo=True)


def _seconds(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be a non-negative number of seconds, got {v!r}")
    return float(v)


class SelfPacedRunner:
    """Runs one text block (see the module docstring); the block starts when this is made.

    ``host_ns`` (int ns; replaceable in tests) drives the state machine (countdown, debounce,
    fixed durations) and stamps the saved host times. Each key method applies the events already
    due first. A timed block (only replayed now) raises ``ValueError``.
    """

    def __init__(
        self,
        block: TextBlock,
        countdown_s: float = COUNTDOWN_S,
        go_debounce_s: float = GO_DEBOUNCE_S,
        start_index: int = 0,
        conditions: str = "",
        host_ns: Callable[[], int] = _host_ns,
    ):
        n = len(block.prompts)
        if block.mode == TIMED:
            raise ValueError(f"{block.name}: a timed block can only be replayed, not run")
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise ValueError(f"{block.name}: start_index must be an integer, got {start_index!r}")
        if not 0 <= start_index < n:
            raise ValueError(f"{block.name}: start_index must be in 0..{n - 1}, got {start_index}")
        self.block = block
        self.countdown_s = _seconds(countdown_s, "countdown_s")
        self.go_debounce_s = _seconds(go_debounce_s, "go_debounce_s")
        self.start_index = start_index
        self.conditions = conditions
        self._host_ns = host_ns
        self.attempts: list[Attempt] = []
        self.phase = PREVIEW
        self.finished = False
        self.completed = False
        self.finish_ns = -1  # host ns of the key that finished the block (-1: not finished)
        self._markers: list[str] = []
        self._space_from = -math.inf  # host ns; Space is ignored before this (debounce)
        self.start_ns = host_ns()  # block_start_host_ns
        self._preview(start_index, self.start_ns)

    # --- events ---

    def _mark(self, event: str, a: Attempt) -> None:
        label = f"|{self.block.prompts[a.index].label}" if event == "go" else ""
        self._markers.append(f"text|{event}|{a.index}{label}")

    def _preview(self, i: int, ns: int) -> None:
        self.attempts.append(Attempt(i, ns))
        self._mark("preview", self.attempts[-1])
        self.phase = PREVIEW

    def _stop(self, a: Attempt, ns: int, reason: str, status: str) -> None:
        a.end_ns, a.end_reason, a.status = ns, reason, status
        self._mark("end", a)

    def _end(self, ns: int, reason: str) -> None:
        """The running attempt is done: preview the next prompt, or wait to finish."""
        a = self.attempts[-1]
        self._stop(a, ns, reason, OK)
        self._space_from = ns + _ns(self.go_debounce_s)
        if a.index + 1 < len(self.block.prompts):
            self._preview(a.index + 1, ns)
        else:
            self.phase = DONE

    def update(self) -> None:
        """Apply the events due by now (the ``host_ns`` clock): Go after the countdown, a fixed
        duration's end."""
        if not self.finished:
            self._update(self._host_ns())

    def _update(self, t: int) -> None:
        a = self.attempts[-1]
        if self.phase == COUNTDOWN and t >= a.countdown_ns + _ns(self.countdown_s):
            a.go_ns = a.countdown_ns + _ns(self.countdown_s)
            self._space_from = a.go_ns + _ns(self.go_debounce_s)
            self.phase = GO
            self._mark("go", a)
        duration = self.block.prompts[a.index].duration_s
        if self.phase == GO and duration is not None and t >= a.go_ns + _ns(duration):
            self._end(a.go_ns + _ns(duration), TIMER)

    # --- keys ---

    def space(self) -> bool:
        """Space: start the countdown (preview), mark the end (Go), or finish the block (after
        the last end). Returns whether it did anything (ignored in the countdown, during the
        debounce and once finished)."""
        if self.finished:
            return False
        t = self._host_ns()
        self._update(t)
        if self.phase == COUNTDOWN or t < self._space_from:
            return False
        a = self.attempts[-1]
        if self.phase == PREVIEW:
            a.countdown_ns = t
            self.phase = COUNTDOWN
            self._mark("countdown", a)
            self._update(t)  # a countdown of 0 s is at Go already
        elif self.phase == GO:
            self._end(t, OPERATOR)
        else:  # DONE
            self._finish(t, completed=True)
        return True

    def redo(self) -> bool:
        """``r``: void the most recent attempt (under way, or just finished: "redo"; in its
        countdown: "cancelled") and preview its prompt again. Returns whether it did anything
        (not when the latest attempt is already voided, before the first, or once finished)."""
        if self.finished:
            return False
        ns = self._host_ns()
        self._update(ns)
        a = self.attempts[-1]
        if self.phase in (COUNTDOWN, GO):
            self._stop(a, ns, OPERATOR, CANCELLED if self.phase == COUNTDOWN else REDO)
        elif self.phase == DONE:
            a.status = REDO
        elif len(self.attempts) > 1 and self.attempts[-2].status == OK:  # previewing the next
            self.attempts.pop()  # that preview's attempt: the redo's preview starts now
            a = self.attempts[-1]
            a.status = REDO
        else:
            return False
        self._space_from = -math.inf  # a deliberate key; no double press to guard against
        self._preview(a.index, ns)
        return True

    def end_block(self) -> bool:
        """Enter (or Esc): end the block. After the last end it finishes as complete;
        otherwise the attempt under way stops ("block_end", "incomplete") and the block is
        incomplete. Returns whether the block is complete."""
        if self.finished:
            return self.completed
        ns = self._host_ns()
        self._update(ns)
        if self.finished:
            return self.completed
        if self.phase == DONE:
            self._finish(ns, completed=True)
            return True
        a = self.attempts[-1]
        if a.status == "":
            self._stop(a, ns, BLOCK_END, INCOMPLETE)
        self._finish(ns, completed=False)
        return False

    def _finish(self, ns: int, completed: bool) -> None:
        self.phase, self.finished, self.completed, self.finish_ns = IDLE, True, completed, ns

    # --- results ---

    def view(self) -> TextView:
        """What the prompter shows now (``text_view_at`` at the host clock's now); idle once
        the block has finished."""
        return text_view_at(self.block, self.attempts, self._host_ns(), start_ns=self.start_ns,
                            countdown_s=self.countdown_s, conditions=self.conditions,
                            finish_ns=self.finish_ns)  # fmt: skip

    def resume_index(self) -> int:
        """Where a new run of this block would start: after the last prompt with an "ok"
        attempt in this run (prompt order is fixed, so they form a prefix), else where this run
        started."""
        return max((a.index + 1 for a in self.attempts if a.status == OK),
                   default=self.start_index)  # fmt: skip

    def arrays(self, go_paint_ns: Mapping[int, int] | None = None) -> dict[str, np.ndarray]:
        """The trial's ``text_*`` arrays, ``block_start_host_ns`` and the scalar
        ``text_finish_host_ns``; ``go_paint_ns`` maps an attempt row to its first painted Go
        frame (``PrompterWidget.go_paint_ns``)."""
        return self.block.block_arrays(self.attempts, self.start_ns, go_paint_ns,
                                       finish_ns=self.finish_ns)  # fmt: skip

    def markers(self) -> list[str]:
        """The marker events since the last call, in order: ``text|preview|i``,
        ``text|countdown|i``, ``text|go|i|<label>``, ``text|end|i`` (``i`` is the prompt's
        index; an end is sent whenever an attempt stops)."""
        out, self._markers = self._markers, []
        return out
