"""Text-prompt blocks: data model and JSON files, and the grasp calibration block.

A text block is one block of written prompts ("Tuna can, from the top / Thumb-index grasp /
Light grasp"), recorded as one trial, like a lane block (``sequence.Sequence``). Most blocks
are self-paced: the operator starts each prompt and marks its end, or a prompt with a fixed
duration ends by itself, as each brief maximum of the grasp calibration does. Timed blocks
(each prompt has an onset and a duration) are the grasp calibrations of the first recordings;
they still load and replay, but are no longer made or run.

A block file is self-contained: each prompt carries the text the prompter shows, so a block
does not need the catalogue it was generated from. Its JSON has ``"kind": "text_block"``;
lane block files have no ``kind``, and ``load_block`` reads both.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from .sequence import CALIBRATION_REPS, DIGITS, Sequence, calibration_for

KIND = "text_block"
TASK = "text_prompter"  # the trial's ``task``; lane blocks are "scroll_prompter"
GRASP_CALIBRATION = "grasp_calibration"  # not sequence.CALIBRATION: press analysis skips it
TASK_SETS = ("grasp", "natural", GRASP_CALIBRATION)
SELF_PACED = "self_paced"  # the operator starts each prompt and marks its end
TIMED = "timed"  # each prompt has an onset and a duration (old grasp calibrations; replay only)
MODES = (SELF_PACED, TIMED)
BLOCK_KEYS = ("kind", "name", "task", "task_set", "mode", "prompts")  # JSON; "params" optional
# Timed onsets are rounded to the millisecond, so a prompt may start this much before the
# previous one ends without counting as an overlap.
OVERLAP_TOLERANCE_S = 1e-3
# The trial's per-attempt arrays (``TextBlock.block_arrays``): int64, and numpy unicode.
TEXT_INT_ARRAYS = ("text_prompt_index", "text_preview_host_ns", "text_countdown_host_ns",
                   "text_go_host_ns", "text_end_host_ns", "text_go_paint_ns")  # fmt: skip
PROMPT_ARRAY_FIELDS = ("object", "grasp", "approach", "variation", "step", "text")
TEXT_STR_ARRAYS = ("text_end_reason", "text_status", *(f"text_{k}" for k in PROMPT_ARRAY_FIELDS))
HAND = "hand"  # highlight the whole hand (a power grasp) instead of fingertips
# Per-prompt time assumptions behind ``TextBlock.estimate_s``, used when the block's
# ``params["timing"]`` does not give them (s).
COUNTDOWN_S = 3.0
RESET_S = 5.0  # the operator's Space for the end, the reset, the next Space

# The grasp calibration: one brief maximum of each grasp type, each started by the operator
# and held for ``hold_s``. A catalogue's ``[calibration]`` table (embedded in each block's
# ``params["calibration"]``) overrides any of these.
CALIBRATION_DEFAULTS = {
    "hold_s": 1.5,
    "maxima": [
        {
            "grasp": "TI",
            "object": "hardcover_book",
            "text": "Hardcover book\nThumb-index grasp\nBrief maximum, as hard as is comfortable",
        },
        {
            "grasp": "TIM",
            "object": "hardcover_book",
            "text": "Hardcover book\nThumb-index-middle grasp\n"
            "Brief maximum, as hard as is comfortable",
        },
        {
            "grasp": "AD",
            "object": "hardcover_book",
            "text": "Hardcover book\nAll-fingertips grasp\n"
            "Brief maximum, as hard as is comfortable",
        },
        {
            "grasp": "Power",
            "object": "ball",
            "text": "Ball\nPower grasp\nBrief maximum, as hard as is comfortable",
        },
    ],
}
CALIBRATION_KEYS = tuple(CALIBRATION_DEFAULTS)
# The timed calibration's spacing, in the ``params["calibration"]`` of blocks saved before it
# became self-paced: accepted there and dropped (a catalogue rejects them)
LEGACY_CALIBRATION_KEYS = ("rest_s", "gap_s", "swap_gap_s")
MAXIMUM_KEYS = ("grasp", "object", "text")  # each of ``maxima``; "highlight" is optional
# What the hand outline highlights for the default grasp types, for maxima that don't say.
GRASP_HIGHLIGHTS = {
    "TI": ("thumb", "index"),
    "TIM": ("thumb", "index", "middle"),
    "AD": DIGITS,
    "Power": HAND,
}
GRASP_CALIBRATION_TASK = "brief maximum grasps, as hard as is comfortable"  # params["start"]


def _seconds(v, where: str, positive: bool = False) -> float:
    """A finite number of seconds, not negative (``positive``: above 0); not a bool."""
    ok = (
        not isinstance(v, bool)
        and isinstance(v, int | float)
        and math.isfinite(v)
        and (v > 0 if positive else v >= 0)
    )
    if not ok:
        raise ValueError(f"{where}: expected a {'positive' if positive else 'non-negative'} "
                         f"number of seconds, got {v!r}")  # fmt: skip
    return float(v)


def _highlight(v, where: str) -> tuple[str, ...] | str:
    """A highlight from a file or a caller: "hand", or digits (a list or tuple)."""
    if v == HAND:
        return HAND
    if not isinstance(v, list | tuple) or not all(isinstance(d, str) for d in v):
        raise ValueError(f"{where}: highlight must be {HAND!r} or a list of digits, got {v!r}")
    for d in v:
        if d not in DIGITS:
            raise ValueError(f"{where}: unknown digit {d!r} in highlight")
    if len(set(v)) != len(v):
        raise ValueError(f"{where}: repeated digit in highlight {v!r}")
    return tuple(v)


@dataclass(frozen=True)
class TextPrompt:
    text: str  # what the screen shows, one line per part (object and approach, grasp, ...)
    object: str  # object id (grasp prompts) or task id (natural-task steps)
    grasp: str | None = None  # grasp type id
    approach: str | None = None
    variation: str | None = None
    step: str | None = None  # natural-task step ("loosen")
    group: str | None = None  # natural-task group (the block it joined)
    highlight: tuple[str, ...] | str = ()  # digits, or "hand"; () = none
    duration_s: float | None = None  # fixed duration (page turning); None = operator ends it
    onset_s: float | None = None  # timed blocks only (s after the block starts)
    est_s: float | None = None  # estimated time of the task itself, for the block's estimate

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"prompt text must be a non-empty string: {self}")
        where = f"prompt {self.text!r}"
        if not isinstance(self.object, str) or not self.object:
            raise ValueError(f"{where}: object must be a non-empty string, got {self.object!r}")
        for k in ("grasp", "approach", "variation", "step", "group"):
            v = getattr(self, k)
            if v is not None and not isinstance(v, str):
                raise ValueError(f"{where}: {k} must be a string or None, got {v!r}")
        object.__setattr__(self, "highlight", _highlight(self.highlight, where))
        for k in ("duration_s", "onset_s", "est_s"):
            v = getattr(self, k)
            if v is not None:
                object.__setattr__(self, k, _seconds(v, f"{where}.{k}", positive=k == "duration_s"))

    @property
    def offset_s(self) -> float | None:
        """End of a timed prompt; None for a self-paced one."""
        if self.onset_s is None or self.duration_s is None:
            return None
        return self.onset_s + self.duration_s

    @property
    def label(self) -> str:
        """Short label for markers and replay, e.g. "tuna_can:TI" or "screw_panel:loosen"."""
        return ":".join(x for x in (self.object, self.grasp or self.step) if x)

    def to_dict(self) -> dict:
        h = self.highlight
        return {**asdict(self), "highlight": h if isinstance(h, str) else list(h)}

    @classmethod
    def from_dict(cls, d: dict) -> TextPrompt:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"prompt {d.get('text')!r}: unknown key(s) {', '.join(unknown)}")
        return cls(**d)


@dataclass
class TextBlock:
    name: str
    task_set: str  # "grasp", "natural" or "grasp_calibration"
    mode: str  # "self_paced" or "timed"
    prompts: list[TextPrompt]
    # generator settings, for the record: seed, session, est_s, conditions (names),
    # instructions, catalogue sha256, timing assumptions, calibration overrides
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a text block's name must be a non-empty string, got {self.name!r}")
        if self.task_set not in TASK_SETS:
            raise ValueError(
                f"{self.name}: task_set must be one of {TASK_SETS}, got {self.task_set!r}"
            )
        if self.mode not in MODES:
            raise ValueError(f"{self.name}: mode must be one of {MODES}, got {self.mode!r}")
        if not isinstance(self.prompts, list) or not self.prompts:
            raise ValueError(f"{self.name}: a text block needs a list of at least one prompt")
        if not all(isinstance(p, TextPrompt) for p in self.prompts):
            raise ValueError(f"{self.name}: prompts must be TextPrompts")
        if self.mode == TIMED:
            if any(p.onset_s is None or p.duration_s is None for p in self.prompts):
                raise ValueError(
                    f"{self.name}: every prompt of a timed block needs onset_s and duration_s"
                )
            for i, (a, b) in enumerate(zip(self.prompts, self.prompts[1:], strict=False)):
                if b.onset_s < a.offset_s - OVERLAP_TOLERANCE_S:
                    raise ValueError(
                        f"{self.name}: timed prompt {i + 1} starts at {b.onset_s} s, before "
                        f"prompt {i} ends ({a.offset_s} s); prompts must be in order, not "
                        "overlapping"
                    )
        elif any(p.onset_s is not None for p in self.prompts):
            raise ValueError(f"{self.name}: prompts of a self-paced block have no onset_s")
        if not isinstance(self.params, dict):
            raise ValueError(f"{self.name}: params must be a table, got {self.params!r}")
        timing = self.params.get("timing", {})
        if not isinstance(timing, dict):
            raise ValueError(f"{self.name}: params.timing must be a table, got {timing!r}")
        for k in ("countdown_s", "reset_s"):  # used by estimate_s
            if k in timing:
                _seconds(timing[k], f"{self.name}: params.timing.{k}")
        if "calibration" in self.params:  # fail at load, not when the calibration comes due
            try:
                calibration_settings(self.params["calibration"])
            except ValueError as e:
                raise ValueError(f"{self.name}: params.{e}") from None

    @property
    def is_calibration(self) -> bool:
        return self.task_set == GRASP_CALIBRATION

    def estimate_s(self, start_index: int = 0) -> float:
        """Estimated length (s) from prompt ``start_index`` on (a resumed block counts only its
        remaining prompts; ``ValueError`` outside ``0..N-1``, and for a timed block): per
        prompt, the countdown, the task (a fixed duration, else ``est_s``, else 0) and the reset
        (``params["timing"]``'s ``countdown_s`` and ``reset_s``, else ``COUNTDOWN_S`` and
        ``RESET_S``)."""
        n = len(self.prompts)
        if self.mode == TIMED:
            raise ValueError(f"{self.name}: a timed block has no estimate (it is only replayed)")
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise ValueError(f"{self.name}: start_index must be an integer, got {start_index!r}")
        if not 0 <= start_index < n:
            raise ValueError(f"{self.name}: start_index must be in 0..{n - 1}, got {start_index}")
        timing = self.params.get("timing", {})
        per = timing.get("countdown_s", COUNTDOWN_S) + timing.get("reset_s", RESET_S)
        return sum(per + (p.duration_s or p.est_s or 0.0) for p in self.prompts[start_index:])

    def to_dict(self) -> dict:
        return {
            "kind": KIND,
            "name": self.name,
            "task": TASK,
            "task_set": self.task_set,
            "mode": self.mode,
            "params": self.params,
            "prompts": [p.to_dict() for p in self.prompts],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TextBlock:
        if not isinstance(d, dict) or d.get("kind") != KIND:
            got = d.get("kind") if isinstance(d, dict) else d
            raise ValueError(f"not a text block (kind {got!r}, expected {KIND!r})")
        name = d.get("name")
        if d.get("task") != TASK:
            raise ValueError(f"{name}: a text block's task is {TASK!r}, got {d.get('task')!r}")
        missing = [k for k in BLOCK_KEYS if k not in d]
        unknown = sorted(set(d) - {*BLOCK_KEYS, "params"})
        if missing:
            raise ValueError(f"{name}: missing {', '.join(missing)}")
        if unknown:
            raise ValueError(f"{name}: unknown key(s) {', '.join(unknown)}")
        if not isinstance(d["prompts"], list) or not all(isinstance(p, dict) for p in d["prompts"]):
            raise ValueError(f"{name}: prompts must be a list of tables")
        prompts = [TextPrompt.from_dict(p) for p in d["prompts"]]
        return cls(d["name"], d["task_set"], d["mode"], prompts, d.get("params", {}))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> TextBlock:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def block_arrays(
        self, attempts: list, block_start_ns: int, go_paint_ns: dict[int, int] | None = None
    ) -> dict[str, np.ndarray]:
        """The trial's ``text_*`` arrays, one row per attempt (a redo adds a row), and the
        scalar ``block_start_host_ns``. ``attempts`` are ``selfpaced.Attempt`` rows (prompt
        index, host times, end reason, status); ``go_paint_ns`` maps a row to the host time of
        its first painted Go frame. Dtypes: ``TEXT_INT_ARRAYS`` int64 (-1 = did not happen),
        ``TEXT_STR_ARRAYS`` numpy unicode (a prompt field that is None is "")."""
        go_paint_ns = go_paint_ns or {}
        prompts = [self.prompts[a.index] for a in attempts]
        ints = {
            "text_prompt_index": [a.index for a in attempts],
            "text_preview_host_ns": [a.preview_ns for a in attempts],
            "text_countdown_host_ns": [a.countdown_ns for a in attempts],
            "text_go_host_ns": [a.go_ns for a in attempts],
            "text_end_host_ns": [a.end_ns for a in attempts],
            "text_go_paint_ns": [go_paint_ns.get(k, -1) for k in range(len(attempts))],
        }
        strs = {
            "text_end_reason": [a.end_reason for a in attempts],
            "text_status": [a.status for a in attempts],
            **{f"text_{k}": [getattr(p, k) or "" for p in prompts] for k in PROMPT_ARRAY_FIELDS},
        }
        return {
            **{k: np.array(v, dtype=np.int64) for k, v in ints.items()},
            **{k: np.array(v, dtype=np.str_) for k, v in strs.items()},
            "block_start_host_ns": np.int64(block_start_ns),
        }


def load_block(path: Path) -> Sequence | TextBlock:
    """A block file: a ``TextBlock`` if its ``kind`` is "text_block", else a lane ``Sequence``
    (lane files have no ``kind``)."""
    d = json.loads(Path(path).read_text())
    kind = d.get("kind")
    if kind is None and d.get("task") == TASK:
        raise ValueError(f'{path}: a {TASK!r} block needs "kind": {KIND!r}')
    if kind is None:
        return Sequence.from_dict(d)
    if kind == KIND:
        return TextBlock.from_dict(d)
    raise ValueError(f"{path}: unknown block kind {kind!r}")


# ---- the grasp calibration ----


def calibration_settings(overrides: dict | None = None) -> dict:
    """The grasp calibration's settings: ``CALIBRATION_DEFAULTS`` with ``overrides`` (a
    catalogue's ``[calibration]`` table, or a block's ``params["calibration"]``) applied,
    checked. Unknown keys raise ``ValueError`` (a misspelt ``hold_s`` is not ignored), except
    ``LEGACY_CALIBRATION_KEYS``, which are dropped. Each maximum gets a ``highlight``: its own,
    else ``GRASP_HIGHLIGHTS`` for its grasp type."""
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError(f"calibration: expected a table, got {overrides!r}")
    overrides = {k: v for k, v in overrides.items() if k not in LEGACY_CALIBRATION_KEYS}
    unknown = sorted(set(overrides) - set(CALIBRATION_KEYS))
    if unknown:
        known = ", ".join(CALIBRATION_KEYS)
        raise ValueError(f"calibration: unknown key(s) {', '.join(unknown)} (known: {known})")
    d = {**CALIBRATION_DEFAULTS, **overrides}
    out = {"hold_s": _seconds(d["hold_s"], "calibration.hold_s", positive=True)}
    if not isinstance(d["maxima"], list) or not d["maxima"]:
        raise ValueError(f"calibration.maxima: expected a non-empty list, got {d['maxima']!r}")
    maxima = []
    for i, m in enumerate(d["maxima"]):
        where = f"calibration.maxima[{i}]"
        if not isinstance(m, dict):
            raise ValueError(f"{where}: expected a table, got {m!r}")
        missing = [k for k in MAXIMUM_KEYS if k not in m]
        unknown = sorted(set(m) - {*MAXIMUM_KEYS, "highlight"})
        if missing:
            raise ValueError(f"{where}: missing {', '.join(missing)}")
        if unknown:
            raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}")
        for k in MAXIMUM_KEYS:
            if not isinstance(m[k], str) or not m[k].strip():
                raise ValueError(f"{where}.{k}: expected a non-empty string, got {m[k]!r}")
        if "highlight" in m:
            highlight = _highlight(m["highlight"], where)
        elif m["grasp"] in GRASP_HIGHLIGHTS:
            highlight = GRASP_HIGHLIGHTS[m["grasp"]]
        else:
            raise ValueError(f"{where}: grasp type {m['grasp']!r} needs a highlight")
        h = highlight if isinstance(highlight, str) else list(highlight)
        maxima.append({**{k: m[k] for k in MAXIMUM_KEYS}, "highlight": h})
    out["maxima"] = maxima
    return out


def grasp_calibration_block(
    overrides: dict | None = None, reps: int = CALIBRATION_REPS
) -> TextBlock:
    """The grasp calibration: a self-paced block of brief maximum grasps, "as hard as is
    comfortable", each started by the operator and ending by itself after ``hold_s``: by default
    thumb-index, thumb-index-middle and all-fingertips on a hardcover book, then power on a
    ball. Consecutive maxima on the same object form a group; with ``reps`` above 1 each group
    is repeated ``reps`` times before the next, so there are no extra object swaps (the swap
    happens while the next object's maximum is previewed). ``overrides`` as in
    ``calibration_settings``.
    """
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
        raise ValueError(f"reps must be a positive integer, got {reps!r}")
    s = calibration_settings(overrides)
    groups: list[list[dict]] = []
    for m in s["maxima"]:
        if groups and groups[-1][0]["object"] == m["object"]:
            groups[-1].append(m)
        else:
            groups.append([m])
    prompts = [
        TextPrompt(m["text"], m["object"], grasp=m["grasp"], highlight=m["highlight"],
                   duration_s=s["hold_s"])
        for group in groups for _ in range(reps) for m in group
    ]  # fmt: skip
    # what to set out: each maximum's first line (its object, as the subject reads it)
    set_out = list(dict.fromkeys(m["text"].strip().splitlines()[0].strip() for m in s["maxima"]))
    params = {"calibration": s, "reps": reps,
              "start": {"set_out": set_out, "task": GRASP_CALIBRATION_TASK}}  # fmt: skip
    return TextBlock(GRASP_CALIBRATION, GRASP_CALIBRATION, SELF_PACED, prompts, params)


def calibration_for_block(
    block: Sequence | TextBlock, reps: int = CALIBRATION_REPS
) -> Sequence | TextBlock:
    """The calibration block to run before ``block``: for a lane block, exactly what
    ``calibration_for`` gives for its task set; for a text block, the grasp calibration, with
    the block's ``params["calibration"]`` overrides."""
    if isinstance(block, TextBlock):
        return grasp_calibration_block(block.params.get("calibration"), reps)
    return calibration_for(block.task_set, reps)
