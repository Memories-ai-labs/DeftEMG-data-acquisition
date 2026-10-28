"""Prompt sequences for cued finger-press tasks: data model, generator, and JSON files.

A sequence is one block of prompts, recorded as one trial. Each prompt asks for one or more
digits to be pressed at a force level from ``onset_s`` (seconds after the block starts) for
``duration_s``. Digits are named anatomically, independent of which hand is used.
"""

from __future__ import annotations

import itertools
import json
import secrets
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

DIGITS = ("thumb", "index", "middle", "ring", "pinky")
LEVELS = ("low", "high")  # saved as level index 0, 1 (see prompt_arrays)
# Calibration blocks: rest, then each digit pressed as hard as is comfortable ("max").
CALIBRATION = "calibration"
CALIBRATION_LEVELS = ("max",)
# Maximum presses per digit in a calibration block (2 were found too tiring).
CALIBRATION_REPS = 1
# Task sets whose chords always join the thumb with one or more other digits, and the
# instructions the prompter shows before each of their blocks.
THUMB_SETS = {
    "pinch": (
        "Pinch: press the prompted fingertips against the tip of the thumb while their capsules "
        "are on the target, gently for narrow capsules and firmly for wide ones. Relax the hand "
        "fully between presses."
    ),
    "squeeze": (
        "Squeeze: close the thumb and the prompted fingers together while their capsules are on "
        "the target, gently for narrow capsules and firmly for wide ones. Relax the hand fully "
        "between presses."
    ),
}
# What a calibration block asks for, matching the task blocks it precedes: "press" (each digit
# on its own, for single and multi), or the THUMB_SETS ("pinch", "squeeze": the thumb with each
# other finger, in the posture of those blocks). Instructions shown before the block.
CALIBRATION_TASKS = {
    "press": (
        "Rest quietly until the first capsule reaches the target. Then give each digit a "
        "brief press, as hard as is comfortable, while its capsule is on the target, and "
        "relax fully in between. Rest a minute before the next block."
    ),
    "pinch": (
        "Rest quietly until the first capsule reaches the target. Then pinch each prompted "
        "fingertip against the tip of the thumb, briefly and as hard as is comfortable, while "
        "its capsule is on the target, and relax fully in between. Rest a minute before the "
        "next block."
    ),
    "squeeze": (
        "Rest quietly until the first capsule reaches the target. Then close the thumb and the "
        "prompted finger together, briefly and as hard as is comfortable, while its capsule is "
        "on the target, and relax fully in between. Rest a minute before the next block."
    ),
}
# Recordings made before the switch to two levels used ("low", "medium", "high") without saving
# level names; they can still be loaded (e.g. for replay) but are never generated.
LEGACY_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class Prompt:
    onset_s: float
    duration_s: float
    digits: tuple[str, ...]
    levels: tuple[str, ...]  # one per digit

    def __post_init__(self):
        if len(self.digits) != len(self.levels) or not self.digits:
            raise ValueError(f"need one level per digit: {self}")
        if len(set(self.digits)) != len(self.digits):
            raise ValueError(f"repeated digit: {self}")
        for d in self.digits:
            if d not in DIGITS:
                raise ValueError(f"unknown digit {d!r}")
        for lv in self.levels:
            if lv not in LEVELS and lv not in LEGACY_LEVELS and lv not in CALIBRATION_LEVELS:
                raise ValueError(f"unknown level {lv!r}")

    @property
    def offset_s(self) -> float:
        return self.onset_s + self.duration_s

    @property
    def label(self) -> str:
        return "+".join(f"{d}:{lv}" for d, lv in zip(self.digits, self.levels, strict=True))


@dataclass
class Sequence:
    name: str
    task_set: str  # "single", "multi", "pinch", "squeeze" or "calibration"
    prompts: list[Prompt]
    params: dict = field(default_factory=dict)  # generator settings, for the record

    @property
    def duration_s(self) -> float:
        return max((p.offset_s for p in self.prompts), default=0.0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": "scroll_prompter",
            "task_set": self.task_set,
            "params": self.params,
            "prompts": [
                {**asdict(p), "digits": list(p.digits), "levels": list(p.levels)}
                for p in self.prompts
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Sequence:
        prompts = [
            Prompt(p["onset_s"], p["duration_s"], tuple(p["digits"]), tuple(p["levels"]))
            for p in d["prompts"]
        ]
        prompts.sort(key=lambda p: p.onset_s)
        return cls(d["name"], d["task_set"], prompts, d.get("params", {}))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> Sequence:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @property
    def calibration_task(self) -> str | None:
        """For a calibration block, what it asks for (``CALIBRATION_TASKS``; "press" for
        blocks recorded before pinch and squeeze calibrations); None for a task block."""
        if self.task_set != CALIBRATION:
            return None
        return self.params.get("task", "press")

    @property
    def implied_digits(self) -> tuple[str, ...]:
        """Digits the prompter leaves undrawn because the task implies them: the thumb in pinch
        and squeeze (their blocks and calibrations), which it always joins. They stay in the
        prompts, so labels include them."""
        task = self.calibration_task or self.task_set
        return ("thumb",) if task in THUMB_SETS else ()

    @property
    def level_names(self) -> tuple[str, ...]:
        """The level coding: ``LEVELS`` (task blocks), ``CALIBRATION_LEVELS``, or
        ``LEGACY_LEVELS`` for sequences rebuilt from older recordings. ``params["level_names"]``
        (a sequence rebuilt from a recording) says which one when the levels used fit several."""
        used = {lv for p in self.prompts for lv in p.levels}
        codings = (LEVELS, CALIBRATION_LEVELS, LEGACY_LEVELS)
        given = tuple(self.params.get("level_names") or ())
        if given in codings and used <= set(given):
            return given
        for names in codings:
            if used <= set(names):
                return names
        raise ValueError(f"{self.name}: levels {sorted(used)} mix task and calibration levels")

    def prompt_arrays(self) -> dict[str, np.ndarray]:
        """Per-prompt arrays: onset/offset (s), digit mask (n, 5), level index (n, 5; -1 = none).

        Level indices refer to ``level_names``, saved alongside so the coding is explicit.
        """
        names = self.level_names
        n = len(self.prompts)
        mask = np.zeros((n, len(DIGITS)), bool)
        level = np.full((n, len(DIGITS)), -1, np.int8)
        for i, p in enumerate(self.prompts):
            for d, lv in zip(p.digits, p.levels, strict=True):
                mask[i, DIGITS.index(d)] = True
                level[i, DIGITS.index(d)] = names.index(lv)
        return {
            "prompt_onset_s": np.array([p.onset_s for p in self.prompts]),
            "prompt_offset_s": np.array([p.offset_s for p in self.prompts]),
            "prompt_digits": mask,
            "prompt_level": level,
            "digit_names": np.array(DIGITS),
            "level_names": np.array(names),
        }


def calibration_sequence(
    rest_s: float = 3.0,
    hold_s: float = 1.5,
    gap_s: float = 3.0,
    reps: int = CALIBRATION_REPS,
    lead_s: float = 1.0,
    task: str = "press",
) -> Sequence:
    """A calibration block for the start of a session (about 25 s; 46 s with ``reps`` 2).

    ``rest_s`` of quiet rest (nothing to press; it sets each channel's noise level), then
    ``reps`` rounds of brief maximum efforts of ``hold_s``, "as hard as is comfortable" (level
    "max"), ``gap_s`` apart. ``task`` (``CALIBRATION_TASKS``) matches the task blocks: "press",
    each digit on its own, thumb to pinky; "pinch" / "squeeze", the thumb with each other
    finger, index to pinky (about 19 s), in the posture of those blocks, the thumb implied as
    there (its lane left empty). The digits share forearm flexors, and near-maximal efforts
    tire them quickly, so the efforts are brief, with twice as long to relax, one round (two,
    22.5 s apart, were too tiring). The first effort comes before the prompter's look-ahead, so
    its capsule is already on screen when the block starts. Low / high presses can then be read
    relative to the maximum of the same digit or pinch, and a force-sensing glove can later
    record the same block.
    """
    if task not in CALIBRATION_TASKS:
        raise ValueError(f"calibration task must be one of {list(CALIBRATION_TASKS)}, got {task!r}")
    chords = [(d,) for d in DIGITS] if task == "press" else [("thumb", d) for d in DIGITS[1:]]
    t = lead_s + rest_s
    prompts = []
    for _ in range(reps):
        for chord in chords:
            prompts.append(Prompt(round(t, 3), hold_s, chord, ("max",) * len(chord)))
            t += hold_s + gap_s
    params = {"rest_s": rest_s, "hold_s": hold_s, "gap_s": gap_s, "reps": reps, "task": task,
              "instructions": CALIBRATION_TASKS[task]}  # fmt: skip
    name = CALIBRATION if task == "press" else f"{CALIBRATION}_{task}"
    return Sequence(name, CALIBRATION, prompts, params)


def calibration_for(task_set: str, reps: int = CALIBRATION_REPS) -> Sequence:
    """The calibration block for task blocks of ``task_set``: maximum pinches before pinch
    blocks, squeezes before squeeze blocks, presses before anything else."""
    return calibration_sequence(reps=reps, task=task_set if task_set in THUMB_SETS else "press")


def active_levels(seq: Sequence, t: float) -> dict[str, str]:
    """Digits that should be pressed at time ``t`` (s from block start) -> level."""
    out = {}
    for p in seq.prompts:
        if p.onset_s <= t < p.offset_s:
            out.update(zip(p.digits, p.levels, strict=True))
    return out


def active_label(seq: Sequence, t: float) -> str:
    """E.g. ``"index:high+middle:high"``, or ``""`` when nothing should be pressed."""
    act = active_levels(seq, t)
    return "+".join(f"{d}:{act[d]}" for d in DIGITS if d in act)


@dataclass
class GeneratorConfig:
    # "single", "multi" (chords of any digits), or "pinch" / "squeeze" (THUMB_SETS: the thumb
    # with one or more other digits)
    task_set: str = "single"
    reps: int = 2  # times each (digits, levels) combination appears across all blocks
    # Chords: how many digits per press, the thumb included. None: (2,) for multi; (2, 5) for
    # pinch / squeeze: the thumb with one other digit, or the whole hand (5 chords; two- and
    # three-finger chords, sizes 3 and 4, were too hard to perform).
    chord_sizes: tuple[int, ...] | None = None
    chord_force: str = "same"  # chords: "same" level for all digits, or "independent"
    hold_s: tuple[float, float] = (0.5, 2.0)  # min, max hold duration (uniform)
    gap_s: tuple[float, float] = (0.8, 1.5)  # min, max rest between prompts (uniform)
    lead_in_s: float = 6.5  # time before the first prompt; >= the prompter's look-ahead
    per_block: int = 15
    # None: a fresh random seed, so each subject / session gets different blocks (identical
    # blocks make the next press predictable and can leak across data splits). The seed used
    # is saved in every block's params, so a set can be regenerated.
    seed: int | None = None


def _conditions(cfg: GeneratorConfig, rng: np.random.Generator):
    """All (digits, levels) combinations, each repeated ``reps`` times."""
    if cfg.task_set == "single":
        base = [((d,), (lv,)) for d in DIGITS for lv in LEVELS]
        return base * cfg.reps
    if cfg.task_set in THUMB_SETS:
        if not set(cfg.chord_sizes) <= {2, 3, 4, 5}:
            raise ValueError(f"{cfg.task_set}: chord sizes are 2-5 digits, thumb included")
        chords = [("thumb", *c) for k in cfg.chord_sizes
                  for c in itertools.combinations(DIGITS[1:], k - 1)]  # fmt: skip
    elif cfg.task_set == "multi":
        chords = [c for k in cfg.chord_sizes for c in itertools.combinations(DIGITS, k)]
    else:
        sets = ["single", "multi", *THUMB_SETS]
        raise ValueError(f"task_set must be one of {sets}, got {cfg.task_set!r}")
    if cfg.chord_force == "same":
        base = [(c, (lv,) * len(c)) for c in chords for lv in LEVELS]
        return base * cfg.reps
    if cfg.chord_force != "independent":
        raise ValueError(f"chord_force must be 'same' or 'independent', got {cfg.chord_force!r}")
    # Independent levels: every chord appears len(LEVELS) * reps times with random per-digit levels,
    # balanced so each digit sees each level equally often within a chord where possible.
    out = []
    for c in chords:
        for _ in range(cfg.reps):
            cols = [rng.permutation(LEVELS) for _ in c]
            out += [(c, tuple(col[i] for col in cols)) for i in range(len(LEVELS))]
    return out


def generate(cfg: GeneratorConfig, name_prefix: str) -> list[Sequence]:
    """Shuffle the balanced condition list and split it into timed blocks."""
    if cfg.seed is None:
        cfg = replace(cfg, seed=secrets.randbits(31))
    if cfg.chord_sizes is None:
        cfg = replace(cfg, chord_sizes=(2, 5) if cfg.task_set in THUMB_SETS else (2,))
    rng = np.random.default_rng(cfg.seed)
    conds = _conditions(cfg, rng)
    order = rng.permutation(len(conds))
    conds = [conds[i] for i in order]
    params = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(cfg).items()}
    if cfg.task_set in THUMB_SETS:
        params["instructions"] = THUMB_SETS[cfg.task_set]

    blocks = []
    for b, start in enumerate(range(0, len(conds), cfg.per_block)):
        t = cfg.lead_in_s
        prompts = []
        for digits, levels in conds[start : start + cfg.per_block]:
            hold = float(rng.uniform(*cfg.hold_s))
            prompts.append(Prompt(round(t, 3), round(hold, 3), digits, levels))
            t += hold + float(rng.uniform(*cfg.gap_s))
        name = f"{name_prefix}_{cfg.task_set}_b{b + 1:02d}"
        blocks.append(Sequence(name, cfg.task_set, prompts, {**params, "block": b + 1}))
    return blocks
