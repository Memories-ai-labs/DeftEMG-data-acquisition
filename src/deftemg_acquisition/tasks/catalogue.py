"""Catalogue of text prompts for grasp and natural-task blocks, read from a TOML file
(``protocols/grasp_natural/catalogue.toml``, whose header describes the format).

Grasp blocks prompt an object, an approach (e.g. from the side or the top), a grasp type (which
digits) and a task variation (e.g. a light grasp, or pick and place); natural blocks prompt
open-ended tasks ("Loosen the wing nut all the way off"). Adding an object or a task is an edit
to the file. ``load_catalogue`` reads and checks it (errors name the entry); ``grasp_trials``
expands it to every allowed (object, approach, grasp, variation) combination, once each, and
``natural_units`` to units of natural-task steps: one variant's steps (loosen, then tighten the
same fastener), which stay consecutive and in order when blocks are shuffled.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

from .sequence import DIGITS

_ID = re.compile(r"[A-Za-z0-9_-]+")  # ids also name block files
VARIANT = "{variant}"  # replaced by the variant's name in natural-task texts


class CatalogueError(ValueError):
    """A catalogue that can't be used; the message names the entry (and the file, if loaded)."""


@dataclass(frozen=True)
class Timing:
    """Assumptions behind the generator's time estimates (s)."""

    countdown_s: float = 3.0
    reset_s: float = 5.0  # the operator's Space for the end, the reset, the next Space
    object_swap_s: float = 45.0
    session_setup_s: float = 180.0
    calibration_s: float = 90.0
    session_limit_s: float = 1800.0


@dataclass(frozen=True)
class GraspType:
    id: str  # "TI"
    name: str  # "thumb-index"
    tips: tuple[str, ...] = ()  # fingertips highlighted on the hand outline (DIGITS)
    whole_hand: bool = False  # the whole hand highlighted instead (a power grasp)

    @property
    def highlight(self) -> tuple[str, ...] | str:
        """What the prompter highlights: fingertips, or "hand"."""
        return "hand" if self.whole_hand else self.tips


@dataclass(frozen=True)
class Variation:
    id: str
    text: str
    est_s: float  # estimated time of one trial, countdown and reset not included


@dataclass(frozen=True)
class Condition:
    """Asked once at the session start (e.g. a bottle filled or empty) and saved with the
    blocks of the objects that name it."""

    id: str
    ask: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    name: str
    approaches: tuple[str, ...] = ()  # () = one approach, not named
    grasps: tuple[str, ...] | None = None  # None = every grasp type
    exclude_grasps: tuple[str, ...] = ()
    exclude_variations: tuple[str, ...] = ()
    approach_grasps: dict[str, tuple[str, ...]] = field(default_factory=dict)  # replace grasps
    note: str = ""  # shown with every prompt of the object
    notes: dict[str, str] = field(default_factory=dict)  # grasp type -> note
    condition: str | None = None


@dataclass(frozen=True)
class NaturalVariant:
    id: str
    name: str  # replaces "{variant}" in the step texts
    # step id -> this variant's own wording (key None: the one step of a task given by text)
    text: dict[str | None, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NaturalStepSpec:
    id: str | None  # None for a one-step task given by its ``text``
    text: str


@dataclass(frozen=True)
class NaturalTask:
    id: str
    group: str  # the natural block it joins
    est_s: float  # estimated time of one step
    steps: tuple[NaturalStepSpec, ...]  # run in this order, one after the other
    variants: tuple[NaturalVariant, ...] = ()  # () = the task once, with no variant
    fixed_s: float | None = None  # every step runs this long; None = the operator ends it


@dataclass
class Catalogue:
    name: str
    timing: Timing
    grasps: dict[str, GraspType]
    variations: dict[str, Variation]
    approaches: dict[str, str]  # id -> the words shown ("from the side")
    conditions: dict[str, Condition]
    objects: list[ObjectSpec]
    natural: list[NaturalTask]
    calibration: dict = field(default_factory=dict)  # optional overrides, read by the blocks
    sha256: str = ""  # of the file it was loaded from


@dataclass(frozen=True)
class GraspTrial:
    """One grasp combination, with the text the prompter shows for it."""

    object: str
    approach: str | None
    grasp: str
    variation: str
    note: str
    text: str
    est_s: float
    condition: str | None


@dataclass(frozen=True)
class NaturalStep:
    task: str
    group: str
    variant: str | None
    step: str | None
    text: str
    est_s: float
    duration_s: float | None  # fixed duration; None = the operator ends it


@dataclass(frozen=True)
class NaturalUnit:
    """One variant of a natural task: its steps, kept together and in order."""

    task: str
    group: str
    variant: str | None
    steps: tuple[NaturalStep, ...]


def catalogue_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_catalogue(path: Path) -> Catalogue:
    """Read and check a catalogue file (``CatalogueError`` names the file and the entry)."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode())
        return parse_catalogue(data, sha256=hashlib.sha256(raw).hexdigest())
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, CatalogueError) as e:
        raise CatalogueError(f"{path}: {e}") from None


# ---- reading and checking ----


def _table(d, where: str, required=(), optional=()) -> dict:
    if not isinstance(d, dict):
        raise CatalogueError(f"{where}: expected a table, got {d!r}")
    missing = [k for k in required if k not in d]
    if missing:
        raise CatalogueError(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(d) - set(required) - set(optional))
    if unknown:
        raise CatalogueError(f"{where}: unknown key(s) {', '.join(unknown)}")
    return d


def _mapping(d, where: str) -> dict:
    """A table whose keys are names (grasp types, approaches, ...), not fixed fields."""
    return _table(d, where, optional=d.keys() if isinstance(d, dict) else ())


def _str(v, where: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise CatalogueError(f"{where}: expected a non-empty string, got {v!r}")
    return v


def _id(v, where: str) -> str:
    if not isinstance(v, str) or not _ID.fullmatch(v):
        raise CatalogueError(f"{where}: ids are letters, digits, '_' and '-', got {v!r}")
    return v


def _strs(v, where: str) -> tuple[str, ...]:
    if not isinstance(v, list):
        raise CatalogueError(f"{where}: expected a list of strings, got {v!r}")
    return tuple(_str(x, where) for x in v)


def _seconds(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or v <= 0:
        raise CatalogueError(f"{where}: expected a positive number of seconds, got {v!r}")
    return float(v)


def _unique(ids, where: str) -> None:
    seen = set()
    for i in ids:
        if i in seen:
            raise CatalogueError(f"{where}: duplicate id {i!r}")
        seen.add(i)


def _known(names, known, where: str, kind: str) -> None:
    for n in names:
        if n not in known:
            raise CatalogueError(f"{where}: unknown {kind} {n!r} (known: {', '.join(known)})")


def _grasp(gid: str, d) -> GraspType:
    where = f"grasps.{gid}"
    d = _table(d, where, ["name"], ["tips", "whole_hand"])
    tips = _strs(d.get("tips", []), f"{where}.tips")
    _known(tips, DIGITS, f"{where}.tips", "digit")
    whole = d.get("whole_hand", False)
    if not isinstance(whole, bool) or whole == bool(tips):
        raise CatalogueError(f"{where}: give either tips or whole_hand = true")
    return GraspType(_id(gid, where), _str(d["name"], f"{where}.name"), tips, whole)


def _object(i: int, d, cat: Catalogue) -> ObjectSpec:
    where = f"objects[{i}]"
    opt = [
        "approaches",
        "grasps",
        "exclude_grasps",
        "exclude_variations",
        "approach_grasps",
        "note",
        "notes",
        "condition",
    ]
    d = _table(d, where, ["id", "name"], opt)
    oid = _id(d["id"], f"{where}.id")
    where = f"objects {oid!r}"
    approaches = _strs(d.get("approaches", []), f"{where}.approaches")
    _known(approaches, cat.approaches, f"{where}.approaches", "approach")
    _unique(approaches, f"{where}.approaches")
    grasps = None if "grasps" not in d else _strs(d["grasps"], f"{where}.grasps")
    exclude = _strs(d.get("exclude_grasps", []), f"{where}.exclude_grasps")
    by_approach = {
        a: _strs(g, f"{where}.approach_grasps.{a}")
        for a, g in _mapping(d.get("approach_grasps", {}), f"{where}.approach_grasps").items()
    }
    _known(by_approach, approaches, f"{where}.approach_grasps", "approach of this object")
    notes = {
        g: _str(n, f"{where}.notes.{g}")
        for g, n in _mapping(d.get("notes", {}), f"{where}.notes").items()
    }
    for name, gs in [
        ("grasps", grasps or ()),
        ("exclude_grasps", exclude),
        ("notes", notes),
        *((f"approach_grasps.{a}", g) for a, g in by_approach.items()),
    ]:
        _known(gs, cat.grasps, f"{where}.{name}", "grasp type")
    no_var = _strs(d.get("exclude_variations", []), f"{where}.exclude_variations")
    _known(no_var, cat.variations, f"{where}.exclude_variations", "variation")
    condition = d.get("condition")
    if condition is not None:
        _id(condition, f"{where}.condition")
        _known([condition], cat.conditions, f"{where}.condition", "condition")
    note = _str(d["note"], f"{where}.note") if "note" in d else ""
    return ObjectSpec(
        oid,
        _str(d["name"], f"{where}.name"),
        approaches=approaches,
        grasps=grasps,
        exclude_grasps=exclude,
        exclude_variations=no_var,
        approach_grasps=by_approach,
        note=note,
        notes=notes,
        condition=condition,
    )


def _natural(i: int, d) -> NaturalTask:
    where = f"natural[{i}]"
    opt = ["est_s", "text", "steps", "variants", "fixed_s"]
    d = _table(d, where, ["id", "group"], opt)
    tid = _id(d["id"], f"{where}.id")
    where = f"natural {tid!r}"
    if ("text" in d) == ("steps" in d):
        raise CatalogueError(f"{where}: give either text (one step) or steps")
    if "text" in d:
        steps = (NaturalStepSpec(None, _str(d["text"], f"{where}.text")),)
    else:
        if not isinstance(d["steps"], list) or not d["steps"]:
            raise CatalogueError(f"{where}.steps: expected a list of tables")
        steps = tuple(
            NaturalStepSpec(_id(s["id"], f"{where}.steps.id"), _str(s["text"], f"{where}.steps"))
            for s in (_table(s, f"{where}.steps", ["id", "text"]) for s in d["steps"])
        )
        _unique([s.id for s in steps], f"{where}.steps")
    variants = []
    if not isinstance(d.get("variants", []), list):
        raise CatalogueError(f"{where}.variants: expected a list of tables ([[natural.variants]])")
    for v in d.get("variants", []):
        v = _table(v, f"{where}.variants", ["id", "name"], ["text"])
        vid = _id(v["id"], f"{where}.variants.id")
        vwhere = f"{where}.variants {vid!r}.text"
        if "text" not in d:  # per step: text.<step> = "..."
            text = {
                k: _str(t, f"{vwhere}.{k}") for k, t in _mapping(v.get("text", {}), vwhere).items()
            }
            _known(text, [s.id for s in steps], vwhere, "step")
        elif isinstance(v.get("text"), dict):
            raise CatalogueError(f'{vwhere}: a one-step task takes text = "...", not per step')
        else:  # one step: text = "..."
            text = {} if "text" not in v else {None: _str(v["text"], vwhere)}
        variants.append(NaturalVariant(vid, _str(v["name"], f"{where}.variants.name"), text))
    _unique([v.id for v in variants], f"{where}.variants")
    if not variants and any(VARIANT in s.text for s in steps):
        raise CatalogueError(f"{where}: a text uses {VARIANT} but the task has no variants")
    fixed = None if "fixed_s" not in d else _seconds(d["fixed_s"], f"{where}.fixed_s")
    if "est_s" in d:
        est = _seconds(d["est_s"], f"{where}.est_s")
    elif fixed is not None:
        est = fixed  # a fixed-duration step takes its duration
    else:
        raise CatalogueError(f"{where}: missing est_s (needed without fixed_s)")
    return NaturalTask(tid, _id(d["group"], f"{where}.group"), est, steps, tuple(variants), fixed)


def parse_catalogue(data: dict, sha256: str = "") -> Catalogue:
    """A catalogue from the file's parsed contents, checked: unknown or missing keys, unknown
    names (grasp types, approaches, variations, conditions, steps), duplicate ids, and objects
    or tasks that come out with no prompts all raise ``CatalogueError``."""
    top = ["catalogue", "grasps", "variations", "objects"]
    d = _table(
        data,
        "catalogue file",
        top,
        ["timing", "approaches", "conditions", "natural", "calibration"],
    )
    name = _str(_table(d["catalogue"], "catalogue", ["name"])["name"], "catalogue.name")
    timing_keys = [f.name for f in fields(Timing)]
    timing = _table(d.get("timing", {}), "timing", optional=timing_keys)
    timing = Timing(**{k: _seconds(v, f"timing.{k}") for k, v in timing.items()})
    grasps = {g: _grasp(g, v) for g, v in _mapping(d["grasps"], "grasps").items()}
    if not grasps:
        raise CatalogueError("grasps: no grasp types")
    if not isinstance(d["variations"], list) or not d["variations"]:
        raise CatalogueError("variations: expected a list of tables")
    variations = {}
    for i, v in enumerate(d["variations"]):
        v = _table(v, f"variations[{i}]", ["id", "text", "est_s"])
        vid = _id(v["id"], f"variations[{i}].id")
        _unique([*variations, vid], "variations")
        variations[vid] = Variation(
            vid,
            _str(v["text"], f"variations {vid!r}.text"),
            _seconds(v["est_s"], f"variations {vid!r}.est_s"),
        )
    approaches = {
        _id(a, f"approaches.{a}"): _str(t, f"approaches.{a}")
        for a, t in _mapping(d.get("approaches", {}), "approaches").items()
    }
    conditions = {}
    for c, v in _mapping(d.get("conditions", {}), "conditions").items():
        v = _table(v, f"conditions.{c}", ["ask", "choices"])
        choices = _strs(v["choices"], f"conditions.{c}.choices")
        if len(choices) < 2 or len(set(choices)) != len(choices):
            raise CatalogueError(f"conditions.{c}.choices: need two or more different choices")
        conditions[_id(c, f"conditions.{c}")] = Condition(
            c, _str(v["ask"], f"conditions.{c}.ask"), choices
        )
    calibration = d.get("calibration", {})
    if not isinstance(calibration, dict):
        raise CatalogueError(f"calibration: expected a table, got {calibration!r}")
    cat = Catalogue(
        name, timing, grasps, variations, approaches, conditions, [], [], calibration, sha256
    )
    if not isinstance(d["objects"], list):
        raise CatalogueError("objects: expected a list of tables")
    cat.objects = [_object(i, o, cat) for i, o in enumerate(d["objects"])]
    _unique([o.id for o in cat.objects], "objects")
    natural = d.get("natural", [])
    if not isinstance(natural, list):
        raise CatalogueError("natural: expected a list of tables")
    cat.natural = [_natural(i, t) for i, t in enumerate(natural)]
    _unique([t.id for t in cat.natural], "natural")
    for obj in cat.objects:  # every object and approach has prompts, and every note is shown
        used = set()
        for approach in obj.approaches or (None,):
            grasps = _object_grasps(cat, obj, approach)
            if not grasps or not _object_variations(cat, obj):
                at = "" if approach is None else f" from approach {approach!r}"
                raise CatalogueError(f"objects {obj.id!r}: no combinations{at}")
            used.update(grasps)
        _known(obj.notes, sorted(used), f"objects {obj.id!r}.notes", "grasp type of this object")
    for task in cat.natural:  # the subject must see which variant a prompt is for
        texts = Counter(s.text for u in _task_units(task) for s in u.steps)
        same = [t for t, n in texts.items() if n > 1]
        if same:
            raise CatalogueError(
                f"natural {task.id!r}: variants must differ in their prompt text, so the subject "
                f"sees which one; {same[0]!r} is shown for more than one (use {VARIANT} or "
                "text.<step> in each variant)"
            )
    return cat


# ---- expansion ----


def _object_grasps(cat: Catalogue, obj: ObjectSpec, approach: str | None) -> list[str]:
    allowed = obj.approach_grasps.get(approach) if approach is not None else None
    if allowed is None:
        allowed = obj.grasps if obj.grasps is not None else tuple(cat.grasps)
    return [g for g in cat.grasps if g in allowed and g not in obj.exclude_grasps]


def _object_variations(cat: Catalogue, obj: ObjectSpec) -> list[str]:
    return [v for v in cat.variations if v not in obj.exclude_variations]


def grasp_text(
    cat: Catalogue, obj: ObjectSpec, approach: str | None, grasp: str, variation: str, note: str
) -> str:
    """What the prompter shows: the object and approach, the grasp type (and the note), and the
    variation, one line each, e.g. "Small coffee mug / Thumb-index grasp, on the lip / Light
    grasp"."""
    where = obj.name if approach is None else f"{obj.name}, {cat.approaches[approach]}"
    name = cat.grasps[grasp].name
    how = f"{name[0].upper()}{name[1:]} grasp" + (f", {note}" if note else "")
    return "\n".join([where, how, cat.variations[variation].text])


def grasp_trials(cat: Catalogue) -> list[GraspTrial]:
    """Every allowed (object, approach, grasp, variation) combination, once each, in catalogue
    order (object, then approach, grasp type and variation)."""
    out = []
    for obj in cat.objects:
        for approach in obj.approaches or (None,):
            for g in _object_grasps(cat, obj, approach):
                note = "; ".join(n for n in (obj.note, obj.notes.get(g, "")) if n)
                for v in _object_variations(cat, obj):
                    text = grasp_text(cat, obj, approach, g, v, note)
                    est = cat.variations[v].est_s
                    out.append(GraspTrial(obj.id, approach, g, v, note, text, est, obj.condition))
    return out


def natural_text(step: NaturalStepSpec, variant: NaturalVariant | None) -> str:
    """What the prompter shows for a natural-task step: its text, or the variant's own wording
    for it, with "{variant}" replaced by the variant's name."""
    if variant is None:
        return step.text
    return variant.text.get(step.id, step.text).replace(VARIANT, variant.name)


def _task_units(task: NaturalTask) -> list[NaturalUnit]:
    units = []
    for v in task.variants or (None,):
        vid = None if v is None else v.id
        steps = tuple(
            NaturalStep(
                task.id, task.group, vid, s.id, natural_text(s, v), task.est_s, task.fixed_s
            )
            for s in task.steps
        )
        units.append(NaturalUnit(task.id, task.group, vid, steps))
    return units


def natural_units(cat: Catalogue) -> list[NaturalUnit]:
    """Every natural task's units, one per variant (one for a task without variants), in
    catalogue order; a unit's steps run one after the other, in order."""
    return [u for task in cat.natural for u in _task_units(task)]
