"""Generate a set of grasp and natural-task text blocks from a catalogue, split into sessions.

One set covers every combination of the catalogue once (``catalogue.grasp_trials`` and
``catalogue.natural_units``), shuffled with a seeded ``numpy`` generator:

- Grasp blocks: one per object, with all of its combinations (up to 32) in a shuffled order.
  A block is never split; an early-ended block resumes where it stopped.
- Natural blocks: one per ``group``, with its units (one variant's steps, e.g. loosen, then
  tighten the same fastener) in a shuffled order; a unit's steps stay together and in order.
- Sessions: each block is placed whole. A block's estimate is the object swap plus, per prompt,
  the countdown, the task's estimate (or fixed duration) and the reset (``object_swap_s`` +
  ``TextBlock.estimate_s()``). A session's capacity is the limit less the setup and the grasp
  calibrations the prompter schedules (one at the start, then due ``recalibrate_s`` after the
  previous one started). Natural blocks are dealt round-robin over the sessions first, so they
  are spread across sessions; grasp blocks then go, largest first, to the least-loaded session.
  If a session would be over its capacity, another session is added and the blocks are placed
  again. The block order within each session is shuffled.

``generate_task_blocks`` is pure; the ``make-task-sequences`` command writes the files.
"""

from __future__ import annotations

import copy
import math
import secrets
from dataclasses import asdict, dataclass, field

import numpy as np

from .catalogue import Catalogue, NaturalUnit, Timing, grasp_trials, natural_units
from .textblock import SELF_PACED, TextBlock, TextPrompt

# The prompter's default recalibration (``--recalibrate 10 20``): the next grasp calibration is
# due 10 min after the first one started, later ones 20 min after the previous one.
RECALIBRATE_S = (600.0, 1200.0)

# The task line of a block's start screen (``params["start"]["task"]``); the grasp
# calibration's is ``textblock.GRASP_CALIBRATION_TASK`` (plus the swap when it sets out
# more than one object).
GRASP_TASK = "grasp as prompted, the hand shows which fingers"
NATURAL_TASK = "do each task as written"


@dataclass
class Session:
    name: str  # "S01"
    blocks: list[TextBlock]  # in run order
    calibrations: int  # grasp calibrations the prompter will run in it
    est_s: float  # setup, calibrations and blocks (object swaps included)


@dataclass
class TaskSet:
    seed: int
    timing: Timing
    recalibrate_s: tuple[float, ...]
    capacity_s: float  # time for blocks in one session
    sessions: list[Session] = field(default_factory=list)

    @property
    def blocks(self) -> list[TextBlock]:
        return [b for s in self.sessions for b in s.blocks]


def _due_by(elapsed_s: float, recalibrate_s) -> int:
    """Calibrations started before ``elapsed_s`` (s since the first one started)."""
    n, due = 1, recalibrate_s[0]
    while due < elapsed_s:
        n += 1
        due += recalibrate_s[min(n - 1, len(recalibrate_s) - 1)]
    return n


def calibrations_due(work_s: float, timing: Timing, recalibrate_s=RECALIBRATE_S) -> int:
    """Grasp calibrations the prompter runs in a session with ``work_s`` of blocks after the
    setup: one at the start, then one each time the next is due (``recalibrate_s[0]`` after
    the first started, then every ``recalibrate_s[-1]``), the calibrations' own time counted."""
    n = 1
    while (m := _due_by(work_s + n * timing.calibration_s, recalibrate_s)) != n:
        n = m
    return n


def session_capacity(timing: Timing, recalibrate_s=RECALIBRATE_S) -> float:
    """Time left for blocks in a session: the limit less the setup and the calibrations due
    before the limit."""
    work = timing.session_limit_s - timing.session_setup_s
    return work - _due_by(work, recalibrate_s) * timing.calibration_s


def block_estimate_s(block: TextBlock, timing: Timing) -> float:
    """A block's estimate when placed in a session: the object swap, then its prompts."""
    return timing.object_swap_s + block.estimate_s()


def _grasp_blocks(cat: Catalogue, rng: np.random.Generator, params: dict) -> list[TextBlock]:
    by_object: dict[str, list] = {}
    for t in grasp_trials(cat):
        by_object.setdefault(t.object, []).append(t)
    blocks = []
    for obj in cat.objects:
        trials = by_object[obj.id]
        prompts = [
            TextPrompt(
                t.text,
                t.object,
                grasp=t.grasp,
                approach=t.approach,
                variation=t.variation,
                highlight=cat.grasps[t.grasp].highlight,
                est_s=t.est_s,
                bold_line=t.bold_line,
            )
            for t in (trials[i] for i in rng.permutation(len(trials)))
        ]
        p = {
            **copy.deepcopy(params),
            "object": obj.id,
            "conditions": [obj.condition] if obj.condition else [],
            "start": {"set_out": [obj.name], "task": GRASP_TASK},
        }
        if obj.condition:  # so the prompter can ask for it and check the answer
            c = cat.conditions[obj.condition]
            p["condition_questions"] = {c.id: {"ask": c.ask, "choices": list(c.choices)}}
        blocks.append(TextBlock(obj.id, "grasp", SELF_PACED, prompts, p))
    return blocks


def _natural_blocks(cat: Catalogue, rng: np.random.Generator, params: dict) -> list[TextBlock]:
    by_group: dict[str, list[NaturalUnit]] = {}
    for u in natural_units(cat):
        by_group.setdefault(u.group, []).append(u)
    # what to set out for a group: its tasks' items, in catalogue order, each once
    items: dict[str, dict[str, None]] = {}
    for task in cat.natural:
        items.setdefault(task.group, {}).update(dict.fromkeys(task.items))
    blocks = []
    for group, units in by_group.items():
        prompts = [
            TextPrompt(
                s.text,
                s.task,  # natural steps have no object: the task id
                variation=s.variant,  # the task's variant ("hex_ratchet"), if any
                step=s.step,
                group=s.group,
                highlight=(),
                duration_s=s.duration_s,
                est_s=s.est_s,
            )
            for u in (units[i] for i in rng.permutation(len(units)))
            for s in u.steps
        ]
        p = {
            **copy.deepcopy(params),
            "group": group,
            "conditions": [],
            "start": {"set_out": list(items[group]), "task": NATURAL_TASK},
        }
        blocks.append(TextBlock(group, "natural", SELF_PACED, prompts, p))
    return blocks


def _place(natural, grasp, est, n: int) -> list[list[TextBlock]]:
    """Natural blocks round-robin over ``n`` sessions, then grasp blocks (already in placing
    order) each to the least-loaded session."""
    sessions: list[list[TextBlock]] = [[] for _ in range(n)]
    loads = [0.0] * n
    for i, b in enumerate(natural):
        sessions[i % n].append(b)
        loads[i % n] += est[id(b)]
    for b in grasp:
        i = min(range(n), key=loads.__getitem__)
        sessions[i].append(b)
        loads[i] += est[id(b)]
    return sessions


def generate_task_blocks(
    cat: Catalogue,
    timing: Timing | None = None,
    seed: int | None = None,
    recalibrate_s=RECALIBRATE_S,
) -> TaskSet:
    """Every combination of ``cat`` once, as grasp and natural blocks split into sessions.

    ``timing`` defaults to the catalogue's; ``seed`` to a new random one (saved in every block's
    ``params`` and in the result). ``ValueError`` if the catalogue has nothing to prompt, a
    session has no time for blocks, or one block is longer than a session's capacity."""
    timing = cat.timing if timing is None else timing
    recalibrate_s = tuple(float(s) for s in recalibrate_s)
    if not recalibrate_s or min(recalibrate_s) <= 0:
        raise ValueError(f"recalibrate_s must be positive times, got {recalibrate_s!r}")
    seed = secrets.randbits(31) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    capacity = session_capacity(timing, recalibrate_s)
    if capacity <= 0:
        raise ValueError(
            f"a session of {timing.session_limit_s:.0f} s has no time left for blocks after the "
            "setup and calibrations"
        )
    params = {
        "seed": seed,
        "catalogue": cat.name,
        "catalogue_sha256": cat.sha256,
        "timing": {**asdict(timing), "recalibrate_s": list(recalibrate_s)},
    }
    if cat.calibration:
        params["calibration"] = cat.calibration
    if cat.terms:  # every block carries them, so any block can show the terms slide
        params["terms"] = [{"term": t.term, "text": t.text} for t in cat.terms]
    grasp = _grasp_blocks(cat, rng, params)
    natural = _natural_blocks(cat, rng, params)
    if not grasp and not natural:
        raise ValueError(f"catalogue {cat.name!r} has no objects or natural tasks")
    est = {id(b): block_estimate_s(b, timing) for b in grasp + natural}
    too_long = [b for b in grasp + natural if est[id(b)] > capacity]
    if too_long:
        b = too_long[0]
        raise ValueError(
            f"block {b.name!r} is estimated at {est[id(b)]:.0f} s, more than the "
            f"{capacity:.0f} s a session has for blocks"
        )
    natural = [natural[i] for i in rng.permutation(len(natural))]
    grasp = [grasp[i] for i in rng.permutation(len(grasp))]
    grasp.sort(key=lambda b: -est[id(b)])  # stable: equal sizes stay in the shuffled order
    n = max(1, math.ceil(sum(est.values()) / capacity))
    while True:
        placed = _place(natural, grasp, est, n)
        if all(sum(est[id(b)] for b in s) <= capacity for s in placed):
            break
        n += 1
    # natural blocks were dealt from the first session on: shuffle the sessions, so they are
    # not always in the first ones
    placed = [placed[i] for i in rng.permutation(len(placed))]
    out = TaskSet(seed, timing, recalibrate_s, capacity)
    for k, blocks in enumerate(placed, start=1):
        name = f"S{k:02d}"
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        work = sum(est[id(b)] for b in blocks)
        for j, b in enumerate(blocks, start=1):
            b.name = f"{name}_b{j:02d}_{b.name}"
            # plan_est_s: the block's estimate in the session plan, which includes the object
            # swap; TextBlock.estimate_s() is the prompts' alone
            b.params = {**b.params, "session": name, "plan_est_s": round(est[id(b)], 1)}
        n_cal = calibrations_due(work, timing, recalibrate_s)
        total = timing.session_setup_s + n_cal * timing.calibration_s + work
        out.sessions.append(Session(name, blocks, n_cal, total))
    return out


def block_file(block: TextBlock) -> str:
    """The block's file name in its session folder: ``bNN_<object or group>.json`` (the
    prompter runs a folder's files in name order)."""
    return block.name.split("_", 1)[1] + ".json"
