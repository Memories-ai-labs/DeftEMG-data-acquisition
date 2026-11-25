"""Generate a set of grasp and natural-task text blocks from a catalogue, split into sessions.

    pixi run make-task-sequences --catalogue protocols/grasp_natural/catalogue.toml \\
        --out protocols/G01
    pixi run make-task-sequences --catalogue protocols/grasp_natural/catalogue.toml --summary

Every combination in the catalogue appears once in the set, shuffled: one grasp block per
object, one natural block per group (paired steps, such as loosen then tighten, stay together
and in order). Blocks are split into session folders (``S01/``, ``S02/``, ...), each estimated
to fit the session limit, with files named ``bNN_<object or group>.json`` in run order; run a
session with ``pixi run prompter protocols/G01/S01``. ``plan.json`` records the seed, the timing
assumptions and every session's blocks and estimate. Timing assumptions come from the
catalogue's ``[timing]`` table; the options below override them. ``--summary`` prints the
counts and session estimates without writing anything. Lane (finger-press) blocks come from
``make-sequences``; the README's "Grasp and natural-task blocks" covers the catalogue format.

The seed is random unless ``--seed`` is given, so the command refuses to write into a folder
that already holds a set (``plan.json`` or ``S01/``...). ``--force`` replaces the set: the new
one is written first, then the files the old ``plan.json`` lists are deleted, and only those
(it refuses if a session folder or block file is a symbolic link, or a session folder holds
a block file the old plan doesn't list).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

from deftemg_acquisition.tasks.catalogue import (
    CatalogueError,
    Timing,
    grasp_trials,
    load_catalogue,
    natural_units,
)
from deftemg_acquisition.tasks.textgen import (
    TaskSet,
    block_estimate_s,
    block_file,
    generate_task_blocks,
)

PLAN = "plan.json"
SESSION_DIR = re.compile(r"S\d\d")
LISTED_FILE = re.compile(r"S\d\d/[A-Za-z0-9_-]+\.json")


def _session_entries(out: Path) -> list[Path]:
    """Everything in ``out`` named like a session folder (``S01``...), links included."""
    if not out.is_dir():
        return []
    return sorted(p for p in out.iterdir() if SESSION_DIR.fullmatch(p.name))


def check_old_set(out: Path, force: bool) -> list[Path]:
    """Check ``out`` before writing a set into it, and return the files of the set already
    there (those its ``plan.json`` lists, as found) for deleting once the new set is in place.
    Deletes nothing.

    Without ``force``, an existing set (``plan.json`` or a session folder) stops it. With
    ``force``, so does anything that could make deleting the old set reach outside ``out``
    or delete a file the old plan does not list: ``out``, ``plan.json`` or a session entry
    that is a symbolic link (a linked ``out`` is refused, not followed), a session entry that
    is not a folder, a listed path that is not a session block file or whose file is a link or
    not in its session folder, and a block file in a session folder that the plan doesn't
    list."""
    if out.is_symlink():
        raise SystemExit(f"{out} is a symbolic link; give the folder it points to")
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} is not a folder")
    plan, entries = out / PLAN, _session_entries(out)
    for e in entries:
        if e.is_symlink() or not e.is_dir():
            raise SystemExit(f"{e} is not a plain folder; move or remove it by hand")
    if not plan.exists() and not plan.is_symlink() and not entries:
        return []
    found = PLAN if plan.exists() or plan.is_symlink() else f"{entries[0].name}/"
    if not force:
        raise SystemExit(
            f"{out} already holds a set ({found}); the default seed is random, so it would be "
            "replaced by different blocks. Use another --out, or --force"
        )
    if plan.is_symlink():
        raise SystemExit(f"{plan} is a symbolic link; move or remove it by hand")
    if not plan.exists():
        raise SystemExit(f"{out} has {found} but no {PLAN} listing its files; remove it by hand")
    try:
        listed = json.loads(plan.read_text())["files"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"{plan}: can't read its list of files ({e})") from None
    if not isinstance(listed, list) or not all(
        isinstance(f, str) and LISTED_FILE.fullmatch(f) for f in listed
    ):
        raise SystemExit(f"{plan}: 'files' must list session block files (S01/b01_x.json)")
    root = out.resolve()
    old = []
    for f in listed:
        session, path = out / f.split("/")[0], out / f
        if not session.exists():
            continue
        if session.resolve().parent != root:
            raise SystemExit(f"{session} is not a folder directly in {out}")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise SystemExit(f"{path} is not a plain file; move or remove it by hand")
        if path.exists():
            if path.resolve().parent != session.resolve():
                raise SystemExit(f"{path} is not in {session}")
            old.append(path)
    unlisted = [
        p for d in entries for p in sorted(d.glob("*.json")) if f"{d.name}/{p.name}" not in listed
    ]
    if unlisted:
        raise SystemExit(
            f"{unlisted[0]} is not listed in {plan}, so the new set would run with it; "
            "move or remove it by hand"
        )
    return old


def write_set(out: Path, ts: TaskSet, plan: dict, old: list[Path]) -> None:
    """Write the set into ``out``: first all of it into a staging folder in ``out`` (a failed
    write leaves the old set as it was), then move the files into place, delete the old set's
    files that the new set did not replace, and its session folders left empty, and last move
    ``plan.json`` into place."""
    out.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=out))
    try:
        for s in ts.sessions:
            for b in s.blocks:
                b.save(stage / s.name / block_file(b))
        (stage / PLAN).write_text(json.dumps(plan, indent=2))
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    new = {out / f for f in plan["files"]}
    for f in plan["files"]:
        (out / f).parent.mkdir(exist_ok=True)
        os.replace(stage / f, out / f)
    for path in old:
        if path not in new:
            path.unlink(missing_ok=True)
    for d in {path.parent for path in old}:
        if not any(d.iterdir()):
            d.rmdir()
    os.replace(stage / PLAN, out / PLAN)
    shutil.rmtree(stage)


# the source checkout this module runs from (src/deftemg_acquisition/cli/ -> the repo)
REPO_ROOT = Path(__file__).resolve().parents[3]


def catalogue_label(catalogue: Path) -> str:
    """The catalogue's path as plan.json records it: relative to the repo when the catalogue is
    inside it, else just its file name, so the plan doesn't record where a computer keeps it."""
    try:
        return Path(catalogue).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(catalogue).name


def plan_dict(ts: TaskSet, catalogue: Path, sha256: str, name: str) -> dict:
    sessions = []
    for s in ts.sessions:
        blocks = [
            {
                "file": f"{s.name}/{block_file(b)}",
                "name": b.name,
                "task_set": b.task_set,
                "trials": len(b.prompts),
                "est_s": round(block_estimate_s(b, ts.timing), 1),
            }
            for b in s.blocks
        ]
        sessions.append(
            {
                "session": s.name,
                "est_s": round(s.est_s, 1),
                "calibrations": s.calibrations,
                "grasp_trials": sum(b["trials"] for b in blocks if b["task_set"] == "grasp"),
                "natural_trials": sum(b["trials"] for b in blocks if b["task_set"] == "natural"),
                "blocks": blocks,
            }
        )
    return {
        "seed": ts.seed,
        "catalogue": name,
        "catalogue_path": catalogue_label(catalogue),
        "catalogue_sha256": sha256,
        "timing": {**asdict(ts.timing), "recalibrate_s": list(ts.recalibrate_s)},
        "capacity_s": round(ts.capacity_s, 1),
        "counts": {
            "grasp": sum(s["grasp_trials"] for s in sessions),
            "natural": sum(s["natural_trials"] for s in sessions),
        },
        "sessions": sessions,
        "files": [b["file"] for s in sessions for b in s["blocks"]],
    }


def print_counts(cat) -> None:
    grasp = Counter(t.object for t in grasp_trials(cat))
    natural = Counter(s.group for u in natural_units(cat) for s in u.steps)
    print(f"catalogue {cat.name} (sha256 {cat.sha256[:12]})")
    print(f"grasp combinations: {sum(grasp.values())}")
    for obj, n in grasp.items():
        print(f"  {obj:24s} {n:3d}")
    print(f"natural trials: {sum(natural.values())}")
    for group, n in natural.items():
        print(f"  {group:24s} {n:3d}")


def print_sessions(ts: TaskSet, plan: dict) -> None:
    print(f"seed {ts.seed} (regenerate with --seed); up to {ts.capacity_s / 60:.1f} min of "
          f"blocks per session")  # fmt: skip
    for s in plan["sessions"]:
        print(
            f"{s['session']}  {len(s['blocks']):2d} blocks  {s['grasp_trials']:3d} grasp + "
            f"{s['natural_trials']:2d} natural trials  {s['calibrations']} calibration"
            f"{'' if s['calibrations'] == 1 else 's'}  "
            f"est {s['est_s'] / 60:.1f} min"
        )
    total = sum(s.est_s for s in ts.sessions)
    n = len(ts.sessions)
    print(f"{n} session{'' if n == 1 else 's'}, {total / 3600:.1f} h in all")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--catalogue", type=Path, default=Path("protocols/grasp_natural/catalogue.toml"),
        help="catalogue file (default: %(default)s)",
    )  # fmt: skip
    ap.add_argument("--out", type=Path, help="output directory (not needed with --summary)")
    ap.add_argument(
        "--seed", type=int, default=None,
        help="random seed (default: a new random one, printed and saved in each file)",
    )  # fmt: skip
    ap.add_argument("--force", action="store_true", help="replace the set already in --out")
    ap.add_argument("--summary", action="store_true", help="print counts and estimates only")
    for f in fields(Timing):
        ap.add_argument(
            f"--{f.name.replace('_', '-')}", type=float, default=None, metavar="S",
            help="override the catalogue's [timing] value",
        )  # fmt: skip
    ap.add_argument(
        "--recalibrate", type=float, nargs="+", default=[10.0, 20.0], metavar="MIN",
        help="when the prompter's calibrations come due, as its --recalibrate (default: 10 20)",
    )  # fmt: skip
    args = ap.parse_args(argv)
    if args.out is None and not args.summary:
        ap.error("--out is required (or --summary)")
    try:
        cat = load_catalogue(args.catalogue)
    except (OSError, CatalogueError) as e:
        raise SystemExit(str(e)) from None
    overrides = {f.name: getattr(args, f.name) for f in fields(Timing)}
    bad = [k for k, v in overrides.items() if v is not None and v <= 0]
    if bad:
        ap.error(f"--{bad[0].replace('_', '-')} must be positive")
    timing = replace(cat.timing, **{k: v for k, v in overrides.items() if v is not None})
    try:
        ts = generate_task_blocks(cat, timing, args.seed, [60 * m for m in args.recalibrate])
    except ValueError as e:
        raise SystemExit(str(e)) from None
    plan = plan_dict(ts, args.catalogue, cat.sha256, cat.name)
    if args.summary:
        print_counts(cat)
        print_sessions(ts, plan)
        return
    old = check_old_set(args.out, args.force)
    write_set(args.out, ts, plan, old)
    print(f"wrote {len(plan['files'])} blocks and {PLAN} to {args.out}")
    print_sessions(ts, plan)


if __name__ == "__main__":
    main()
