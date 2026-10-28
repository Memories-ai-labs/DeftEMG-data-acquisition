"""Load recorded trials: EMG on a fitted timeline, continuous segments, prompts, metadata.

One loader for every consumer (replay, timing checks, filtering, model training).
:func:`scan_trials` reads only metadata (cheap), :func:`load_trial` the full arrays. Nothing is
filtered here: :func:`select_trials` applies the rules for a training set and says why each
trial was skipped.

Times are host-clock ns (see ``deftemg_acquisition.clock``). EMG sample times are always fitted with
``fit_sample_times`` (what check-timing saves too), so results never depend on whether
check-timing was run. Prompt times are when each prompt was drawn at the target, falling back
to the schedule where no draw time was logged.
"""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deftemg_acquisition.analysis.timing import emg_segments, fit_sample_times
from deftemg_acquisition.tasks.sequence import DIGITS, LEGACY_LEVELS

PROMPTER_TASK = "scroll_prompter"


def read_session_info(session_dir: Path) -> dict[str, Any]:
    path = Path(session_dir) / "session.json"
    return json.loads(path.read_text()) if path.exists() else {}


def is_simulated(session: str | Path | dict[str, Any]) -> bool:
    """Recorded with the simulated armband (``--simulate``)."""
    info = session if isinstance(session, dict) else read_session_info(Path(session))
    simulated_flag = bool(info.get("metadata", {}).get("simulated"))
    return simulated_flag or info.get("port") in ("simulated", "sim")  # older sessions: port


def read_records(session_dir: Path) -> list[dict[str, Any]]:
    """The session's trials.jsonl lines (empty if there is none)."""
    log = Path(session_dir) / "trials.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def find_record(npz_path: Path) -> dict[str, Any]:
    """The trial's line in its session's trials.jsonl (empty if not found)."""
    for rec in read_records(Path(npz_path).parent):
        if Path(rec["npz_path"]).name == Path(npz_path).name:
            return rec
    return {}


def find_sessions(patterns: list[str] | tuple[str, ...]) -> list[Path]:
    """Session folders (with a trials.jsonl) matching glob patterns, sorted, no duplicates."""
    found: set[Path] = set()
    for pattern in patterns:
        for p in glob.glob(str(pattern)):
            p = Path(p)
            if p.is_file() and p.name == "trials.jsonl":
                p = p.parent
            if (p / "trials.jsonl").exists():
                found.add(p)
    return sorted(found)


@dataclass
class Prompts:
    """Prompts on the host timeline, one row per prompt."""

    onset_ns: np.ndarray  # drawn reaching the target where logged, else scheduled
    offset_ns: np.ndarray
    digits: np.ndarray  # (n, 5) bool, columns = digit_names
    level: np.ndarray  # (n, 5) int, index into level_names; -1 = digit not pressed
    digit_names: tuple[str, ...]
    level_names: tuple[str, ...]
    onset_s: np.ndarray  # scheduled, seconds from the prompter's t = 0
    offset_s: np.ndarray
    reference: str  # "paint", "schedule" or "mixed": what onset_ns / offset_ns are
    legacy_levels: bool  # recorded before level names were saved (low / medium / high)

    def __len__(self) -> int:
        return len(self.onset_ns)


@dataclass
class TrialInfo:
    """Metadata of one trial, available without loading its EMG."""

    id: str  # "<session folder>/<npz stem>"
    npz_path: Path
    session_dir: Path
    record: dict[str, Any]  # trials.jsonl line
    session: dict[str, Any]  # session.json
    simulated: bool
    synthetic: bool  # written by deftemg_acquisition.data.synthetic
    is_prompter: bool
    completed: bool
    legacy_levels: bool
    prompt_hash: str | None  # same prompts (times, digits, levels) -> same hash
    level_names: tuple[str, ...]
    n_prompts: int = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return self.record.get("metadata", {})

    @property
    def label(self) -> str:
        return self.record.get("label") or self.npz_path.stem

    @property
    def subject(self) -> str | None:
        return self.metadata.get("subject") or self.session.get("metadata", {}).get("subject")

    @property
    def is_calibration(self) -> bool:
        """A calibration block (rest + maximum presses), not task data."""
        return self.metadata.get("task_set") == "calibration" or self.level_names == ("max",)

    @property
    def donning(self) -> int | None:
        """The band donning this trial was recorded in (see acquisition.subjects)."""
        d = self.metadata.get("donning", self.session.get("metadata", {}).get("donning"))
        return None if d is None else int(d)

    @property
    def arm(self) -> str:
        meta = self.session.get("metadata", {})
        return (
            self.metadata.get("arm")
            or meta.get("arm")
            or meta.get("hand")
            or self.metadata.get("hand", "right")
        )

    @property
    def filter_status(self) -> dict[str, Any]:
        return self.session.get("filter_status", {})


@dataclass
class Trial:
    info: TrialInfo
    emg_uv: np.ndarray  # (n, E)
    emg_t_ns: np.ndarray  # (n,) fitted host time per sample
    segment: np.ndarray  # (n,) continuous-segment id per sample (breaks: stalls, lost packets)
    imu_t_ns: np.ndarray
    gyro: np.ndarray  # rad/s (m, 3)
    acc: np.ndarray  # m/s^2 (m, 3)
    prompts: Prompts | None
    scroll_start_ns: int | None  # host time of the prompter's t = 0
    arrays: dict[str, np.ndarray] = field(repr=False, default_factory=dict)  # if asked for

    @property
    def id(self) -> str:
        return self.info.id

    def segment_bounds(self) -> list[tuple[int, int]]:
        """(start, end) sample indices of each continuous segment."""
        edges = np.flatnonzero(np.diff(self.segment)) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(self.segment)]])
        return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _prompt_hash(d) -> str:
    h = hashlib.sha1()
    for key, dtype in (("prompt_onset_s", np.float64), ("prompt_offset_s", np.float64),
                       ("prompt_digits", np.uint8), ("prompt_level", np.int8)):  # fmt: skip
        a = np.asarray(d[key])
        a = np.round(a, 3) if dtype is np.float64 else a
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a, dtype=dtype).tobytes())
    return h.hexdigest()[:16]


def scan_trial(npz_path: Path) -> TrialInfo:
    npz_path = Path(npz_path)
    session_dir = npz_path.parent
    session = read_session_info(session_dir)
    record = find_record(npz_path)
    meta = record.get("metadata", {})
    with np.load(npz_path) as d:
        files = set(d.files)
        has_prompts = "prompt_onset_host_ns" in files
        legacy = has_prompts and "level_names" not in files
        level_names = (
            tuple(str(x) for x in d["level_names"])
            if "level_names" in files
            else (LEGACY_LEVELS if has_prompts else ())
        )
        prompt_hash = _prompt_hash(d) if has_prompts else None
        n_prompts = len(d["prompt_onset_host_ns"]) if has_prompts else 0
    return TrialInfo(
        id=f"{session_dir.name}/{npz_path.stem}",
        npz_path=npz_path,
        session_dir=session_dir,
        record=record,
        session=session,
        simulated=is_simulated(session),
        synthetic=bool(session.get("metadata", {}).get("synthetic")),
        is_prompter=has_prompts and meta.get("task", PROMPTER_TASK) == PROMPTER_TASK,
        completed=bool(meta.get("completed", True)),
        legacy_levels=legacy,
        prompt_hash=prompt_hash,
        level_names=level_names,
        n_prompts=n_prompts,
    )


def scan_trials(sessions: list[Path]) -> list[TrialInfo]:
    """Every trial listed in the sessions' trials.jsonl files whose .npz exists."""
    infos = []
    for session_dir in sessions:
        for rec in read_records(session_dir):
            npz = Path(session_dir) / Path(rec["npz_path"]).name
            if npz.exists():
                infos.append(scan_trial(npz))
    return infos


def load_trial(npz_path: Path, keep_arrays: bool = False) -> Trial:
    """A trial with fitted EMG times, segments and prompts (no filtering).

    ``keep_arrays`` also keeps every array of the file in ``Trial.arrays``.
    """
    info = scan_trial(npz_path)
    with np.load(info.npz_path) as d:
        arrays = {k: d[k] for k in d.files}
    emg_t, _ = fit_sample_times(arrays["emg_host_time_ns"], arrays["emg_seq"])
    # Each segment gets its own fitted offset, so a very short one (lost packets) can start a
    # few ms before the previous one ends; keep the timeline non-decreasing for lookups.
    emg_t = np.maximum.accumulate(emg_t)
    prompts = None
    if "prompt_onset_host_ns" in arrays:
        sched_on, sched_off = arrays["prompt_onset_host_ns"], arrays["prompt_offset_host_ns"]
        on, off, reference = sched_on, sched_off, "schedule"
        if "prompt_onset_paint_ns" in arrays:  # when each prompt was actually drawn
            drawn_on, drawn_off = arrays["prompt_onset_paint_ns"], arrays["prompt_offset_paint_ns"]
            on = np.where(drawn_on >= 0, drawn_on, sched_on)
            off = np.where(drawn_off >= 0, drawn_off, sched_off)
            n_drawn = int((drawn_on >= 0).sum() + (drawn_off >= 0).sum())
            reference = {0: "schedule", 2 * len(on): "paint"}.get(n_drawn, "mixed")
        prompts = Prompts(
            onset_ns=on.astype(np.int64),
            offset_ns=off.astype(np.int64),
            digits=arrays["prompt_digits"].astype(bool),
            level=arrays["prompt_level"].astype(np.int64),
            digit_names=tuple(str(x) for x in arrays.get("digit_names", DIGITS)),
            level_names=info.level_names,
            onset_s=arrays["prompt_onset_s"],
            offset_s=arrays["prompt_offset_s"],
            reference=reference,
            legacy_levels=info.legacy_levels,
        )
    scroll = arrays.get("scroll_start_host_ns")
    return Trial(
        info=info,
        emg_uv=arrays["emg_uv"],
        emg_t_ns=emg_t,
        segment=emg_segments(arrays["emg_host_time_ns"], arrays["emg_seq"]),
        imu_t_ns=arrays["imu_host_time_ns"],
        gyro=arrays["gyro_rad_s"],
        acc=arrays["acc_m_s2"],
        prompts=prompts,
        scroll_start_ns=int(scroll) if scroll is not None else None,
        arrays=arrays if keep_arrays else {},
    )


def select_trials(
    infos: list[TrialInfo],
    *,
    exclude_simulated: bool = True,
    include_incomplete: bool = False,
    level_names: tuple[str, ...] = ("low", "high"),
    allow_mixed: bool = False,
) -> tuple[list[TrialInfo], list[tuple[str, str]]]:
    """Trials usable for training, and (id, reason) for every trial left out.

    Usable: a prompter trial, recorded with the armband (unless simulated ones are allowed),
    completed (unless incomplete ones are allowed), whose force levels are ``level_names``.
    Unless ``allow_mixed``, all usable trials must share filter settings and arm, and be all
    synthetic or all recorded; a mix raises.
    """
    usable, skipped = [], []
    for info in infos:
        if exclude_simulated and info.simulated:
            reason = "simulated armband"
        elif not info.is_prompter:
            reason = "not a prompter trial"
        elif info.n_prompts == 0:
            reason = "no prompts"
        elif info.is_calibration:
            reason = "calibration block"
        elif info.legacy_levels:
            reason = "legacy force levels (low/medium/high)"
        elif tuple(info.level_names) != tuple(level_names):
            reason = f"force levels {list(info.level_names)}, expected {list(level_names)}"
        elif not info.completed and not include_incomplete:
            reason = "incomplete block"
        else:
            usable.append(info)
            continue
        skipped.append((info.id, reason))
    if not allow_mixed and usable:
        filters = {json.dumps(i.filter_status, sort_keys=True) for i in usable}
        arms = {i.arm for i in usable}
        kinds = {"synthetic" if i.synthetic else "recorded" for i in usable}
        if len(filters) > 1 or len(arms) > 1 or len(kinds) > 1:
            raise ValueError(
                f"usable trials mix filter settings {sorted(filters)}, arms {sorted(arms)} or "
                f"{' and '.join(sorted(kinds))} data; set data.allow_mixed to use them together"
            )
    return usable, skipped
