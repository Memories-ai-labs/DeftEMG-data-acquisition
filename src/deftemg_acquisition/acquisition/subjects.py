"""Subject and donning ids for recording sessions.

A *donning* is one continuous wearing of the band: electrode positions and skin contact stay
the same within it and change when the band is taken off and put back on. Models are meant to
be judged across donnings, so every session records

- ``subject``: the user's id;
- ``donning``: 1, 2, ... per subject, incremented when the band was re-donned since the
  subject's last session;
- ``donning_started``: when the first session of this donning was opened (ISO time);
- ``session_in_donning``: 1 for that first session, then 2, 3, ... while the band stays on;
- ``previous_session``: the folder name of the subject's last session, if any.

The history comes from the ``session.json`` files in the data folder (so it is per
``--data-dir``). Only sessions with at least one recorded trial count, and real and simulated
sessions keep separate histories, so trying things out with ``--simulate`` never uses up a
real donning id.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Donning:
    subject: str
    donning: int
    donning_started: str
    session_in_donning: int
    previous_session: str | None

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PastSession:
    folder: str
    opened_at: str
    when_ns: int  # for ordering: the session's wall-clock anchor (UTC), else best effort
    n_trials: int
    donning: int | None
    donning_started: str | None
    session_in_donning: int


def _simulated(info: dict[str, Any]) -> bool:
    meta = info.get("metadata") or {}
    return bool(meta.get("simulated")) or info.get("port") in ("simulated", "sim")


def _int_or_none(x) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def subject_sessions(data_dir: Path, subject: str, simulated: bool = False) -> list[PastSession]:
    """The subject's earlier sessions with at least one trial, oldest first.

    Real and simulated sessions are kept apart (``simulated``). Folders whose session.json
    can't be read are skipped.
    """
    out = []
    for info_path in Path(data_dir).glob("*/session.json"):
        try:
            info = json.loads(info_path.read_text())
        except (OSError, ValueError):
            continue
        meta = info.get("metadata") if isinstance(info, dict) else None
        if not isinstance(meta, dict) or meta.get("subject") != subject:
            continue
        if _simulated(info) != simulated:
            continue
        log = info_path.parent / "trials.jsonl"
        n_trials = sum(1 for line in log.read_text().splitlines() if line.strip()) if (
            log.exists()) else 0  # fmt: skip
        if n_trials == 0:  # opened and closed without recording: not a session to continue
            continue
        anchor = info.get("clock_anchor")
        when = _int_or_none(anchor.get("wall_ns")) if isinstance(anchor, dict) else None
        if when is None:
            try:
                when = int(datetime.fromisoformat(info["opened_at"]).timestamp() * 1e9)
            except (KeyError, TypeError, ValueError):
                when = int(info_path.stat().st_mtime * 1e9)
        out.append(
            PastSession(
                folder=info_path.parent.name,
                opened_at=str(info.get("opened_at", "")),
                when_ns=when,
                n_trials=n_trials,
                donning=_int_or_none(meta.get("donning")),
                donning_started=meta.get("donning_started"),
                session_in_donning=_int_or_none(meta.get("session_in_donning")) or 1,
            )
        )
    return sorted(out, key=lambda s: (s.when_ns, s.folder))


def next_donning(
    history: list[PastSession], subject: str, redonned: bool, now: datetime | None = None
) -> Donning:
    """The donning a new session of ``subject`` belongs to, given its earlier sessions."""
    now_iso = (now or datetime.now()).isoformat(timespec="seconds")
    if not history:
        return Donning(subject, 1, now_iso, 1, None)
    last = history[-1]
    known = [s.donning for s in history if s.donning is not None]
    if redonned or last.donning is None:  # sessions from before donning ids: start counting
        return Donning(subject, max(known, default=0) + 1, now_iso, 1, last.folder)
    return Donning(
        subject,
        last.donning,
        last.donning_started or last.opened_at or now_iso,
        last.session_in_donning + 1,
        last.folder,
    )


def ask_donning(
    data_dir: Path,
    subject: str | None = None,
    redonned: bool | None = None,
    simulated: bool = False,
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> Donning:
    """Subject and donning for a new session, asking for whatever wasn't given.

    Asks for the user ID if ``subject`` is None, and "Did you re-don the band since your last
    session?" if ``redonned`` is None and the subject's last session has a donning id.
    """
    try:
        while not subject:
            subject = ask("User ID: ").strip()
        history = subject_sessions(data_dir, subject, simulated)
        if not history:
            similar = _similar_subjects(data_dir, subject, simulated)
            if similar:
                say(f"note: no sessions for {subject!r}, but there are for {similar}")
        elif history[-1].donning is None:
            say(f"{subject}'s earlier sessions have no donning id; starting a new donning")
            redonned = True
        while history and redonned is None:
            last = history[-1]
            answer = (
                ask(
                    f"Last session {_when(last)} ({last.n_trials} trials, donning {last.donning}). "
                    "Did you re-don the band since your last session? [y/n]: "
                )
                .strip()
                .lower()
            )
            redonned = {"y": True, "yes": True, "n": False, "no": False}.get(answer)
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nno answer: stopping") from None
    d = next_donning(history, subject, bool(redonned))
    if not history:
        say(f"First session for {subject}: donning {d.donning}")
    elif d.session_in_donning == 1:
        say(f"{subject}: new donning {d.donning}")
    else:
        hours = (datetime.now().timestamp() * 1e9 - history[-1].when_ns) / 3.6e12
        say(f"{subject}: donning {d.donning} (session {d.session_in_donning} of this donning)")
        if hours > 12:
            say(f"note: the last session was {hours:.0f} h ago; was the band really kept on?")
    return d


def _when(s: PastSession) -> str:
    try:
        return datetime.fromisoformat(s.opened_at).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return s.folder


def _similar_subjects(data_dir: Path, subject: str, simulated: bool) -> list[str]:
    names = set()
    for info_path in Path(data_dir).glob("*/session.json"):
        try:
            meta = json.loads(info_path.read_text()).get("metadata") or {}
        except (OSError, ValueError, AttributeError):
            continue
        other = meta.get("subject") if isinstance(meta, dict) else None
        if isinstance(other, str) and other != subject and other.lower() == subject.lower():
            names.add(other)
    return sorted(names)


def add_arguments(parser) -> None:
    """``--subject`` and ``--redon`` / ``--same-donning`` (for scripted runs) on a command."""
    parser.add_argument("--subject", help="user ID (asked for if not given)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--redon", dest="redonned", action="store_true",
        help="(scripted runs) the band was re-donned since the last session",
    )  # fmt: skip
    group.add_argument(
        "--same-donning", dest="redonned", action="store_false",
        help="(scripted runs) the band stayed on since the last session",
    )  # fmt: skip
    parser.set_defaults(redonned=None)


def donning_from_args(args, data_dir: Path) -> Donning:
    """Ask on the terminal for anything missing. Simulated sessions don't ask: subject "sim"
    unless ``--subject`` is given. Without a terminal, real sessions need the flags."""
    subject, redonned = args.subject, args.redonned
    simulated = bool(getattr(args, "simulate", False))
    if simulated:
        subject = subject or "sim"
        if redonned is None and not sys.stdin.isatty():
            redonned = False
    elif not sys.stdin.isatty():
        if subject is None:
            raise SystemExit("no terminal to ask: pass --subject (and --redon / --same-donning)")
        if redonned is None and subject_sessions(data_dir, subject):
            raise SystemExit(f"{subject} has recorded before: pass --redon or --same-donning")
    return ask_donning(data_dir, subject, redonned, simulated)
