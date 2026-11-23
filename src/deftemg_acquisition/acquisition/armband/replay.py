"""Replay a recording as if it were the armband: the live path without hardware.

    stream = RecordedStream.load(["data/raw/<session>"])            # every trial, in order
    device = ReplayArmband(stream, speed=1.0)
    config = SessionConfig(filter_enabled=stream.filter_status.enabled, ...)
    with AcquisitionSession(config, device=device) as session: ...  # sinks get the bytes

Every trial keeps the exact bytes it received (``.bin``) and when each serial read arrived
(``chunk_host_time_ns`` / ``chunk_end_offset`` in its ``.npz``). ``ReplayArmband`` hands those
same chunks to the attached sinks (the trial recorder, the live stream, the effort meter), from
a thread, as ``Armband``'s reader does: ``speed`` 1 at the pace they arrived, faster for tests.
Their timestamps keep the original spacing on the host clock whatever the speed, so a stall
stays a stall and a re-recorded trial decodes to the same samples and segments.

Each trial is trimmed to whole packets (a trial starts and ends mid-stream), and trials follow
each other ``gap_s`` apart. The armband's filter setting comes from the recording (packets
differ with it on or off); ``configure_filters`` refuses another.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deftemg_acquisition.clock import host_ns

from . import protocol as P
from .device import Armband, DeviceError

log = logging.getLogger(__name__)


@dataclass
class RecordedStream:
    """Recorded serial chunks, in order, with their arrival times (host ns, original spacing
    within a trial; ``gap_s`` between trials)."""

    chunks: list[bytes]
    t_ns: np.ndarray  # (n,) int64
    filter_status: P.FilterStatus
    trials: list[tuple[str, int]] = field(default_factory=list)  # (npz path, first chunk)

    @property
    def duration_s(self) -> float:
        return float(self.t_ns[-1] - self.t_ns[0]) / 1e9 if len(self.t_ns) else 0.0

    @classmethod
    def load(cls, paths: list[str | Path], gap_s: float = 0.5) -> RecordedStream:
        """From trial ``.npz`` files (with their ``.bin``) and session folders (every trial,
        calibration blocks included, in recording order)."""
        npzs = []
        for p in map(Path, paths):
            if p.suffix == ".npz":
                npzs.append(p)
            else:
                from deftemg_acquisition.data.trials import read_records

                npzs += [p / Path(r["npz_path"]).name for r in read_records(p)]
        if not npzs:
            raise FileNotFoundError(f"no trials in {[str(p) for p in paths]}")
        chunks, times, trials, status = [], [], [], None
        next_t = 0
        for npz in npzs:
            c, t, s = _trial_chunks(npz)
            if status is not None and s != status:
                raise ValueError(f"{npz}: filter settings {s}, the trials before {status}")
            status = s
            if not c:
                continue
            trials.append((str(npz), len(chunks)))
            t = t - t[0] + next_t
            next_t = int(t[-1]) + round(gap_s * 1e9)
            chunks += c
            times.append(t)
        return cls(chunks, np.concatenate(times).astype(np.int64), status, trials)


def emg_packets(emg_uv: np.ndarray, seq0: int = 0) -> list[bytes]:
    """EMG packets as the armband sends them with its filter off (24-bit, 1/60 µV), one per
    sample, sequence numbers from ``seq0``: for recordings without their bytes (synthetic
    sessions) and for tests."""
    counts = np.clip(np.round(np.asarray(emg_uv) * 60), -(2**23), 2**23 - 1).astype(">i4")
    payload = counts.view(np.uint8).reshape(len(counts), -1, 4)[:, :, 1:].reshape(len(counts), -1)
    return [P.SYNC + bytes([P.TYPE_EMG, (seq0 + i) & 0xFF]) + row.tobytes()
            for i, row in enumerate(payload)]  # fmt: skip


def _trial_chunks(npz: Path) -> tuple[list[bytes], np.ndarray, P.FilterStatus]:
    """One trial's chunks (trimmed to whole packets), arrival times and filter settings. A trial
    without its ``.bin`` (synthetic) is sent as packets made from its EMG (``emg_packets``),
    eight per chunk, at their recorded arrival times."""
    from deftemg_acquisition.data.trials import read_session_info

    info = read_session_info(npz.parent).get("filter_status") or {}
    status = P.FilterStatus(bool(info.get("enabled", True)), int(info.get("powerline_hz", 60)),
                            int(info.get("highpass_hz", 60)))  # fmt: skip
    if not npz.with_suffix(".bin").exists():
        with np.load(npz) as d:
            uv, t = d["emg_uv"], d["emg_host_time_ns"].astype(np.int64)
        packets = emg_packets(uv)
        ends = list(range(8, len(packets), 8)) + [len(packets)]
        chunks = [b"".join(packets[a:b]) for a, b in zip([0, *ends[:-1]], ends, strict=True)]
        return chunks, t[np.array(ends) - 1], P.FilterStatus(False, status.powerline_hz,
                                                             status.highpass_hz)  # fmt: skip
    raw = npz.with_suffix(".bin").read_bytes()
    with np.load(npz) as d:
        ends, t = d["chunk_end_offset"].astype(np.int64), d["chunk_host_time_ns"].astype(np.int64)
    length = P.packet_len(status.enabled)
    starts = P.find_packets(raw, length)
    if not len(starts):
        return [], t[:0], status
    lo, hi = int(starts[0]), int(starts[-1]) + length
    bounds = np.clip(np.concatenate([[0], ends]), lo, hi)
    keep = np.flatnonzero(np.diff(bounds) > 0)
    return [raw[bounds[i] : bounds[i + 1]] for i in keep], t[keep], status


class ReplayArmband(Armband):
    """An ``Armband`` whose bytes come from a ``RecordedStream`` instead of a serial port.

    ``start_acquisition`` starts delivering (from the start, or from where ``stop_acquisition``
    left off); ``finished`` is set when the recording has run out. Commands are accepted and do
    nothing, as no device is there to receive them.
    """

    def __init__(self, stream: RecordedStream, speed: float = 1.0):
        super().__init__("replay")
        if speed <= 0:
            raise ValueError(f"speed {speed}: must be > 0")
        self.stream, self.speed = stream, speed
        self.finished = threading.Event()
        self.n_delivered = 0  # chunks handed to the sinks so far
        self.late_ms: list[float] = []  # how far behind schedule each chunk was delivered
        self._feeder: threading.Thread | None = None
        self._stop_feed = threading.Event()

    # Armband's interface, without a port --------------------------------------------------

    def open(self) -> None:
        pass

    def close(self) -> None:
        self.stop_acquisition()

    def ensure_stopped(self) -> None:
        self.stop_acquisition()

    def filter_status(self) -> P.FilterStatus:
        return self.stream.filter_status

    def configure_filters(self, powerline_hz: int, highpass_hz: int, enabled: bool | None = None):
        status = self.stream.filter_status
        if enabled is not None and enabled != status.enabled:
            state = "on" if status.enabled else "off"
            raise DeviceError(f"the recording was made with the armband's filter {state}: "
                              "replay it with the same setting")  # fmt: skip
        return status

    def set_timestamp(self, unix_ms: int) -> None:
        pass

    def send(self, cmd: str, expect: str | None = None, timeout: float = 1.0,
             allow_unknown: bool = False) -> bytes:  # fmt: skip
        return b""

    def start_acquisition(self) -> None:
        if self._feeder is not None:
            return
        self._stop_feed.clear()
        self._feeder = threading.Thread(target=self._feed, name="replay-feeder", daemon=True)
        self.acquiring = True
        self._feeder.start()

    def stop_acquisition(self) -> None:
        self._stop_feed.set()
        if self._feeder is not None:
            self._feeder.join(timeout=2.0)
            self._feeder = None
        self.acquiring = False

    # Delivery --------------------------------------------------------------------------------

    def _feed(self) -> None:
        t = self.stream.t_ns
        first = self.n_delivered
        if first >= len(t):
            self.finished.set()
            return
        start_perf, start_host = time.perf_counter(), host_ns()
        for i in range(first, len(t)):
            due = (t[i] - t[first]) / 1e9 / self.speed
            wait = due - (time.perf_counter() - start_perf)
            if wait > 0.001 and self._stop_feed.wait(wait):
                return
            if self._stop_feed.is_set():
                return
            self.late_ms.append(max(0.0, -wait) * 1e3)
            stamp = start_host + int(t[i] - t[first])  # the original spacing, on the host clock
            for name, sink in self._sinks.items():
                try:
                    sink(stamp, self.stream.chunks[i])
                except Exception:
                    log.exception("sink %r failed", name)
            self.n_delivered = i + 1
        self.finished.set()
