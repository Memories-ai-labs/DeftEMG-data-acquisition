"""Accumulate the raw serial stream for one trial and save it raw + decoded."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import protocol as P


@dataclass
class TrialFiles:
    raw: Path
    npz: Path
    data: dict[str, np.ndarray | int]


class TrialRecorder:
    """Chunk sink for :meth:`Armband.attach_sink` that keeps every byte of a trial.

    Alongside the bytes it logs (host receive time, byte offset) for each serial read, so
    every decoded packet can be assigned the host time at which it arrived.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._chunk_t_ns: list[int] = []
        self._chunk_end: list[int] = []
        self._lock = threading.Lock()

    def __call__(self, t_ns: int, data: bytes) -> None:
        with self._lock:
            self._buf += data
            self._chunk_t_ns.append(t_ns)
            self._chunk_end.append(len(self._buf))

    @property
    def n_bytes(self) -> int:
        return len(self._buf)

    def snapshot(self) -> tuple[bytes, np.ndarray, np.ndarray]:
        with self._lock:
            return (
                bytes(self._buf),
                np.asarray(self._chunk_t_ns, dtype=np.int64),
                np.asarray(self._chunk_end, dtype=np.int64),
            )

    def decode(self, filter_enabled: bool = True) -> dict[str, np.ndarray | int]:
        raw, chunk_t_ns, chunk_end = self.snapshot()
        d = P.decode(raw, filter_enabled).as_dict()
        # A packet is fully received in the first chunk whose end offset covers its last byte.
        length = P.packet_len(filter_enabled)
        for kind in ("emg", "imu"):
            last_byte = d[f"{kind}_offset"] + length
            idx = np.searchsorted(chunk_end, last_byte, side="left")
            d[f"{kind}_host_time_ns"] = (
                chunk_t_ns[np.minimum(idx, len(chunk_t_ns) - 1)]
                if len(chunk_t_ns)
                else np.empty(0, np.int64)
            )
        d["chunk_host_time_ns"] = chunk_t_ns
        d["chunk_end_offset"] = chunk_end
        return d

    def save(
        self, stem: Path, filter_enabled: bool = True, extra: dict[str, object] | None = None
    ) -> TrialFiles:
        """Write ``<stem>.bin`` (exact bytes received) and ``<stem>.npz`` (decoded arrays)."""
        stem.parent.mkdir(parents=True, exist_ok=True)
        raw, _, _ = self.snapshot()
        arrays = self.decode(filter_enabled)
        for k, v in (extra or {}).items():
            arrays[k] = np.asarray(v)
        files = TrialFiles(stem.with_suffix(".bin"), stem.with_suffix(".npz"), arrays)
        files.raw.write_bytes(raw)
        np.savez_compressed(files.npz, **arrays)
        return files
