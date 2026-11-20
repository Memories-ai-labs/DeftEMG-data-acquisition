"""Publish the armband stream live over Lab Streaming Layer (LSL), in parallel to recording.

:class:`LiveStreamer` is an :class:`Armband` sink. The reader thread only drops each chunk into a
queue; decoding and publishing happen on a separate worker thread, so a slow consumer can never
delay the reader (or the trial recorder, which is the authoritative copy of the data).

Three LSL streams are created, all sharing ``source_id`` so viewers can match them up:

- EMG: 18 channels, 2000 Hz, float32 microvolts. With the armband's filters off it can be
  filtered causally on the way out (``emg_filter``, :class:`LiveFilter`); recordings stay raw.
- IMU: gyro xyz (rad/s), acc xyz (m/s^2), temperature, battery mV/%, dropped-packet total; 208 Hz
- Markers: irregular strings such as ``trial_start|3|pinch`` and ``trial_end|3``
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np
import pylsl
from scipy import signal

from deftemg_acquisition.analysis.filtering import (
    NOTCHES,
    FilterConfig,
    design_filter,
    named_filter_config,
    notch_frequencies,
)
from deftemg_acquisition.clock import host_ns
from deftemg_acquisition.errors import ConfigError

from ..lsl import STREAM_PREFIX, make_outlet
from . import protocol as P

log = logging.getLogger(__name__)

EMG_STREAM_TYPE = "EMG"
IMU_STREAM_TYPE = "IMU"
MARKER_STREAM_TYPE = "Markers"

EMG_CHANNELS = [f"ch{i}" for i in range(1, P.N_EMG_CHANNELS + 1)]
IMU_CHANNELS = [
    "gyro_x", "gyro_y", "gyro_z",
    "acc_x", "acc_y", "acc_z",
    "temperature_c", "battery_mv", "battery_pct", "dropped_total",
]  # fmt: skip
IMU_UNITS = ["rad/s"] * 3 + ["m/s^2"] * 3 + ["degC", "mV", "percent", "count"]

# Live-stream notch choices: see analysis.filtering.NOTCHES.
LIVE_NOTCHES = NOTCHES


def live_filter_config(highpass_hz: float | None, notch: str) -> FilterConfig:
    """The live stream's filter: the offline one (``analysis.filtering``), run causally."""
    try:
        return named_filter_config(highpass_hz, notch, zero_phase=False)
    except ConfigError as e:
        raise ValueError(str(e)) from None


def describe_filter(cfg: FilterConfig | None, fs_hz: float = P.EMG_RATE_HZ) -> str:
    """E.g. "high-pass 20 Hz, notch 60, 180, 300, 420 Hz"; "none" when nothing is filtered."""
    if cfg is None:
        return "none"
    parts = [f"high-pass {cfg.highpass_hz:g} Hz"] if cfg.highpass_hz else []
    if notches := notch_frequencies(cfg, fs_hz):
        parts.append(f"notch {', '.join(f'{f:g}' for f in notches)} Hz")
    return ", ".join(parts) or "none"


class LiveFilter:
    """``cfg``'s filter run causally over a stream of (n, channels) chunks, keeping its state
    from one chunk to the next (``cfg.zero_phase`` is ignored: a live stream can't look ahead).

    The state starts as if the first sample had been held forever (``sosfilt_zi * x[0]``), so
    the electrode offset of raw EMG doesn't ring through the high-pass: a live stream can't be
    de-meaned first. :meth:`reset` starts over that way at the next chunk, after a break.
    """

    def __init__(self, cfg: FilterConfig, fs_hz: float = P.EMG_RATE_HZ):
        self.cfg = cfg
        self.sos = design_filter(cfg, fs_hz)  # None: nothing to filter
        self._zi_unit = None if self.sos is None else signal.sosfilt_zi(self.sos)
        self._zi: np.ndarray | None = None

    def reset(self) -> None:
        self._zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.sos is None or not len(x):
            return x
        if self._zi is None:
            self._zi = self._zi_unit[:, :, None] * x[0]
        y, self._zi = signal.sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


class GapFiller:
    """Fills the EMG samples missing between packets, so the live stream keeps its time axis.

    Packets lost on a wireless link leave a gap of a few samples. A filter run across the gap
    sees the signal jump, and the mains hum's phase with it, and rings on every channel at once.
    The missing samples are interpolated instead (a straight line between the samples either
    side), which a filter takes as a smooth continuation. A gap longer than ``max_fill`` samples
    is not filled; ``long_gap`` then says so, for the caller to restart its filter.

    EMG and IMU packets share one sequence counter, so the samples missing before an EMG packet
    are its sequence step less one, less the IMU packets that sit in between. A lost IMU packet
    counts as a lost EMG sample, one in ten at most of the gap.
    """

    def __init__(self, max_fill: int = 40):
        self.max_fill = max_fill
        self.reset()

    def reset(self) -> None:
        self._last_seq: int | None = None  # last EMG packet's sequence number
        self._last_uv: np.ndarray | None = None
        self._imu_since = 0  # IMU packets since that one

    def __call__(self, d: P.DecodedStream) -> tuple[np.ndarray, bool]:
        """``d.emg_uv`` with the missing samples filled in, and whether a gap was too long."""
        seq, uv = d.emg_seq, d.emg_uv
        imu = d.imu_seq
        if len(seq) == 0:
            self._imu_since += len(imu)
            return uv, False
        prev_seq = np.concatenate(
            [[self._last_seq if self._last_seq is not None else seq[0] - 1], seq[:-1]]
        )
        between = np.searchsorted(imu, seq) - np.searchsorted(imu, prev_seq, side="right")
        between[0] += self._imu_since if self._last_seq is not None else 0
        missing = np.maximum(seq - prev_seq - 1 - between, 0)
        if self._last_seq is None:
            missing[0] = 0
        long_gap = bool((missing > self.max_fill).any())
        fill = np.flatnonzero((missing > 0) & (missing <= self.max_fill))
        if fill.size:
            before = np.vstack([self._last_uv if self._last_uv is not None else uv[:1], uv[:-1]])
            pieces, at = [], 0
            for k in fill:
                m = int(missing[k])
                pieces.append(uv[at:k])
                w = (np.arange(1, m + 1) / (m + 1))[:, None]
                pieces.append(before[k] + (uv[k] - before[k]) * w)
                at = k
            pieces.append(uv[at:])
            uv = np.concatenate(pieces)
        self._last_seq, self._last_uv = int(seq[-1]), d.emg_uv[-1].copy()
        self._imu_since = int(len(imu) - np.searchsorted(imu, seq[-1], side="right"))
        return uv, long_gap


class LiveStreamer:
    """Armband sink that decodes chunks on a worker thread and pushes them to LSL outlets."""

    def __init__(
        self,
        source_id: str,
        filter_enabled: bool = True,
        max_queued_chunks: int = 5000,
        emg_filter: FilterConfig | None = None,
    ):
        self.source_id = source_id
        self.decoder = P.StreamDecoder(filter_enabled)
        self.emg_filter = None if emg_filter is None else LiveFilter(emg_filter)
        self._last_t_ns = 0  # receive time of the previous chunk
        self._filler = GapFiller()
        # (receive time, bytes, chunks discarded before this one)
        self._queue: queue.Queue[tuple[int, bytes, int] | None] = queue.Queue(max_queued_chunks)
        self._worker: threading.Thread | None = None
        # Host clock -> LSL clock (seconds), so samples carry their serial receive time.
        self._lsl_offset = pylsl.local_clock() - host_ns() / 1e9
        self.n_chunks_discarded = 0

        self._emg = make_outlet(
            f"{STREAM_PREFIX}-EMG", EMG_STREAM_TYPE, EMG_CHANNELS, ["uV"] * len(EMG_CHANNELS),
            P.EMG_RATE_HZ, pylsl.cf_float32, source_id,
            device_filter="on" if filter_enabled else "off",
            software_filter=describe_filter(emg_filter),
        )  # fmt: skip
        self._imu = make_outlet(
            f"{STREAM_PREFIX}-IMU", IMU_STREAM_TYPE, IMU_CHANNELS, IMU_UNITS,
            P.IMU_RATE_HZ, pylsl.cf_float32, source_id,
        )  # fmt: skip
        self._markers = make_outlet(
            f"{STREAM_PREFIX}-Markers", MARKER_STREAM_TYPE, ["marker"], [""],
            pylsl.IRREGULAR_RATE, pylsl.cf_string, source_id,
        )  # fmt: skip

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="lsl-streamer", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=2.0)
        self._worker = None

    def __call__(self, t_ns: int, data: bytes) -> None:
        """Called on the armband reader thread: never blocks."""
        try:
            self._queue.put_nowait((t_ns, data, self.n_chunks_discarded))
        except queue.Full:
            self.n_chunks_discarded += 1

    def push_marker(self, marker: str) -> None:
        self._markers.push_sample([marker])

    def _run(self) -> None:
        discarded_seen = 0
        while (item := self._queue.get()) is not None:
            t_ns, data, discarded = item
            if discarded != discarded_seen:
                # The live view fell behind and lost bytes just before this chunk; resync rather
                # than splice packets.
                log.warning("live stream fell behind, %d chunks discarded", discarded)
                discarded_seen = discarded
                self.decoder.reset()
                self._filler.reset()
                if self.emg_filter is not None:
                    self.emg_filter.reset()
            try:
                self._publish(t_ns, data)
            except Exception:
                log.exception("live stream publish failed")

    def _publish(self, t_ns: int, data: bytes) -> None:
        d = self.decoder.feed(data)
        ts = t_ns / 1e9 + self._lsl_offset  # receive time of the last packet in this chunk
        if self.emg_filter is not None and t_ns - self._last_t_ns > P.STALL_GAP_S * 1e9:
            self.emg_filter.reset()  # the stream was interrupted: start over from the next sample
        self._last_t_ns = t_ns
        if len(d.emg_seq):
            emg_uv, long_gap = self._filler(d)
            if long_gap and self.emg_filter is not None:
                self.emg_filter.reset()  # too much missing to bridge: start over after it
            emg = emg_uv if self.emg_filter is None else self.emg_filter(emg_uv)
            self._emg.push_chunk(emg.astype(np.float32), ts)
        if len(d.imu_seq):
            imu = np.column_stack([
                d.gyro_rad_s, d.acc_m_s2, d.temperature_c, d.battery_mv, d.battery_pct,
                np.full(len(d.imu_seq), self.decoder.n_dropped_total),
            ]).astype(np.float32)  # fmt: skip
            self._imu.push_chunk(imu, ts)
