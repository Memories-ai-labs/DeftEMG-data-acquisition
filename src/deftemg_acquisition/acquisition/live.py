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

from . import protocol as P

log = logging.getLogger(__name__)

EMG_STREAM_TYPE = "EMG"
IMU_STREAM_TYPE = "IMU"
MARKER_STREAM_TYPE = "Markers"
STREAM_PREFIX = "Waveletech"

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


def make_outlet(
    name, stype, channels, units, rate, fmt, source_id, **desc: str
) -> pylsl.StreamOutlet:
    """An LSL outlet with channel labels and units. Another stream joins the session's by
    sharing its ``source_id`` (``LiveStreamer.source_id``), named with ``STREAM_PREFIX``."""
    info = pylsl.StreamInfo(name, stype, len(channels), rate, fmt, source_id)
    info.desc().append_child_value("manufacturer", "WAVELETECH")
    for key, value in desc.items():
        info.desc().append_child_value(key, value)
    chans = info.desc().append_child("channels")
    for label, unit in zip(channels, units, strict=True):
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
    return pylsl.StreamOutlet(info)


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
            emg = d.emg_uv if self.emg_filter is None else self.emg_filter(d.emg_uv)
            self._emg.push_chunk(emg.astype(np.float32), ts)
        if len(d.imu_seq):
            imu = np.column_stack([
                d.gyro_rad_s, d.acc_m_s2, d.temperature_c, d.battery_mv, d.battery_pct,
                np.full(len(d.imu_seq), self.decoder.n_dropped_total),
            ]).astype(np.float32)  # fmt: skip
            self._imu.push_chunk(imu, ts)
