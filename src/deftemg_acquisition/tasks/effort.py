"""Live effort meter: overall EMG amplitude relative to the calibration maximum of what is
prompted, for the prompter's effort bar (``TaskWindow(effort_meter=True)``).

Effort is one number whatever digits a prompt involves: the EMG amplitude above the rest noise
over the last ``WINDOW_S`` (RMS over channels of sqrt(max(RMS^2 - SD_rest^2, 0)), as the
calibration measures it), divided by the amplitude of the prompt's maximum in the session's
latest calibration block:

- a single digit: that digit's maximum press;
- a pinch / squeeze (thumb + finger): that pinch's maximum, from a pinch / squeeze calibration;
- other chords: the digits' maxima combined by adding EMG power (sqrt of the sum of squares),
  so every digit at 30 % reads 30 % (an estimate: a calibration has no chords of its own).

0 is rest, 1 the calibration maximum. Both the live EMG and the reference go through the live
stream's causal filter (``AcquisitionSession.live_emg_filter``), so they compare like with
like. The meter is a sink on the armband: chunks are decoded and filtered on a worker thread,
like the LSL stream, and the window is read from the GUI thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deftemg_acquisition.acquisition import protocol as P
from deftemg_acquisition.analysis.filtering import EmgFilter
from deftemg_acquisition.data.calibration import Calibration, calibration_from_trial
from deftemg_acquisition.data.trials import load_trial

from .sequence import DIGITS, LEVELS

log = logging.getLogger(__name__)

WINDOW_S = 0.15  # the amplitude window: ~0.1-0.15 s behind the press
# Target zones per level, as fractions of the maximum: targets 30 % and 75 %, +- 10 %.
ZONES = {"low": (0.20, 0.40), "high": (0.65, 0.85)}


def amplitude_uv(rms_uv: np.ndarray, rest_sd_uv: np.ndarray) -> float:
    """Overall EMG amplitude above the rest noise, from per-channel RMS (as the calibration)."""
    excess = np.maximum(np.square(rms_uv) - np.square(rest_sd_uv), 0.0)
    return float(np.sqrt(np.mean(excess)))


def chord_key(digits) -> str:
    """Digits in DIGITS order joined as the calibration names efforts: "index", "thumb+index"."""
    return "+".join(sorted(digits, key=DIGITS.index))


@dataclass
class EffortReference:
    """What 100 % means, from one calibration block (measured with the live filter)."""

    calibration_id: str
    rest_sd_uv: np.ndarray  # (E,)
    efforts_uv: dict[str, float]  # "index" / "thumb+index" -> amplitude of its maximum
    digits_uv: dict[str, float]  # digit -> amplitude of its best maximum (chord fallback)

    @classmethod
    def from_calibration(cls, cal: Calibration) -> EffortReference:
        def amp(v) -> float:
            return float(np.sqrt(np.mean(np.square(v))))  # rest noise already removed

        return cls(cal.trial_id, np.asarray(cal.rest_sd_uv, dtype=np.float64),
                   {k: amp(v) for k, v in cal.chord_max_rms_uv.items()},
                   {d: amp(v) for d, v in cal.max_rms_uv.items()})  # fmt: skip

    def maximum_uv(self, digits) -> float | None:
        """Amplitude of the maximum for these digits (see the module docstring), or None."""
        if not digits:
            return None
        key = chord_key(digits)
        if key in self.efforts_uv:
            return self.efforts_uv[key] or None
        parts = [self.digits_uv.get(d) for d in digits]
        if any(a is None for a in parts):
            return None
        return float(np.sqrt(np.sum(np.square(parts)))) or None


def reference_from_trial(npz_path: Path, emg_filter: EmgFilter | None) -> EffortReference | None:
    """The reference from a recorded calibration block, filtered with ``emg_filter`` (the live
    filter, causal); None (with a warning) when the block can't be used."""
    try:
        cal = calibration_from_trial(load_trial(Path(npz_path)), emg_filter)
    except Exception as e:  # noqa: BLE001 (an optional display must never stop a recording)
        log.warning("effort meter: calibration block %s can't be used (%s: %s)", npz_path,
                    type(e).__name__, e)  # fmt: skip
        return None
    for w in cal.warnings:
        log.warning("effort meter: %s: %s", cal.trial_id, w)
    if not cal.max_rms_uv:
        log.warning("effort meter: %s has no usable maximum; meter off", cal.trial_id)
        return None
    return EffortReference.from_calibration(cal)


class EffortMeter:
    """Armband sink keeping the last ``window_s`` of live EMG; :meth:`effort` reads it."""

    def __init__(
        self,
        filter_enabled: bool = True,
        emg_filter: EmgFilter | None = None,
        window_s: float = WINDOW_S,
        fs_hz: float = P.EMG_RATE_HZ,
        max_queued_chunks: int = 5000,
    ):
        from deftemg_acquisition.acquisition.live import LiveFilter  # lazily: live.py loads liblsl

        self.decoder = P.StreamDecoder(filter_enabled)
        self.emg_filter = emg_filter
        self._filter = None if emg_filter is None else LiveFilter(emg_filter.config(), fs_hz)
        self.window_s = window_s
        self._buf = np.zeros((max(2, round(window_s * fs_hz)), P.N_EMG_CHANNELS))
        self._filled = 0
        self._pos = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[int, bytes, int] | None] = queue.Queue(max_queued_chunks)
        self._worker: threading.Thread | None = None
        self._last_t_ns = 0
        self.n_chunks_discarded = 0
        self.reference: EffortReference | None = None

    def start(self) -> None:
        self._worker = threading.Thread(target=self._run, name="effort-meter", daemon=True)
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

    def _run(self) -> None:
        seen = 0
        while (item := self._queue.get()) is not None:
            t_ns, data, discarded = item
            if discarded != seen:  # fell behind and lost bytes: resync, start the window over
                seen = discarded
                self.decoder.reset()
                self._restart()
            try:
                d = self.decoder.feed(data)
                if len(d.emg_seq):
                    self.push(d.emg_uv, t_ns)
            except Exception:
                log.exception("effort meter: chunk failed")

    def _restart(self) -> None:
        if self._filter is not None:
            self._filter.reset()
        with self._lock:
            self._filled = 0

    def push(self, emg_uv: np.ndarray, t_ns: int | None = None) -> None:
        """Add (n, channels) EMG in µV, as recorded (filtered here)."""
        if t_ns is not None:
            if self._last_t_ns and t_ns - self._last_t_ns > P.STALL_GAP_S * 1e9:
                self._restart()  # the stream was interrupted: the filter starts over
            self._last_t_ns = t_ns
        x = np.asarray(emg_uv, dtype=np.float64)
        if self._filter is not None:
            x = self._filter(x)
        n = len(self._buf)
        x = x[-n:]
        with self._lock:
            idx = (self._pos + np.arange(len(x))) % n
            self._buf[idx] = x
            self._pos = (self._pos + len(x)) % n
            self._filled = min(n, self._filled + len(x))

    def rms_uv(self) -> np.ndarray | None:
        """(E,) RMS per channel over the window (its mean removed, as the calibration's), or
        None until the window is full."""
        with self._lock:
            if self._filled < len(self._buf):
                return None
            x = self._buf.copy()
        return np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))

    def effort(self, digits) -> float | None:
        """Current effort for a prompt of ``digits`` (0 = rest, 1 = its calibration maximum), or
        None without a reference, a maximum for these digits, or a full window."""
        ref = self.reference
        top = None if ref is None else ref.maximum_uv(digits)
        rms = self.rms_uv()
        if top is None or rms is None:
            return None
        return amplitude_uv(rms, ref.rest_sd_uv) / top


def prompt_level(levels) -> str | None:
    """The level a single meter shows for a prompt: its highest task level (chords press every
    digit at the same level; with independent levels, the firmer one)."""
    known = [lv for lv in levels if lv in LEVELS]
    return max(known, key=LEVELS.index) if known else None
