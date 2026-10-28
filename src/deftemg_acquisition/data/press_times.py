"""When each prompted press actually starts and ends in the EMG.

The envelope is the RMS over channels of the EMG z-scored by each channel's rest SD (so a noisy
channel doesn't dominate), smoothed over ``envelope_s``. Its rest baseline b and spread s
(median and 1.4826 x MAD over the rest span) set the thresholds. For each prompt:

- **detected**: the envelope stays above b + ``detect_k`` x s for ``detect_min_s`` or more
  somewhere in the hold (drawn onset to offset, widened by 0.2 s each side);
- **onset**: from the peak, back to where the envelope first drops below
  b + ``onset_fraction`` x (peak - b), then further back to where it drops below
  b + ``edge_k`` x s; the next sample is the onset;
- **offset**: the first sample after the peak below b + ``edge_k`` x s that stays below for
  ``offset_hold_s``.

Search spans: an onset is searched from ``search_before_s`` before the drawn onset (but not
before the previous prompt's drawn offset); an offset up to ``search_after_s`` after the drawn
offset, but not past the next prompt's detected onset (or, if that wasn't detected, its search
start). An edge that runs into its search bound is flagged as censored: the true edge lies
beyond it. Used by the data report, label alignment and event metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trials import Trial


@dataclass(frozen=True)
class DetectorSettings:
    envelope_s: float = 0.025
    detect_k: float = 8.0
    detect_min_s: float = 0.030
    onset_fraction: float = 0.2
    edge_k: float = 3.0
    search_before_s: float = 0.5
    search_after_s: float = 0.8
    offset_hold_s: float = 0.100


@dataclass
class PressTimes:
    onset_ns: np.ndarray  # (n_prompts,) host ns; -1 if not detected
    offset_ns: np.ndarray
    detected: np.ndarray  # bool
    onset_censored: np.ndarray  # bool: the onset ran into the search start
    offset_censored: np.ndarray
    peak: np.ndarray  # envelope peak during the hold, in rest-noise units

    def lags_ms(self, prompts) -> tuple[np.ndarray, np.ndarray]:
        """(onset lag, offset lag) vs the drawn prompt times, ms; NaN if not detected."""
        on = np.where(self.detected, (self.onset_ns - prompts.onset_ns) / 1e6, np.nan)
        off = np.where(self.detected, (self.offset_ns - prompts.offset_ns) / 1e6, np.nan)
        return on, off


def zscore_envelope(emg_uv: np.ndarray, rest_uv: np.ndarray, fs_hz: float, window_s: float):
    """RMS over channels of the EMG scaled by each channel's robust rest SD, smoothed."""
    rest = np.asarray(rest_uv, dtype=np.float64)
    centre = np.median(rest, axis=0)
    sd = 1.4826 * np.median(np.abs(rest - centre), axis=0)
    sd = np.where(sd > 0, sd, np.inf)  # a dead channel contributes nothing
    power = np.mean(((np.asarray(emg_uv, dtype=np.float64) - centre) / sd) ** 2, axis=1)
    k = max(1, int(round(window_s * fs_hz)))
    return np.sqrt(np.convolve(power, np.ones(k) / k, mode="same"))


def estimate_press_times(
    trial: Trial,
    rest_span_ns: tuple[int, int],
    settings: DetectorSettings | None = None,
    fs_hz: float = 2000.0,
) -> PressTimes:
    settings = settings or DetectorSettings()
    p = trial.prompts
    t = trial.emg_t_ns
    r0, r1 = np.searchsorted(t, rest_span_ns)
    if r1 - r0 < int(0.2 * fs_hz):
        raise ValueError(f"{trial.id}: rest span too short to set a baseline")
    env = zscore_envelope(trial.emg_uv, trial.emg_uv[r0:r1], fs_hz, settings.envelope_s)
    rest_env = env[r0:r1]
    base = float(np.median(rest_env))
    spread = float(1.4826 * np.median(np.abs(rest_env - base))) or 1e-9
    detect_thr = base + settings.detect_k * spread
    edge_thr = base + settings.edge_k * spread
    min_run = max(1, int(round(settings.detect_min_s * fs_hz)))
    hold_run = max(1, int(round(settings.offset_hold_s * fs_hz)))

    n = len(p)
    order = np.argsort(p.onset_ns)
    starts = p.onset_ns[order] - round(settings.search_before_s * 1e9)
    starts = np.maximum(starts, np.concatenate([[t[0]], p.offset_ns[order][:-1]]))
    out = PressTimes(
        onset_ns=np.full(n, -1, np.int64),
        offset_ns=np.full(n, -1, np.int64),
        detected=np.zeros(n, bool),
        onset_censored=np.zeros(n, bool),
        offset_censored=np.zeros(n, bool),
        peak=np.zeros(n),
    )
    peaks = np.full(n, -1)
    # Pass 1: detection and onsets.
    for k, i in enumerate(order):
        next_start = starts[k + 1] if k + 1 < n else t[-1]
        a = int(np.searchsorted(t, starts[k]))
        b = int(np.searchsorted(t, min(next_start, p.offset_ns[i] + 200_000_000)))
        h0 = max(int(np.searchsorted(t, p.onset_ns[i] - 200_000_000)), a)
        if b - h0 < min_run:
            continue
        hold = env[h0:b]
        peak_i = h0 + int(np.argmax(hold))
        out.peak[i] = (env[peak_i] - base) / spread
        if _longest_run(hold > detect_thr) < min_run:
            continue
        out.detected[i] = True
        peaks[i] = peak_i
        # Back from the peak below the fractional level, then below the edge level.
        frac_thr = base + settings.onset_fraction * (env[peak_i] - base)
        j = _last_below(env, a, peak_i, frac_thr)
        j = _last_below(env, a, j + 1, edge_thr) if j > a else j
        out.onset_censored[i] = j <= a
        out.onset_ns[i] = t[min(j + 1, peak_i)]
    # Pass 2: offsets, each bounded by the next press's onset.
    for k, i in enumerate(order):
        if not out.detected[i]:
            continue
        limit = p.offset_ns[i] + round(settings.search_after_s * 1e9)
        if k + 1 < n:
            nxt = order[k + 1]
            limit = min(limit, out.onset_ns[nxt] if out.detected[nxt] else starts[k + 1])
        b = max(int(np.searchsorted(t, limit)), peaks[i] + 1)
        below = env[peaks[i] : b] < edge_thr
        m = _first_run(below, hold_run)
        out.offset_censored[i] = m < 0
        out.offset_ns[i] = t[b - 1] if m < 0 else t[peaks[i] + m]
    return out


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def _last_below(env: np.ndarray, lo: int, hi: int, thr: float) -> int:
    """Largest index in [lo, hi) with env < thr, or lo if there is none."""
    idx = np.flatnonzero(env[lo:hi] < thr)
    return lo + int(idx[-1]) if len(idx) else lo


def _first_run(mask: np.ndarray, length: int) -> int:
    """First index starting ``length`` consecutive True values, or -1."""
    if len(mask) == 0:
        return -1
    run = np.convolve(mask.astype(np.int64), np.ones(length, np.int64), mode="valid")
    hits = np.flatnonzero(run == length)
    return int(hits[0]) if len(hits) else -1
