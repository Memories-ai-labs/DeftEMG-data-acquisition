"""Timing checks for recorded trials: sample clock, USB delay, and prompt-to-EMG alignment.

All times are on the host clock (see ``deftemg_acquisition.clock``: monotonic since recordings
gained a ``host_clock`` field, wall clock before), which is also what prompt onsets are logged
in.

- EMG packets are timestamped when their USB chunk arrives, so neighbouring samples share a
  timestamp. :func:`fit_sample_times` fits host time against sample index to give every sample
  a smooth time; the fit sits at the *average* arrival delay.
- IMU packets also carry the armband's own clock, which is set from the host clock at trial
  start, so ``host receive time - device time`` estimates the USB delay (plus ~1 ms for the
  clock-setting command itself) and the drift between the two clocks.
- :func:`emg_rate_on_device_clock` counts EMG samples against the armband's own clock: AA and
  BB packets share a sequence counter, so each IMU packet shows how many EMG packets preceded
  it. Together with the two rates above this closes the loop between the three clocks
  (EMG sampling, armband ms timer, host): host ppm ~= device ppm - drift ppm.
- :func:`detect_onsets` finds EMG onsets near each prompt, giving prompt-to-EMG lag. Lag is
  measured from when each prompt was drawn reaching the target (``prompt_onset_paint_ns``,
  falling back to the schedule for older recordings). It still mixes the display's own delay
  (measure once with the sync patch) with the subject's timing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from deftemg_acquisition.acquisition.protocol import EMG_RATE_HZ, STALL_GAP_S


def emg_segments(host_ns: np.ndarray, emg_seq: np.ndarray | None = None) -> np.ndarray:
    """Segment id per EMG sample; a new segment starts after a stall or lost packets.

    Within a segment, sample index advances in step with time. EMG sequence numbers step by 1,
    or 2 where an IMU packet sits between; larger steps mean packets were lost.
    """
    breaks = np.diff(host_ns) > STALL_GAP_S * 1e9
    if emg_seq is not None:
        breaks |= np.diff(emg_seq) > 2
    return np.concatenate([[0], np.cumsum(breaks)])


def segmented_fit(
    x: np.ndarray, y: np.ndarray, seg: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Straight lines with one shared slope and a separate intercept per segment.

    Returns (slope, intercept per segment, residuals). Clocks run at one rate, but stalls and
    lost packets shift the offset between sample index and time.
    """
    counts = np.bincount(seg).astype(np.float64)
    xm = np.bincount(seg, x) / counts
    ym = np.bincount(seg, y) / counts
    xc, yc = x - xm[seg], y - ym[seg]
    slope = float(np.sum(xc * yc) / np.sum(xc * xc))
    intercept = ym - slope * xm
    return slope, intercept, y - (slope * x + intercept[seg])


@dataclass
class SampleClock:
    rate_hz: float  # effective EMG rate measured on the host clock
    rate_error_ppm: float  # relative to the nominal 2000 Hz
    arrival_jitter_ms_std: float  # spread of arrival times around the fit
    arrival_jitter_ms_p99: float  # |residual| 99th percentile
    n_segments: int  # continuous stretches of data
    n_stalls: int  # times no data arrived for more than STALL_GAP_S
    stalled_s: float  # total time without data


def fit_sample_times(
    host_ns: np.ndarray, emg_seq: np.ndarray | None = None
) -> tuple[np.ndarray, SampleClock]:
    """Fit host arrival time vs EMG sample index, separately per continuous segment.

    Returns fitted times (int64 ns, one per sample) and fit statistics. The rate is shared
    across segments; each segment gets its own offset, so stalls and lost packets don't bend it.
    """
    seg = emg_segments(host_ns, emg_seq)
    n = np.arange(len(host_ns), dtype=np.float64)
    y = (host_ns - host_ns[0]).astype(np.float64)
    slope, intercept, resid = segmented_fit(n, y, seg)
    fitted = host_ns[0] + np.round(slope * n + intercept[seg]).astype(np.int64)
    gaps_s = np.diff(host_ns) / 1e9
    stalls = gaps_s[gaps_s > STALL_GAP_S]
    rate = 1e9 / slope
    return fitted, SampleClock(
        rate_hz=rate,
        rate_error_ppm=(rate / EMG_RATE_HZ - 1) * 1e6,
        arrival_jitter_ms_std=float(resid.std() / 1e6),
        arrival_jitter_ms_p99=float(np.percentile(np.abs(resid), 99) / 1e6),
        n_segments=int(seg[-1] + 1),
        n_stalls=len(stalls),
        stalled_s=float(stalls.sum()),
    )


@dataclass
class UsbDelay:
    # Delay with the clock drift trend removed, referred to the start of the trial.
    median_ms: float
    p5_ms: float
    p95_ms: float
    drift_ppm: float  # host clock relative to the armband clock over the trial


def usb_delay(
    imu_host_ns: np.ndarray,
    device_time_ms: np.ndarray,
    device_ts_ms: int,
    wall_minus_host_ns: int = 0,
) -> UsbDelay:
    """Host receive time minus device time for IMU packets, and the drift between the clocks.

    The device clock is set from the wall clock at trial start, so host times are converted with
    ``wall_minus_host_ns`` (from the trial's clock anchor; 0 for older wall-clock recordings).
    Drift is fitted first and removed, so the delay statistics describe USB transport only and
    stay meaningful for long trials. Packets whose device time predates the trial's clock
    setting are skipped: in continuous mode the first few can carry the clock from before TIME.
    """
    dev = device_time_ms.astype(np.int64)
    keep = dev >= device_ts_ms
    if not keep.any():
        return UsbDelay(np.nan, np.nan, np.nan, np.nan)
    delay_ms = (imu_host_ns[keep] + wall_minus_host_ns) / 1e6 - dev[keep]
    elapsed_s = (dev[keep] - dev[keep][0]) / 1e3
    slope = np.polyfit(elapsed_s, delay_ms, 1)[0] if np.ptp(elapsed_s) > 1 else 0.0
    detrended = delay_ms - slope * elapsed_s
    return UsbDelay(
        median_ms=float(np.median(detrended)),
        p5_ms=float(np.percentile(detrended, 5)),
        p95_ms=float(np.percentile(detrended, 95)),
        drift_ppm=float(slope * 1e3) if np.ptp(elapsed_s) > 1 else np.nan,
    )


@dataclass
class DeviceClockRate:
    rate_hz: float  # EMG samples per second of the armband's own ms clock
    rate_error_ppm: float  # relative to the nominal 2000 Hz
    resid_sd_samples: float  # scatter around the fit (the ms clock alone gives ~0.3-0.6)


def emg_rate_on_device_clock(
    emg_seq: np.ndarray,
    imu_seq: np.ndarray,
    device_time_ms: np.ndarray,
    device_ts_ms: int,
    emg_seg: np.ndarray | None = None,
) -> DeviceClockRate:
    """Fit EMG packets sent so far against the device time of each IMU packet.

    ``emg_seg`` (from :func:`emg_segments`) splits the fit at stalls and lost packets.
    """
    dev = device_time_ms.astype(np.int64)
    keep = dev >= device_ts_ms  # skip packets carrying the clock from before TIME took effect
    if keep.sum() < 3 or np.ptp(dev[keep]) < 1000:
        return DeviceClockRate(np.nan, np.nan, np.nan)
    dev = dev[keep]
    t_s = (dev - dev[0]) / 1e3
    n_before = np.searchsorted(emg_seq, imu_seq[keep])
    if emg_seg is None:
        emg_seg = np.zeros(len(emg_seq), np.int64)
    from_emg = emg_seg[np.clip(n_before - 1, 0, len(emg_seg) - 1)]
    breaks = (np.diff(from_emg) != 0) | (np.diff(dev) > STALL_GAP_S * 1e3)
    seg = np.concatenate([[0], np.cumsum(breaks)])
    rate, _, resid = segmented_fit(t_s, n_before.astype(np.float64), seg)
    return DeviceClockRate(rate, (rate / EMG_RATE_HZ - 1) * 1e6, float(resid.std()))


def emg_envelope(emg_uv: np.ndarray, window_s: float = 0.025) -> np.ndarray:
    """RMS across channels, smoothed with a centred moving window (no time shift)."""
    power = np.mean(emg_uv.astype(np.float64) ** 2, axis=1)
    k = max(1, int(round(window_s * EMG_RATE_HZ)))
    return np.sqrt(np.convolve(power, np.ones(k) / k, mode="same"))


def detect_onsets(
    t_ns: np.ndarray,
    envelope: np.ndarray,
    onset_ns: np.ndarray,
    offset_ns: np.ndarray,
    baseline_s: tuple[float, float] = (-0.6, -0.15),
    search_from_s: float = -0.15,
    fraction: float = 0.2,
    min_snr: float = 8.0,
) -> np.ndarray:
    """EMG onset time (ns) for each prompt, or -1 where no clear onset was found.

    Onset = first time from ``search_from_s`` before the prompt that the envelope rises above
    baseline + ``fraction`` x (peak - baseline), where the baseline is the median envelope in
    ``baseline_s`` around the prompt and the peak is the maximum during the prompt. A prompt is
    skipped unless its peak exceeds baseline by ``min_snr`` robust standard deviations.
    """
    out = np.full(len(onset_ns), -1, dtype=np.int64)
    for i, (on, off) in enumerate(zip(onset_ns, offset_ns, strict=True)):
        base = envelope[(t_ns >= on + baseline_s[0] * 1e9) & (t_ns < on + baseline_s[1] * 1e9)]
        during = (t_ns >= on + search_from_s * 1e9) & (t_ns < off)
        if base.size < 10 or not during.any():
            continue
        b = np.median(base)
        noise = 1.4826 * np.median(np.abs(base - b)) + 1e-9
        idx = np.flatnonzero(during)
        peak = envelope[idx].max()
        if peak - b < min_snr * noise:
            continue
        above = idx[envelope[idx] > b + fraction * (peak - b)]
        out[i] = t_ns[above[0]]
    return out


@dataclass
class TrialTiming:
    label: str
    n_emg: int
    n_dropped: int
    clock: SampleClock
    usb: UsbDelay
    device_clock: DeviceClockRate
    n_prompts: int = 0
    n_onsets: int = 0
    lag_ms_median: float = np.nan
    lag_ms_q25: float = np.nan
    lag_ms_q75: float = np.nan
    # Scheduled prompt onset -> first frame drawing it at the target.
    paint_delay_ms_median: float = np.nan
    paint_delay_ms_max: float = np.nan

    def as_dict(self) -> dict:
        return asdict(self)


def analyse_trial(data, label: str) -> tuple[TrialTiming, dict[str, np.ndarray]]:
    """Timing summary for one trial's ``.npz`` contents, plus arrays to save alongside it."""
    emg_host = data["emg_host_time_ns"]
    emg_time, clock = fit_sample_times(emg_host, data["emg_seq"])
    # Host clock -> wall clock (which the device clock was set from); 0 for old recordings.
    wall_minus_host = (
        int(data["anchor_wall_ns"]) - int(data["anchor_host_ns"]) if "anchor_wall_ns" in data else 0
    )
    usb = usb_delay(
        data["imu_host_time_ns"],
        data["device_time_ms"],
        int(data["device_timestamp_ms"]),
        wall_minus_host,
    )
    device_clock = emg_rate_on_device_clock(
        data["emg_seq"],
        data["imu_seq"],
        data["device_time_ms"],
        int(data["device_timestamp_ms"]),
        emg_segments(emg_host, data["emg_seq"]),
    )
    timing = TrialTiming(label, len(emg_host), int(data["n_dropped"]), clock, usb, device_clock)
    arrays = {"emg_time_ns": emg_time, "usb_delay_ms_median": np.float64(usb.median_ms)}

    if "prompt_onset_host_ns" in data:
        sched, off = data["prompt_onset_host_ns"], data["prompt_offset_host_ns"]
        drawn = data["prompt_onset_paint_ns"] if "prompt_onset_paint_ns" in data else None
        # Reference time per prompt: when it was drawn at the target, else when scheduled.
        ref = np.where(drawn >= 0, drawn, sched) if drawn is not None else sched
        onsets = detect_onsets(emg_time, emg_envelope(data["emg_uv"]), sched, off)
        found = onsets >= 0
        lags = (onsets[found] - ref[found]) / 1e6
        timing.n_prompts, timing.n_onsets = len(sched), int(found.sum())
        if found.any():
            timing.lag_ms_median = float(np.median(lags))
            timing.lag_ms_q25, timing.lag_ms_q75 = map(float, np.percentile(lags, [25, 75]))
        arrays["emg_onset_ns"] = onsets
        arrays["onset_lag_ms"] = np.where(found, (onsets - ref) / 1e6, np.nan)
        arrays["prompt_ref_ns"] = ref

        if drawn is not None and (drawn >= 0).any():
            delay = (drawn[drawn >= 0] - sched[drawn >= 0]) / 1e6
        elif "sync_flash_paint_ns" in data and len(data["sync_flash_paint_ns"]):
            # Older recordings: each flash belongs to the latest onset at or before it.
            paints = data["sync_flash_paint_ns"]
            k = np.searchsorted(sched, paints, side="right") - 1
            delay = (paints[k >= 0] - sched[k[k >= 0]]) / 1e6
        else:
            delay = np.empty(0)
        if delay.size:
            timing.paint_delay_ms_median = float(np.median(delay))
            timing.paint_delay_ms_max = float(delay.max())
    return timing, arrays
