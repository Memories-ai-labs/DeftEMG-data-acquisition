"""Synthetic prompter sessions in the recorded on-disk format, for tests and dry runs.

Each block is a prompt sequence from ``tasks.sequence.generate`` (a fresh seed per block). EMG:

- rest: band-limited noise with a per-channel SD (``noise_uv`` spread ~3x across channels);
- presses: per-digit spatial pattern (a Gaussian bump 3-5 electrodes wide around the band plus
  a weaker one opposite; neighbouring digits overlap) times amplitude-modulated 60-400 Hz noise.
  Amplitude on the peak channel = SNR x that channel's rest SD x the session's ``press_gain``,
  SNR drawn per press from ``snr_low`` / ``snr_high`` / ``snr_max`` (x ``strength``);
- per session (one donning): the rest SDs, the bump widths and ``press_gain`` (skin contact)
  are drawn once, so a calibration block describes the session's other blocks;
- timing: the EMG ramps up linearly (50-100 ms) from ``lead_s`` before the drawn prompt onset
  and is back to rest ``lag_s`` after the offset (uniform ranges, true sample time);
- ``null=True``: the same prompts are saved, but the EMG presses come from another random
  sequence of the same task set, rotated by a random offset over the whole block (lead-in
  included), so the labels carry no information (leakage / chance-level checks);
- ``stall_s``: one gap in each trial (EMG and IMU packets lost, sequence numbers jump).

Sessions are written like ``AcquisitionSession`` does (session.json, trials.jsonl, trial npz)
with port "synthetic" and ``metadata.synthetic``, so ``deftemg_acquisition.data.trials`` loads
them like recordings. Arrays that nothing offline reads (byte offsets, battery, temperature,
USB chunk log) are left out. USB arrival times follow the hardware's ~2-3 ms delay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import signal

from deftemg_acquisition.acquisition.protocol import BAND_SLOTS, EMG_RATE_HZ, N_EMG_CHANNELS
from deftemg_acquisition.tasks.sequence import (
    CALIBRATION_REPS,
    DIGITS,
    GeneratorConfig,
    Sequence,
    calibration_sequence,
    generate,
)

T0_NS = 1_800_000_000_000_000_000  # an arbitrary host time


@dataclass
class SyntheticConfig:
    n_blocks: int = 4
    task_sets: tuple[str, ...] = ("single", "multi")  # cycled over blocks
    per_block: int = 10
    hold_s: tuple[float, float] = (0.5, 2.0)
    gap_s: tuple[float, float] = (0.8, 1.5)
    lead_in_s: float = 4.0
    tail_s: float = 1.5
    noise_uv: tuple[float, float] = (2.0, 7.0)  # per-channel rest SD range
    snr_low: tuple[float, float] = (1.5, 3.0)
    snr_high: tuple[float, float] = (3.0, 8.0)
    snr_max: tuple[float, float] = (9.0, 12.0)  # calibration: as hard as is comfortable
    press_gain: tuple[float, float] = (0.75, 1.33)  # per session: press amplitude factor
    calibration: bool = False  # start each session with a calibration block
    calibration_reps: int = CALIBRATION_REPS  # maximum presses per digit in it
    calibration_task: str = "press"  # press, or pinch / squeeze (the thumb with each finger)
    strength: float = 1.0  # scales both SNR ranges (>1: easier)
    lead_s: tuple[float, float] = (0.05, 0.30)  # EMG onset before the drawn prompt onset
    lag_s: tuple[float, float] = (0.0, 0.40)  # EMG end after the drawn prompt offset
    paint_delay_s: float = 0.016  # drawn time - scheduled time
    stall_s: float = 0.0  # > 0: one gap of this length per trial
    null: bool = False
    compress: bool = True  # savez_compressed, as recordings; False is faster (tests)
    seed: int = 0
    subject: str | None = "SYN"
    donning: int = 1
    hand: str = "right"
    digit_centres: tuple[float, ...] = (12.0, 15.0, 0.5, 3.5, 6.5)  # electrode index per digit
    extra: dict = field(default_factory=dict)  # stored in session metadata


def _band_noise(rng: np.random.Generator, n: int, lo: float, hi: float) -> np.ndarray:
    sos = signal.butter(4, [lo, hi], "bandpass", fs=EMG_RATE_HZ, output="sos")
    x = signal.sosfilt(sos, rng.normal(size=(n + 500, N_EMG_CHANNELS)), axis=0)[500:]
    return x / x.std(axis=0)


def _spatial(centre: float, width: float) -> np.ndarray:
    """Gain per electrode: distances run around the band's 19 positions (ch18 and ch1 are two
    positions apart, across the opening)."""
    ch = np.arange(N_EMG_CHANNELS)
    d = np.abs(ch - centre)
    d = np.minimum(d, BAND_SLOTS - d)
    opposite = np.abs(ch - (centre + BAND_SLOTS / 2) % BAND_SLOTS)
    opposite = np.minimum(opposite, BAND_SLOTS - opposite)
    return np.exp(-0.5 * (d / width) ** 2) + 0.35 * np.exp(-0.5 * (opposite / width) ** 2)


def _envelope(t: np.ndarray, on: float, off: float, ramp: float) -> np.ndarray:
    rise = np.clip((t - on) / ramp, 0, 1)
    fall = np.clip((off - t) / ramp, 0, 1)
    return np.minimum(rise, fall)


def _decoy(cfg: SyntheticConfig, seq: Sequence, length_s: float, rng) -> Sequence:
    """Another sequence of the same task set, rotated over the whole trial (``length_s`` in
    sequence time: lead-in and tail included, so neither the quiet start nor the end carries
    information): presses unrelated to ``seq``'s prompts."""
    other = _sequences(replace(cfg, n_blocks=1, task_sets=(seq.task_set,)), rng)[0]
    start = 0.5
    span = length_s - start
    shift = rng.uniform(0, span)
    prompts = [
        replace(p, onset_s=start + (p.onset_s - start + shift) % span) for p in other.prompts
    ]
    return Sequence(other.name, other.task_set, sorted(prompts, key=lambda p: p.onset_s))


@dataclass
class _Donning:
    """What stays fixed while the band stays on: noise, spatial patterns, contact."""

    noise_sd: np.ndarray  # (E,) rest SD, µV
    patterns: dict[str, np.ndarray]  # digit -> (E,) gain
    press_gain: float

    @classmethod
    def draw(cls, cfg: SyntheticConfig, rng) -> _Donning:
        noise_sd = rng.uniform(*cfg.noise_uv, size=N_EMG_CHANNELS)
        widths = rng.uniform(1.0, 1.7, size=len(DIGITS))  # bumps ~3-5 electrodes wide
        centres = zip(DIGITS, cfg.digit_centres, widths, strict=True)
        patterns = {d: _spatial(c, w) for d, c, w in centres}
        return cls(noise_sd, patterns, float(rng.uniform(*cfg.press_gain)))


def _emg(cfg: SyntheticConfig, seq: Sequence, duration_s: float, scroll_s: float, don, rng):
    n = int(duration_s * EMG_RATE_HZ)
    t = np.arange(n) / EMG_RATE_HZ
    noise_sd, patterns = don.noise_sd, don.patterns
    emg = _band_noise(rng, n, 20.0, 900.0) * noise_sd
    for p in seq.prompts:
        on = scroll_s + p.onset_s + cfg.paint_delay_s - rng.uniform(*cfg.lead_s)
        off = scroll_s + p.offset_s + cfg.paint_delay_s + rng.uniform(*cfg.lag_s)
        a, b = np.searchsorted(t, [on, off])
        if b - a < 2:  # a null session's decoy press can fall after the trial ends
            continue
        env = _envelope(t[a:b], on, off, rng.uniform(0.05, 0.10))
        for digit, level in zip(p.digits, p.levels, strict=True):
            snr_range = {"low": cfg.snr_low, "high": cfg.snr_high, "max": cfg.snr_max}[level]
            snr = rng.uniform(*snr_range) * cfg.strength
            peak = int(np.argmax(patterns[digit]))
            amp = snr * don.press_gain * noise_sd[peak] * patterns[digit]
            burst = _band_noise(rng, b - a, 60.0, 400.0) * amp
            emg[a:b] += env[:, None] * burst
    return emg


def _sequences(cfg: SyntheticConfig, rng) -> list[Sequence]:
    seqs = []
    for b in range(cfg.n_blocks):
        task_set = cfg.task_sets[b % len(cfg.task_sets)]
        gen = GeneratorConfig(
            task_set=task_set,
            reps=8,
            chord_sizes=(2,),
            chord_force="independent",
            hold_s=cfg.hold_s,
            gap_s=cfg.gap_s,
            lead_in_s=cfg.lead_in_s,
            per_block=cfg.per_block,
            seed=int(rng.integers(1 << 31)),
        )
        seq = generate(gen, "syn")[0]
        seq.name = f"syn_{task_set}_b{b + 1:02d}"
        seqs.append(seq)
    return seqs


def write_session(out_dir: Path, cfg: SyntheticConfig | None = None) -> Path:
    """Write one synthetic session folder; returns its path."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "port": "synthetic",
        "opened_at": "synthetic",
        "filter_status": {"enabled": True, "powerline_hz": 60, "highpass_hz": 60},
        "continuous": True,
        "host_clock": "monotonic",
        "clock_anchor": {"wall_ns": T0_NS, "host_ns": T0_NS},
        "metadata": {
            "hand": cfg.hand,
            "subject": cfg.subject,
            "donning": cfg.donning,
            "synthetic": True,
            "synthetic_config": {k: v for k, v in asdict(cfg).items() if k != "extra"},
            **cfg.extra,
        },
    }
    (out_dir / "session.json").write_text(json.dumps(session, indent=2))
    records = []
    host = T0_NS
    don = _Donning.draw(cfg, rng)
    blocks = _sequences(cfg, rng)
    if cfg.calibration:
        blocks.insert(0, calibration_sequence(reps=cfg.calibration_reps,
                                              task=cfg.calibration_task))  # fmt: skip
    for index, seq in enumerate(blocks, start=1):
        scroll_s = 0.05
        duration = scroll_s + seq.duration_s + cfg.tail_s
        calibrating = seq.task_set == "calibration"
        shown = _decoy(cfg, seq, duration - scroll_s, rng) if cfg.null and not calibrating else seq
        emg = _emg(cfg, shown, duration, scroll_s, don, rng)
        arrays = _trial_arrays(cfg, seq, emg, host, scroll_s, rng)
        stem = f"trial_{index:03d}_{seq.name}"
        save = np.savez_compressed if cfg.compress else np.savez
        save(out_dir / f"{stem}.npz", **arrays)
        records.append(
            {
                "index": index,
                "label": seq.name,
                "host_start_ns": int(host),
                "host_end_ns": int(arrays["host_end_ns"]),
                "n_emg": len(arrays["emg_uv"]),
                "n_dropped": int(arrays["n_dropped"]),
                "raw_path": str(out_dir / f"{stem}.bin"),
                "npz_path": str(out_dir / f"{stem}.npz"),
                "metadata": {
                    "task": "scroll_prompter",
                    "task_set": seq.task_set,
                    "sequence_file": f"synthetic/{seq.name}.json",
                    "sequence": seq.to_dict(),
                    "hand": cfg.hand,
                    "lookahead_s": 6.0,
                    "completed": True,
                },
            }
        )
        host = int(arrays["host_end_ns"]) + 5_000_000_000
    with open(out_dir / "trials.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out_dir


def _trial_arrays(cfg, seq: Sequence, emg_uv, host0: int, scroll_s: float, rng) -> dict:
    n = len(emg_uv)
    counts = np.round(emg_uv * 7.5).astype(np.int32)  # the armband's 1/7.5 µV quantisation
    true_ns = host0 + np.round(np.arange(n) * 1e9 / EMG_RATE_HZ).astype(np.int64)
    # One IMU packet after every 10 EMG packets; they share the sequence counter.
    seq_no = np.arange(n, dtype=np.int64) + np.arange(n) // 10
    keep = np.ones(n, bool)
    if cfg.stall_s > 0:  # lose a stretch of packets in the middle of the trial
        a = n // 2
        keep[a : a + int(cfg.stall_s * EMG_RATE_HZ)] = False
    # Host arrival: chunks of 4 samples plus 1-1.5 ms, so fitted times run ~2.5 ms late, as on
    # the hardware.
    chunk_end = np.minimum((np.arange(n) // 4 + 1) * 4 - 1, n - 1)
    jitter = rng.integers(0, 500_000, size=n // 4 + 1)[np.arange(n) // 4]
    arrive = true_ns[chunk_end] + 1_000_000 + jitter
    all_imu = np.arange(9, n, 10)
    imu_idx = all_imu[keep[all_imu]]
    arrays = seq.prompt_arrays()
    scroll_ns = host0 + round(scroll_s * 1e9)
    sched_on = scroll_ns + np.round(arrays["prompt_onset_s"] * 1e9).astype(np.int64)
    sched_off = scroll_ns + np.round(arrays["prompt_offset_s"] * 1e9).astype(np.int64)
    delay = round(cfg.paint_delay_s * 1e9)
    return {
        "emg_uv": counts[keep] / 7.5,
        "emg_counts": counts[keep],
        "emg_seq": seq_no[keep],
        "emg_host_time_ns": arrive[keep],
        "imu_seq": seq_no[imu_idx] + 1,  # the number after its 10th EMG packet
        "imu_host_time_ns": arrive[imu_idx],
        "gyro_rad_s": np.zeros((len(imu_idx), 3)),
        "acc_m_s2": np.tile([0.0, 0.0, 9.81], (len(imu_idx), 1)),
        "device_time_ms": ((true_ns[imu_idx] - host0) // 1_000_000 + host0 // 1_000_000).astype(
            np.uint64
        ),
        "device_timestamp_ms": np.int64(host0 // 1_000_000),
        "n_dropped": np.int64(np.sum(~keep) + len(all_imu) - len(imu_idx)),
        "host_start_ns": np.int64(host0),
        "host_end_ns": np.int64(true_ns[-1] + 10_000_000),
        "host_clock": np.array("monotonic"),
        "anchor_wall_ns": np.int64(host0),
        "anchor_host_ns": np.int64(host0),
        **arrays,
        "scroll_start_host_ns": np.int64(scroll_ns),
        "prompt_onset_host_ns": sched_on,
        "prompt_offset_host_ns": sched_off,
        "prompt_onset_paint_ns": sched_on + delay,
        "prompt_offset_paint_ns": sched_off + delay,
    }
