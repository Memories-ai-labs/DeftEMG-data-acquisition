"""Standard offline EMG filtering: 20 Hz high-pass, 60 Hz notch, optional mains-harmonic notches.

    cfg = FilterConfig()                               # 20 Hz high-pass + 60 Hz notch
    cfg = FilterConfig(harmonics="odd")                # + 180, 300, 420, ... Hz
    cfg = FilterConfig(harmonics=[3, 5, 9])            # + 180, 300, 540 Hz only
    y = filter_emg(trial.emg_uv, 2000.0, cfg, trial.segment_bounds())
    trial = filter_trial(load_trial(npz_path), cfg)    # same, on a loaded trial

Filtering is zero-phase (forward-backward) by default, for offline analysis: no delay, but the
magnitude response is applied twice (the 4th-order Butterworth high-pass is 6 dB down at its
corner, not 3 dB). ``zero_phase=False`` runs the same filter causally, as a live or inference
pipeline would, with its state started at the first sample's level so a DC offset does not
ring through.

The raw 24-bit stream carries a DC offset of several mV; the high-pass removes it, so there is
no need to subtract the mean first. Each continuous segment is filtered on its own, so a gap
(lost packets, a stall) never becomes a step for the filter to ring on.

Trials recorded with the armband's built-in filter on are already high-passed (20/40/60 Hz) and
comb-notched at every 60 Hz multiple; filtering them again changes little.

Each notch has a fixed bandwidth (``notch_width_hz``) at every harmonic, so harmonic lines,
which are as narrow as the fundamental, are not removed with a wider slice of EMG.

``EmgFilter`` names a filter as configs and command lines give it: a high-pass in Hz and a
mains notch from ``NOTCHES`` (e.g. ``comb60all``: 60 Hz and every harmonic below fs/2). The
live stream (``--live-highpass`` / ``--live-notch``, causal), ``filter-emg --harmonics all``
(zero-phase) and the calibration measurements (``data.calibration``) then share one design.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from deftemg_acquisition.errors import ConfigError


@dataclass
class FilterConfig:
    highpass_hz: float | None = 20.0  # None: no high-pass
    highpass_order: int = 4
    notch_hz: float | None = 60.0  # mains fundamental; None: no notches at all
    notch_width_hz: float = 2.0  # -3 dB bandwidth of each notch (Q = f / width)
    # Extra notches at multiples of notch_hz: "none", "odd" (3, 5, 7, ...), "all" (2, 3, ...),
    # or explicit multiples such as [3, 5, 9]. Only those below fs/2 - notch_width_hz are used.
    harmonics: Literal["none", "odd", "all"] | list[int] = "none"
    zero_phase: bool = True

    def __post_init__(self) -> None:
        if self.highpass_hz is not None and self.highpass_hz <= 0:
            raise ConfigError(f"highpass_hz: {self.highpass_hz} must be > 0 (or None)")
        if self.highpass_order < 1:
            raise ConfigError(f"highpass_order: {self.highpass_order} must be >= 1")
        if self.notch_hz is not None and self.notch_hz <= 0:
            raise ConfigError(f"notch_hz: {self.notch_hz} must be > 0 (or None)")
        if self.notch_width_hz <= 0:
            raise ConfigError(f"notch_width_hz: {self.notch_width_hz} must be > 0")
        if isinstance(self.harmonics, str):
            if self.harmonics not in ("none", "odd", "all"):
                raise ConfigError(f"harmonics: {self.harmonics!r} is not none, odd, all or a list")
        elif any(int(k) != k or k < 2 for k in self.harmonics):
            raise ConfigError(f"harmonics: {self.harmonics} must be integer multiples >= 2")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# Mains notches by name: (mains Hz, multiples of it notched as well), or None for no notch.
# The combs notch every odd harmonic below fs/2 (60 Hz: 180, 300, ..., 900 Hz). Raw recordings
# showed lines at 180, 300, 420, 540 and 660 Hz (stronger with a monitor attached or a loose or
# freshly placed band); even harmonics were 5-20x weaker, up to ~6 µV at 120 Hz. The "all"
# combs notch every harmonic (60 Hz: 120, 180, ..., 960 Hz), at the cost of more EMG band.
COMB_HARMONICS = "odd"
NOTCHES: dict[str, tuple[float, str | list[int]] | None] = {
    "none": None,
    "50": (50.0, "none"),
    "60": (60.0, "none"),
    "comb50": (50.0, COMB_HARMONICS),
    "comb60": (60.0, COMB_HARMONICS),
    "comb50all": (50.0, "all"),
    "comb60all": (60.0, "all"),
}


def named_filter_config(
    highpass_hz: float | None, notch: str, zero_phase: bool = True
) -> FilterConfig:
    """A ``FilterConfig`` from a high-pass (Hz, or None) and a notch name from ``NOTCHES``."""
    if notch not in NOTCHES:
        raise ConfigError(f"notch must be one of {sorted(NOTCHES)}, got {notch!r}")
    notch_hz, harmonics = NOTCHES[notch] or (None, "none")
    return FilterConfig(
        highpass_hz=highpass_hz, notch_hz=notch_hz, harmonics=harmonics, zero_phase=zero_phase
    )


@dataclass(frozen=True)
class EmgFilter:
    """A named filter (hashable, so it can key caches): see ``named_filter_config``."""

    highpass_hz: float | None = 20.0
    notch: str = "comb60all"
    zero_phase: bool = True

    def __post_init__(self) -> None:
        self.config()  # validate

    def config(self) -> FilterConfig:
        return named_filter_config(self.highpass_hz, self.notch, self.zero_phase)

    def describe(self) -> str:
        """E.g. "high-pass 20 Hz + notch comb60all, zero-phase"."""
        parts = ([f"high-pass {self.highpass_hz:g} Hz"] if self.highpass_hz else []) + (
            [f"notch {self.notch}"] if self.notch != "none" else []
        )
        return " + ".join(parts) + (", zero-phase" if self.zero_phase else ", causal")

    def apply(
        self, x_uv: np.ndarray, fs_hz: float, segments: list[tuple[int, int]] | None = None
    ) -> np.ndarray:
        """``filter_emg`` with this filter."""
        return filter_emg(x_uv, fs_hz, self.config(), segments)

    def trial(self, trial, fs_hz: float = 2000.0):
        """``filter_trial`` with this filter."""
        return filter_trial(trial, self.config(), fs_hz)


def notch_frequencies(cfg: FilterConfig, fs_hz: float) -> list[float]:
    """The frequencies (Hz) notched under ``cfg`` at sample rate ``fs_hz``, ascending."""
    if cfg.notch_hz is None:
        return []
    top = fs_hz / 2 - cfg.notch_width_hz
    if cfg.notch_hz >= top:
        raise ConfigError(f"notch_hz: {cfg.notch_hz} must be below fs/2 - notch_width_hz ({top})")
    n_max = int(top // cfg.notch_hz)
    if cfg.harmonics == "none":
        multiples = []
    elif cfg.harmonics == "odd":
        multiples = range(3, n_max + 1, 2)
    elif cfg.harmonics == "all":
        multiples = range(2, n_max + 1)
    else:
        multiples = sorted({int(k) for k in cfg.harmonics if k * cfg.notch_hz < top})
    return [cfg.notch_hz] + [k * cfg.notch_hz for k in multiples]


def design_filter(cfg: FilterConfig, fs_hz: float) -> np.ndarray | None:
    """The whole chain as second-order sections (high-pass first), or None if it is empty."""
    sections = []
    if cfg.highpass_hz is not None:
        if cfg.highpass_hz >= fs_hz / 2:
            raise ConfigError(f"highpass_hz: {cfg.highpass_hz} must be below fs/2 ({fs_hz / 2})")
        sections.append(
            signal.butter(cfg.highpass_order, cfg.highpass_hz, "highpass", fs=fs_hz, output="sos")
        )
    for f in notch_frequencies(cfg, fs_hz):
        b, a = signal.iirnotch(f, f / cfg.notch_width_hz, fs=fs_hz)
        sections.append(signal.tf2sos(b, a))
    return np.vstack(sections) if sections else None


def filter_emg(
    x_uv: np.ndarray,
    fs_hz: float,
    cfg: FilterConfig | None = None,
    segments: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """(n, E) EMG in µV -> filtered (n, E) float64, each continuous segment on its own.

    ``segments`` are (start, end) sample ranges covering the recording (``Trial.segment_bounds``);
    None treats the whole array as one segment. Segments too short to filter are left as they
    are, minus their mean.
    """
    cfg = cfg or FilterConfig()
    x = np.asarray(x_uv, dtype=np.float64)
    sos = design_filter(cfg, fs_hz)
    if sos is None:
        return x.copy()
    out = np.empty_like(x)
    for a, b in segments if segments is not None else [(0, len(x))]:
        seg = x[a:b]
        n = len(seg)
        if n < 2:
            out[a:b] = seg - seg.mean(axis=0) if n else seg
        elif cfg.zero_phase:
            padlen = min(3 * (2 * len(sos) + 1), n - 1)
            out[a:b] = signal.sosfiltfilt(sos, seg, axis=0, padlen=padlen)
        else:
            zi = signal.sosfilt_zi(sos)[:, :, None] * seg[0]
            out[a:b], _ = signal.sosfilt(sos, seg, axis=0, zi=zi)
    return out


def filter_trial(trial, cfg: FilterConfig | None = None, fs_hz: float = 2000.0):
    """A copy of a loaded ``Trial`` with its EMG filtered segment by segment."""
    return dataclasses.replace(
        trial, emg_uv=filter_emg(trial.emg_uv, fs_hz, cfg, trial.segment_bounds())
    )
