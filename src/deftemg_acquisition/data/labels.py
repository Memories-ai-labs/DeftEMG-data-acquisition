"""Per-sample digit force targets from prompts, and the rest span of a trial.

A digit is labelled at the prompted level over

    [onset + onset_shift + delay, offset + offset_shift + delay)

of each prompt that includes it, and "none" elsewhere. The shifts move the prompt edges to where
the EMG starts and ends (in the sync trials the EMG starts ~150-200 ms before the drawn onset and
ends ~150-280 ms after the offset). The delay follows Kaifosh et al. (Nature 2025): labels sit
100 ms after the EMG, so a causal model has seen some of the press before it is asked to report
it.

``margin_s`` (training only) marks targets within that distance of one of the digit's own label
edges as ignored (-100): the exact edge is uncertain, and evaluation scores every sample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .press_times import DetectorSettings, estimate_press_times
from .trials import Prompts, Trial

log = logging.getLogger(__name__)

IGNORE = -100


@dataclass(frozen=True)
class LabelSpec:
    onset_shift_s: float = -0.15
    offset_shift_s: float = 0.20
    delay_s: float = 0.10
    margin_s: float = 0.10


def label_intervals(prompts: Prompts, spec: LabelSpec) -> tuple[np.ndarray, np.ndarray]:
    """Label start / end (host ns) per prompt."""
    start = prompts.onset_ns + round((spec.onset_shift_s + spec.delay_s) * 1e9)
    end = prompts.offset_ns + round((spec.offset_shift_s + spec.delay_s) * 1e9)
    return start, end


def digit_targets(
    t_ns: np.ndarray,
    prompts: Prompts,
    spec: LabelSpec,
    classes: tuple[str, ...] = ("none", "low", "high"),
    digits: tuple[str, ...] | None = None,
    margin: bool = False,
) -> np.ndarray:
    """Targets (n, n_digits) int64 at sample times ``t_ns``: class index, or -100 if ignored.

    ``classes[0]`` is "none"; the others are matched to the trial's level names. ``margin``
    applies ``spec.margin_s`` (training targets).
    """
    digits = digits or prompts.digit_names
    level_class = {}
    for i, name in enumerate(prompts.level_names):
        if name not in classes:
            raise ValueError(f"force level {name!r} is not one of the classes {classes}")
        level_class[i] = classes.index(name)
    start, end = label_intervals(prompts, spec)
    out = np.zeros((len(t_ns), len(digits)), np.int64)
    edges: list[list[int]] = [[] for _ in digits]
    for p in np.argsort(start, kind="stable"):  # later prompts win where intervals touch
        a, b = np.searchsorted(t_ns, [start[p], end[p]])
        for d, name in enumerate(digits):
            col = prompts.digit_names.index(name)
            if prompts.digits[p, col]:
                out[a:b, d] = level_class[int(prompts.level[p, col])]
                edges[d] += [start[p], end[p]]
    if margin and spec.margin_s > 0:
        m = round(spec.margin_s * 1e9)
        for d, e in enumerate(edges):
            for edge in e:
                a, b = np.searchsorted(t_ns, [edge - m, edge + m])
                out[a:b, d] = IGNORE
    return out


def label_span_ns(
    trial: Trial, spec: LabelSpec, skip_start_s: float, tail_s: float
) -> tuple[int, int]:
    """Host-time range in which window label times are used: from ``skip_start_s`` after the
    first sample (the start of a block carries a key-press artefact) to ``tail_s`` after the
    last label offset."""
    t0 = int(trial.emg_t_ns[0])
    _, end = label_intervals(trial.prompts, spec)
    return t0 + round(skip_start_s * 1e9), int(end.max()) + round(tail_s * 1e9)


def rest_span_ns(
    trial: Trial, spec: LabelSpec, skip_start_s: float, guard_s: float
) -> tuple[int, int]:
    """The rest before the first press: [start + skip_start_s, first label onset - guard_s].

    Used for the noise scale, and later the press detector's baseline and false activations.
    May be empty (start >= end) for a block with no lead-in.
    """
    t0 = int(trial.emg_t_ns[0])
    start, _ = label_intervals(trial.prompts, spec)
    return t0 + round(skip_start_s * 1e9), int(start.min()) - round(guard_s * 1e9)


def rest_samples(trial: Trial, span: tuple[int, int]) -> np.ndarray:
    """EMG samples (µV) inside a host-time span."""
    a, b = np.searchsorted(trial.emg_t_ns, span)
    return trial.emg_uv[a:b]


def prompt_counts(
    prompts: Prompts, classes: tuple[str, ...], digits: tuple[str, ...]
) -> np.ndarray:
    """(n_digits, n_classes) number of prompts per digit and class (class 0 = not prompted)."""
    out = np.zeros((len(digits), len(classes)), np.int64)
    for d, name in enumerate(digits):
        col = prompts.digit_names.index(name)
        for p in range(len(prompts)):
            if prompts.digits[p, col]:
                out[d, classes.index(prompts.level_names[prompts.level[p, col]])] += 1
            else:
                out[d, 0] += 1
    return out


@dataclass(frozen=True)
class Alignment:
    """How prompt times become label times (beyond the fixed shifts), and null controls."""

    mode: str = "prompt"  # prompt | emg_trial_median | emg
    mask_undetected: bool = False  # ignore digits of prompts with no detectable press
    null: str = "none"  # none | permute | shift
    seed: int = 0
    skip_start_s: float = 1.0  # rest span for the press detector
    rest_guard_s: float = 0.3
    fs_hz: float = 2000.0
    min_detected: int = 3  # for a trial median
    detector: DetectorSettings | None = None


def aligned_prompts(
    trial: Trial, spec: LabelSpec, how: Alignment
) -> tuple[Prompts, LabelSpec, np.ndarray]:
    """Prompts and label spec to build targets from, and a per-prompt "ignore" flag.

    - ``prompt``: the drawn prompt times with the fixed shifts.
    - ``emg_trial_median``: each prompt gets the drawn times plus this trial's median measured
      onset / offset lag *for its force level* (see ``_measured_edges``).
    - ``emg``: each prompt's own measured onset / offset where detected and plausible, else
      the drawn time plus that median lag. A censored edge (the true edge lies beyond the
      search bound) is never placed short of its bound. The fixed shifts are replaced (0);
      the delay stays.
    - ``mask_undetected``: flag prompts with no detected press (the caller ignores their
      digits in training targets).
    - ``null``: ``permute`` shuffles (digits, levels) across prompts, keeping the times, so
      press *timing* stays learnable but digit and level don't; ``shift`` moves all labels
      circularly by a uniform random offset, so nothing is learnable (controls; average over
      seeds).

    Without enough rest before the first press to set the detector's baseline, a trial falls
    back to ``prompt`` alignment (with a warning).
    """
    import dataclasses

    p = trial.prompts
    ignore = np.zeros(len(p), bool)
    if how.mode != "prompt" or how.mask_undetected:
        rest = rest_span_ns(trial, spec, how.skip_start_s, how.rest_guard_s)
        try:
            times = estimate_press_times(trial, rest, how.detector, fs_hz=how.fs_hz)
        except ValueError as e:
            log.warning("%s; using prompt-aligned labels for this trial", e)
            how = dataclasses.replace(how, mode="prompt", mask_undetected=False)
            return aligned_prompts(trial, spec, how)
        if how.mode not in ("prompt", "emg_trial_median", "emg"):
            raise ValueError(f"alignment {how.mode!r} is not one of prompt, emg_trial_median, emg")
        if how.mode != "prompt":
            on, off = _measured_edges(p, times, spec, how)
            p = dataclasses.replace(p, onset_ns=on, offset_ns=off)
            spec = dataclasses.replace(spec, onset_shift_s=0.0, offset_shift_s=0.0)
        if how.mask_undetected:
            ignore = ~times.detected
    if how.null != "none":
        p = _null_prompts(trial, p, how)
    return p, spec, ignore


def _measured_edges(p: Prompts, times, spec: LabelSpec, how: Alignment):
    """Label onset / offset times (host ns) per prompt from measured press times.

    Plausible presses (detected, ending after the drawn onset + 100 ms and covering at least
    half of the hold) set per-level median lags. Censored edges count at their search bound,
    which for an offset is a lower bound on the lag, so long presses aren't left out. With
    fewer than ``min_detected`` presses of a level, all levels are pooled; with fewer than
    that overall, the fixed shifts are used.
    """
    hold = p.offset_ns - p.onset_ns
    covered = np.minimum(times.offset_ns, p.offset_ns) - np.maximum(times.onset_ns, p.onset_ns)
    plausible = times.detected & (times.offset_ns > p.onset_ns + 100_000_000)
    plausible &= covered >= hold / 2
    on_lag = (times.onset_ns - p.onset_ns).astype(float)
    off_lag = (times.offset_ns - p.offset_ns).astype(float)
    level = p.level.max(axis=1)  # a prompt's (highest) level

    def median(lag, sel, fallback):
        if sel.sum() >= how.min_detected:
            return float(np.median(lag[sel]))
        if plausible.sum() >= how.min_detected:
            return float(np.median(lag[plausible]))
        return fallback * 1e9

    med_on = np.array([median(on_lag, plausible & (level == lv), spec.onset_shift_s)
                       for lv in level])  # fmt: skip
    med_off = np.array([median(off_lag, plausible & (level == lv), spec.offset_shift_s)
                        for lv in level])  # fmt: skip
    on = p.onset_ns + np.round(med_on).astype(np.int64)
    off = p.offset_ns + np.round(med_off).astype(np.int64)
    if how.mode == "emg":
        own_on = plausible & ~times.onset_censored
        own_off = plausible & ~times.offset_censored
        on = np.where(own_on, times.onset_ns, on)
        off = np.where(own_off, times.offset_ns, off)
        # Censored: the true edge is beyond the bound (earlier onset, later offset).
        cen_on = plausible & times.onset_censored
        cen_off = plausible & times.offset_censored
        on = np.where(cen_on, np.minimum(times.onset_ns, on), on)
        off = np.where(cen_off, np.maximum(times.offset_ns, off), off)
    return on.astype(np.int64), np.maximum(off, on + 1).astype(np.int64)


def _null_prompts(trial: Trial, p: Prompts, how: Alignment) -> Prompts:
    import dataclasses
    import zlib

    rng = np.random.default_rng([how.seed, zlib.crc32(trial.id.encode())])
    if how.null == "permute":
        order = rng.permutation(len(p))
        return dataclasses.replace(p, digits=p.digits[order], level=p.level[order])
    if how.null != "shift":
        raise ValueError(f"null {how.null!r} is not one of none, permute, shift")
    t0, t1 = int(trial.emg_t_ns[0]), int(trial.emg_t_ns[-1])
    length = t1 - t0
    # Uniform over the whole trial: excluding shifts near the prompt period would bias the
    # shifted labels towards the gaps between real presses.
    shift = rng.uniform(0, length)
    on = t0 + (p.onset_ns - t0 + round(shift)) % length
    return dataclasses.replace(p, onset_ns=on, offset_ns=on + (p.offset_ns - p.onset_ns))


def ignore_prompts(
    targets: np.ndarray, t_ns: np.ndarray, prompts: Prompts, spec: LabelSpec, ignore: np.ndarray,
    digits: tuple[str, ...],
) -> np.ndarray:  # fmt: skip
    """Set targets of the flagged prompts' digits, over their label intervals, to -100."""
    if not ignore.any():
        return targets
    out = targets.copy()
    start, end = label_intervals(prompts, spec)
    for i in np.flatnonzero(ignore):
        a, b = np.searchsorted(t_ns, [start[i], end[i]])
        for d, name in enumerate(digits):
            if prompts.digits[i, prompts.digit_names.index(name)]:
                out[a:b, d] = IGNORE
    return out
