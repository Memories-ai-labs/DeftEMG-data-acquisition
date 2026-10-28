"""Per-session calibration: the rest noise and each digit's maximum press.

Every prompter session starts with a calibration block (``tasks.sequence.calibration_sequence``):
quiet rest, then brief efforts as hard as is comfortable ("max"), one per digit by default
(``CALIBRATION_REPS``). Its ``task`` matches the session's task blocks: "press", each digit on
its own; "pinch" / "squeeze", the thumb with each other finger, before pinch / squeeze blocks
(their posture differs from pressing). From it:

- ``rest_sd_uv``: robust SD per channel over the rest (the electrodes' noise in this donning);
- ``max_rms_uv[digit]``: per channel, the EMG amplitude (RMS, with the rest noise removed:
  sqrt(max(RMS^2 - SD_rest^2, 0))) of the best maximum effort the digit takes part in (in a
  pinch calibration the thumb takes part in all four), measured from past the ramp (a third of
  the hold after the drawn onset, at most 1 s) to the drawn offset. The best of the
  repetitions, as in maximal-contraction practice; with two, ``rep_ratio`` (second / first, of
  the same effort) shows fatigue. A digit whose best effort is under ``MIN_SNR`` x the rest
  noise on every channel is left out (``warnings``);
- ``chord_max_rms_uv[chord]``: the same per effort ("index", or "thumb+index" for a pinch),
  the reference for task presses of that digit or pinch.

Both are measured on the EMG as it will be analysed: ``emg_filter`` (an ``EmgFilter``, e.g.
the default high-pass 20 Hz + ``comb60all``) filters the block as ``filter-emg`` filters
recordings (each continuous segment, zero-phase); None leaves it as recorded.

They put a session's presses on a common footing: ``press_fraction`` gives a press's EMG
amplitude relative to the maximum of the same digit, or the same pinch / squeeze (EMG, not
force: a force glove will give the real thing), and ``session_gain`` a factor that puts a
session's EMG on a common scale (``max``: the mean maximum-press amplitude -> 1,
so amplitudes are relative to maximum effort; ``rest``: the rest noise SD -> 1, mostly a
control; ``rest_channel``: each channel's rest noise SD -> 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from deftemg_acquisition.analysis.filtering import EmgFilter
from deftemg_acquisition.errors import ConfigError

from .labels import LabelSpec, rest_span_ns
from .trials import Trial, TrialInfo, load_trial, scan_trials

log = logging.getLogger(__name__)

CAL_SKIP_S = 1.0  # at most this much of a maximum press is skipped as its ramp
REST_GUARD_S = 1.0  # rest ends this long before the first maximum press


MIN_SNR = 3.0  # a maximum press below this x the rest noise (best channel) is left out


@dataclass
class Calibration:
    session: str
    trial_id: str
    rest_sd_uv: np.ndarray  # (E,)
    max_rms_uv: dict[str, np.ndarray] = field(default_factory=dict)  # digit -> (E,)
    rep_ratio: dict[str, float] = field(default_factory=dict)  # digit -> rep 2 / rep 1
    warnings: list[str] = field(default_factory=list)
    emg_filter: EmgFilter | None = None  # the filter it was measured with
    task: str = "press"  # what the block asked for: press, pinch or squeeze
    chord_max_rms_uv: dict[str, np.ndarray] = field(default_factory=dict)  # "thumb+index" -> (E,)

    def max_amplitude_uv(self, digit: str) -> float:
        """EMG amplitude of the digit's maximum press: RMS across channels of ``max_rms_uv``."""
        return _amplitude(self.max_rms_uv[digit])


def hold_rms(trial: Trial, onset_ns: int, offset_ns: int, skip_s: float):
    """(E,) RMS per channel from ``skip_s`` after ``onset_ns`` to ``offset_ns``, or None."""
    a, b = np.searchsorted(trial.emg_t_ns, [onset_ns + round(skip_s * 1e9), offset_ns])
    if b - a < 100:
        return None
    x = trial.emg_uv[a:b].astype(np.float64)
    return np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))


def ramp_skip_s(hold_s: float) -> float:
    """How much of a maximum press to skip as its ramp: a third of the scheduled hold, at most
    ``CAL_SKIP_S`` (0.5 s of the brief 1.5 s presses; 1 s of the 3 s ones of the first blocks)."""
    return min(CAL_SKIP_S, round(hold_s / 3, 3))


def _above_noise(rms: np.ndarray, rest_sd: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(rms**2 - rest_sd**2, 0.0))


def calibration_from_trial(trial: Trial, emg_filter: EmgFilter | None = None) -> Calibration:
    """From the block as recorded; ``emg_filter`` filters it first (see the module docstring).
    Raises ValueError when the block can't be used at all (too little rest). A digit without a
    usable maximum press is left out, with a line in ``warnings``."""
    if emg_filter is not None:
        trial = emg_filter.trial(trial)
    p = trial.prompts
    span = rest_span_ns(trial, LabelSpec(), skip_start_s=1.0, guard_s=REST_GUARD_S)
    a, b = np.searchsorted(trial.emg_t_ns, span)
    rest = trial.emg_uv[a:b].astype(np.float64)
    if len(rest) < 1000:
        raise ValueError(f"{trial.id}: calibration rest too short")
    rest_sd = 1.4826 * np.median(np.abs(rest - np.median(rest, axis=0)), axis=0)
    params = (trial.info.metadata.get("sequence") or {}).get("params") or {}
    task = params.get("task", "press")  # blocks from before the task field were all presses
    cal = Calibration(trial.info.session_dir.name, trial.id, rest_sd, emg_filter=emg_filter,
                      task=task)  # fmt: skip
    amps = {}  # effort (prompt) -> (E,) amplitude above the noise
    for i in range(len(p)):
        rms = hold_rms(trial, p.onset_ns[i], p.offset_ns[i],
                       ramp_skip_s(p.offset_s[i] - p.onset_s[i]))  # fmt: skip
        if rms is not None:
            amps[i] = _above_noise(rms, rest_sd)
    level = {i: _amplitude(a) for i, a in amps.items()}
    chord = [_chord(p, i) for i in range(len(p))]

    def usable(best: np.ndarray) -> float:  # on its best channel: an effort is focal
        return float(np.max(best / np.maximum(rest_sd, 1e-9)))

    for key in dict.fromkeys(chord):
        done = [i for i, c in enumerate(chord) if c == key and i in amps]
        if done and usable(best := amps[max(done, key=level.get)]) >= MIN_SNR:
            cal.chord_max_rms_uv[key] = best
    for d, name in enumerate(p.digit_names):
        own = [int(i) for i in np.flatnonzero(p.digits[:, d])]  # the efforts it takes part in
        done = [i for i in own if i in amps]
        if not done:
            cal.warnings.append(f"{name}: no usable maximum {task}; left out")
            continue
        best = amps[max(done, key=level.get)]  # the best repetition
        snr = usable(best)
        if snr < MIN_SNR:
            cal.warnings.append(f"{name}: maximum {task} only {snr:.1f}x the rest noise; left out")
            continue
        cal.max_rms_uv[name] = best
        # Fatigue: the first two repeats of the same effort (the thumb takes part in every pinch
        # of a pinch calibration; only its own repeats compare).
        same = [i for i in own if chord[i] == chord[own[0]]]
        if len(same) >= 2 and same[0] in amps and same[1] in amps and level[same[0]] > 0:
            cal.rep_ratio[name] = level[same[1]] / level[same[0]]
            if cal.rep_ratio[name] < 0.8:
                cal.warnings.append(f"{name}: second maximum {task} {cal.rep_ratio[name]:.0%} "
                                    "of the first (fatigue?)")  # fmt: skip
    return cal


def _chord(prompts, i: int) -> str:
    """A prompt's digits as one name: "index", or "thumb+index" for a pinch."""
    return "+".join(prompts.digit_names[d] for d in np.flatnonzero(prompts.digits[i]))


def _amplitude(rms_uv: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(rms_uv) ** 2)))


def calibration_trials(session_dir: Path) -> list[TrialInfo]:
    return [i for i in scan_trials([Path(session_dir)]) if i.is_calibration]


def session_blocks(
    session_dir: Path, emg_filter: EmgFilter | None = None
) -> list[tuple[int, TrialInfo, Calibration | str]]:
    """The session's completed calibration blocks, oldest first: (host start time, trial,
    its Calibration or why it can't be used)."""
    done = sorted((i for i in calibration_trials(session_dir) if i.completed), key=_start_ns)
    return [(_start_ns(i), i, _load_calibration(str(i.npz_path), _mtime_ns(i), emg_filter))
            for i in done]  # fmt: skip


def session_calibrations(
    session_dir: Path, emg_filter: EmgFilter | None = None
) -> list[tuple[int, Calibration]]:
    """The usable completed calibration blocks with their start times, oldest first."""
    blocks = session_blocks(session_dir, emg_filter)
    return [(t, c) for t, _, c in blocks if isinstance(c, Calibration)]


def session_calibration(
    session_dir: Path, before_ns: int | None = None, emg_filter: EmgFilter | None = None
) -> Calibration | None:
    """The calibration for a session (its latest usable block), or for a trial of it
    starting at ``before_ns``: the latest usable block recorded before it, else the first one
    after (a block recorded at the end of a session still beats none)."""
    cals = session_calibrations(session_dir, emg_filter)
    if not cals:
        return None
    if before_ns is None:
        return cals[-1][1]
    earlier = [c for t, c in cals if t <= before_ns]
    return earlier[-1] if earlier else cals[0][1]


def calibration_status(session_dir: Path, emg_filter: EmgFilter | None = None) -> str:
    """'none', 'incomplete' (only blocks stopped early), 'unusable' (completed blocks that
    can't be used) or 'ok' for the session's calibration blocks."""
    blocks = calibration_trials(session_dir)
    if not blocks:
        return "none"
    if not any(i.completed for i in blocks):
        return "incomplete"
    return "ok" if session_calibrations(session_dir, emg_filter) else "unusable"


def _start_ns(info: TrialInfo) -> int:
    return int(info.record.get("host_start_ns", 0))


def _mtime_ns(info: TrialInfo) -> int:
    try:
        return info.npz_path.stat().st_mtime_ns
    except OSError:
        return 0


@lru_cache(maxsize=64)
def _load_calibration(
    npz_path: str, mtime_ns: int, emg_filter: EmgFilter | None
) -> Calibration | str:
    """Cached per file (modification time, filter): callers ask repeatedly. A
    block that can't be used gives the reason instead. Nothing is logged here (a cached call
    would stay silent): callers report ``warnings``."""
    try:
        return calibration_from_trial(load_trial(Path(npz_path)), emg_filter)
    except ConfigError:  # the filter, not the block
        raise
    except (OSError, ValueError, KeyError) as e:
        return str(e)


def press_fraction(trial: Trial, cal: Calibration) -> list[dict]:
    """Per prompt with the same effort in the calibration block: its EMG amplitude (rest noise
    removed, from a quarter of the hold on) relative to that effort's maximum. A single-digit
    press needs a press calibration, a thumb + finger pinch a pinch calibration (squeezes
    alike); other prompts (chords, or a different posture) are left out, since their EMG has
    no maximum of its own to compare with. Rows name the digit or pinch ("thumb+index").
    ``trial`` must be filtered as the calibration was (``cal.emg_filter``)."""
    p = trial.prompts
    out = []
    for i in range(len(p)):
        key = _chord(p, i)
        if key not in cal.chord_max_rms_uv or (ref := _amplitude(cal.chord_max_rms_uv[key])) <= 0:
            continue
        hold_s = (p.offset_ns[i] - p.onset_ns[i]) / 1e9
        rms = hold_rms(trial, p.onset_ns[i], p.offset_ns[i], max(0.1, 0.25 * hold_s))
        if rms is None:
            continue
        d = int(np.flatnonzero(p.digits[i])[-1])  # a pinch's level: its finger's
        amp = _amplitude(_above_noise(rms, cal.rest_sd_uv))
        out.append({"prompt": int(i), "digit": key, "level": p.level_names[p.level[i, d]],
                    "fraction_of_max": amp / ref})  # fmt: skip
    return out


def session_gain(cal: Calibration, mode: str) -> float | np.ndarray:
    """Factor to multiply a session's EMG by to normalise it (``mode``): a number, or for
    ``rest_channel`` one per channel (E,): each channel's rest noise SD -> 1, floored at 5 % of
    the median so a dead channel isn't blown up (rest-referenced normalisation)."""
    if mode == "rest":
        return 1.0 / float(np.median(cal.rest_sd_uv))
    if mode == "rest_channel":
        sd = np.asarray(cal.rest_sd_uv, dtype=np.float64)
        return 1.0 / np.maximum(sd, 0.05 * float(np.median(sd)))
    if mode == "max":
        if not cal.max_rms_uv:
            raise ValueError(f"{cal.trial_id}: no usable maximum presses")
        return 1.0 / float(np.mean([cal.max_amplitude_uv(d) for d in cal.max_rms_uv]))
    raise ValueError(f"normalisation {mode!r} is not one of rest, rest_channel, max")
