"""Timing report for a recorded session.

    pixi run check-timing                        # latest real session in data/raw
    pixi run check-timing data/raw/<session>

Sessions recorded with a simulated armband (--simulate) are skipped.

For each trial: EMG sample rate on the host clock and on the armband's own clock, dropped
packets, arrival jitter, USB delay (host receive time minus armband clock) and clock drift,
and, for prompter trials, the lag from each prompt being drawn at the target to the detected
EMG onset, and the delay from each prompt's scheduled onset to the frame that drew it.

Writes ``<trial>_timing.npz`` next to each trial (fitted per-sample EMG times ``emg_time_ns``,
detected onsets and lags) and ``timing.json`` with the summary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from deftemg_acquisition.analysis.timing import analyse_trial
from deftemg_acquisition.data.trials import is_simulated, read_records

LEGEND = """
EMG Hz / ppm  EMG sample rate on the host clock (ppm vs 2000 Hz)
dev ppm       EMG sample rate on the armband's own ms clock (ppm vs 2000 Hz)
drift         host clock vs armband clock, ppm; the three clocks agree when ppm ~= dev ppm - drift
drop          packets lost
stalls        times no data arrived for >0.1 s (total time); e.g. the computer slept
jit99         99th-percentile USB arrival jitter, ms
USB ms        host receive time minus armband clock (+~1 ms command delay), median (p5-p95),
              with the drift trend removed (referred to trial start)
onsets        prompts with a detected EMG onset / prompts
lag ms        median EMG onset minus the frame that drew the prompt at the target (IQR);
              includes display delay and the subject's anticipation
paint ms      scheduled prompt onset to that frame, median; display delay comes on top
              (measure once with --sync-patch)"""


def fmt(x: float, spec: str = ".1f") -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?", help="session folder (default: latest)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    session = args.session
    if session is None:
        sessions = sorted(
            p
            for p in args.data_dir.iterdir()
            if (p / "trials.jsonl").exists() and not is_simulated(p)
        )
        if not sessions:
            raise SystemExit(f"no recorded (non-simulated) sessions with trials in {args.data_dir}")
        session = sessions[-1]
    elif is_simulated(session):
        raise SystemExit(f"{session} was recorded with a simulated armband: no timing report.")
    records = read_records(session)
    print(f"Session {session}  ({len(records)} trials)\n")

    header = (
        f"{'trial':<28}{'EMG Hz':>9}{'ppm':>7}{'dev ppm':>9}{'drop':>6}{'stalls':>13}{'jit99':>7}"
        f"{'USB ms':>8}{'(p5-p95)':>13}{'drift':>7}{'onsets':>9}{'lag ms':>8}{'(IQR)':>17}"
        f"{'paint ms':>10}"
    )
    print(header)
    print("-" * len(header))
    summary = []
    warnings: list[str] = []
    for rec in records:
        npz = session / Path(rec["npz_path"]).name
        with np.load(npz) as data:
            timing, arrays = analyse_trial(data, rec.get("label") or npz.stem)
        np.savez_compressed(npz.with_name(npz.stem + "_timing.npz"), **arrays)
        summary.append(timing.as_dict())
        c, u = timing.clock, timing.usb
        onsets = f"{timing.n_onsets}/{timing.n_prompts}" if timing.n_prompts else "-"
        stalls = f"{c.n_stalls} ({c.stalled_s:.1f} s)" if c.n_stalls else "0"
        if c.n_stalls:
            warnings.append(
                f"{npz.stem}: {c.n_stalls} stalls, {c.stalled_s:.1f} s without data. Likely the"
                " computer slept; rates are fitted per continuous stretch."
            )
        iqr = f"({fmt(timing.lag_ms_q25, '.0f')} to {fmt(timing.lag_ms_q75, '.0f')})"
        print(
            f"{npz.stem[:27]:<28}{c.rate_hz:>9.2f}{c.rate_error_ppm:>7.0f}"
            f"{fmt(timing.device_clock.rate_error_ppm, '.0f'):>9}{timing.n_dropped:>6}"
            f"{stalls:>13}"
            f"{c.arrival_jitter_ms_p99:>7.1f}{fmt(u.median_ms):>8}"
            f"{'(' + fmt(u.p5_ms) + '-' + fmt(u.p95_ms) + ')':>13}{fmt(u.drift_ppm, '.0f'):>7}"
            f"{onsets:>9}{fmt(timing.lag_ms_median, '.0f'):>8}{iqr:>17}"
            f"{fmt(timing.paint_delay_ms_median):>10}"
        )
    (session / "timing.json").write_text(json.dumps(summary, indent=2, default=float))
    for w in warnings:
        print(f"\nWARNING {w}")
    print(LEGEND)


if __name__ == "__main__":
    main()
