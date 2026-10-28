"""Filter a recorded session's EMG offline (default: 20 Hz high-pass + 60 Hz and every harmonic).

    pixi run filter-emg                                   # latest real session in data/raw
    pixi run filter-emg data/raw/<session>
    pixi run filter-emg data/raw/<session> --harmonics none    # no harmonics: 60 Hz notch only
    pixi run filter-emg data/raw/<session> --harmonics odd
    pixi run filter-emg data/raw/<session> --harmonics 3,5,9 --notch-width 1.5
    pixi run filter-emg data/raw/<session> --highpass none --notch none   # no filtering

The defaults are ``EmgFilter()`` (high-pass 20 Hz, ``comb60all``), the live stream's default
filter, here zero-phase. Writes ``data/processed/<session>/<trial>.npz`` for each trial with
``emg_uv`` (filtered, µV, float32), ``emg_t_ns`` (fitted host time per sample), ``segment``
(continuous-segment id; each segment is filtered on its own) and ``filter`` (the settings, as
JSON), plus ``filter.json`` in the session folder. Filtering is zero-phase unless ``--causal``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deftemg_acquisition.analysis.filtering import FilterConfig, filter_trial, notch_frequencies
from deftemg_acquisition.data.trials import is_simulated, load_trial, read_records

FS_HZ = 2000.0


def _hz_or_none(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def _harmonics(value: str) -> str | list[int]:
    if value in ("none", "odd", "all"):
        return value
    return [int(k) for k in value.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?", help="session folder (default: latest)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--highpass", type=_hz_or_none, default=20.0, help="Hz, or none")
    ap.add_argument("--notch", type=_hz_or_none, default=60.0, help="mains Hz, or none")
    ap.add_argument("--notch-width", type=float, default=2.0, help="-3 dB width of each notch (Hz)")
    ap.add_argument(
        "--harmonics", type=_harmonics, default="all",
        help="multiples of --notch notched as well: none, odd, all (default, as the live "
             "stream), or e.g. 3,5,9",
    )  # fmt: skip
    ap.add_argument("--causal", action="store_true", help="filter forward only (as live)")
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

    cfg = FilterConfig(
        highpass_hz=args.highpass,
        notch_hz=args.notch,
        notch_width_hz=args.notch_width,
        harmonics=args.harmonics,
        zero_phase=not args.causal,
    )
    notches = notch_frequencies(cfg, FS_HZ)
    out_dir = args.out_dir / session.name
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = {**cfg.as_dict(), "fs_hz": FS_HZ, "notch_frequencies_hz": notches}
    (out_dir / "filter.json").write_text(json.dumps(settings, indent=2))

    print(f"Session {session} -> {out_dir}")
    print(
        f"high-pass {cfg.highpass_hz or 'off'} Hz, notches {notches or 'off'} Hz, "
        f"{'zero-phase' if cfg.zero_phase else 'causal'}\n"
    )
    for rec in read_records(session):
        npz = session / Path(rec["npz_path"]).name
        trial = filter_trial(load_trial(npz), cfg, FS_HZ)
        device = trial.info.filter_status
        np.savez_compressed(
            out_dir / npz.name,
            emg_uv=trial.emg_uv.astype(np.float32),
            emg_t_ns=trial.emg_t_ns,
            segment=trial.segment,
            filter=json.dumps(settings),
        )
        rms = np.sqrt(np.mean(trial.emg_uv**2, axis=0))
        note = "  (device filter was on)" if device.get("enabled") else ""
        print(
            f"{npz.stem[:27]:<28}{len(trial.emg_uv) / FS_HZ:>7.1f} s"
            f"{len(trial.segment_bounds()):>4} seg   median rms {np.median(rms):7.1f} µV{note}"
        )


if __name__ == "__main__":
    main()
