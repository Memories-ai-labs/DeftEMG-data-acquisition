"""Put a recording made over a wireless link on the computer's clock, offline.

    pixi run align-link                           # latest real session in data/raw
    pixi run align-link data/raw/<session>

For each trial writes ``data/processed/<session>/<trial>_aligned.npz``: ``emg_t_ns`` and
``imu_t_ns`` (computer time of every packet, the clock of ``*_host_time_ns``), ``emg_valid`` /
``imu_valid`` (placed by a fitted segment), ``emg_index`` / ``imu_index`` (band packet index),
and prints each segment's fit and its scatter. The method is ``analysis.link_alignment``.

A trial recorded before timing packets were saved is decoded again from its ``.bin``, using the
host times of its chunks saved with it. Trials with no timing packets are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deftemg_acquisition.acquisition import protocol as P
from deftemg_acquisition.analysis.link_alignment import align_link_trial
from deftemg_acquisition.data.trials import (
    is_simulated,
    read_records,
    read_session_info,
)


def trial_arrays(npz: Path, filter_enabled: bool) -> dict[str, np.ndarray]:
    """The trial's arrays with the ``marker_*`` ones, decoded from its ``.bin`` if need be."""
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files}
    if "marker_no" in arrays:
        return arrays
    d = P.decode(npz.with_suffix(".bin").read_bytes(), filter_enabled)
    arrays.update({k: v for k, v in d.as_dict().items() if k.startswith("marker_")})
    ends, times = arrays["chunk_end_offset"], arrays["chunk_host_time_ns"]
    last_byte = d.markers["offset"] + P.packet_len(filter_enabled)
    idx = np.minimum(np.searchsorted(ends, last_byte, side="left"), max(len(times) - 1, 0))
    arrays["marker_host_time_ns"] = times[idx] if len(times) else np.empty(0, np.int64)
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?", help="session folder (default: latest)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
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
    filter_enabled = bool(read_session_info(session).get("filter_status", {}).get("enabled", True))
    out_dir = args.out_dir / session.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Session {session} -> {out_dir}")
    for rec in read_records(session):
        npz = session / Path(rec["npz_path"]).name
        arrays = trial_arrays(npz, filter_enabled)
        if len(arrays["marker_no"]) == 0:
            print(f"{npz.stem}: no timing packets, skipped")
            continue
        al = align_link_trial(arrays)
        np.savez_compressed(
            out_dir / f"{npz.stem}_aligned.npz",
            emg_t_ns=al.emg_t_ns, imu_t_ns=al.imu_t_ns,
            emg_valid=al.emg_valid, imu_valid=al.imu_valid,
            emg_index=al.emg_index, imu_index=al.imu_index,
        )  # fmt: skip
        print(f"{npz.stem}:\n{al.summary()}")


if __name__ == "__main__":
    main()
