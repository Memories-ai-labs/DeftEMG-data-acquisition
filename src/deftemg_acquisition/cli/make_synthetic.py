"""Write synthetic prompter sessions in the recorded format (for dry runs and tests).

    pixi run make-synthetic                               # data/synthetic/S01, S02
    pixi run make-synthetic --sessions 3 --blocks 6 --seed 1
    pixi run make-synthetic --out data/synthetic/null --null    # EMG independent of prompts

Each session has --blocks prompt blocks (single-digit and two-digit chords, low / high force);
see deftemg_acquisition.data.synthetic for the signal model. Every tool reads them like
recordings, e.g. ``pixi run replay data/synthetic/S01`` or ``pixi run check-timing
data/synthetic/S01``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"), help="parent folder")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=6, help="blocks (trials) per session")
    parser.add_argument("--per-block", type=int, default=12, help="prompts per block")
    parser.add_argument("--strength", type=float, default=1.0, help="scale press SNR (>1 easier)")
    parser.add_argument("--stall", type=float, default=0.0, help="one data gap per trial, s")
    parser.add_argument("--null", action="store_true", help="EMG independent of the prompts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-calibration", action="store_true",
                        help="no calibration block at the start of each session")  # fmt: skip
    args = parser.parse_args(argv)
    out = []
    for k in range(args.sessions):
        cfg = SyntheticConfig(
            n_blocks=args.blocks,
            per_block=args.per_block,
            strength=args.strength,
            stall_s=args.stall,
            null=args.null,
            seed=args.seed * 1000 + k,
            donning=k + 1,  # one subject, a new donning per session (to hold out a re-donned band)
            calibration=not args.no_calibration,
        )
        out.append(write_session(args.out / f"S{k + 1:02d}", cfg))
        print(f"wrote {out[-1]}")
    return out


if __name__ == "__main__":
    main()
