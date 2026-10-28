"""Generate balanced, randomised prompt sequences (one JSON file per block).

    pixi run make-sequences --set single --reps 2 --out protocols/S01
    pixi run make-sequences --set multi --chord-sizes 2 3 --chord-force independent \\
        --out protocols/S01
    pixi run make-sequences --set pinch --chord-sizes 2 --reps 3 --per-block 12 \\
        --out protocols/P01
    pixi run make-sequences --set squeeze --chord-sizes 2 --reps 3 --per-block 12 \\
        --out protocols/Q01

Each (digit(s), force level) combination appears ``--reps`` times, shuffled and split into
blocks of ``--per-block`` presses. Files are named ``<prefix>_<set>_bNN.json``. ``pinch`` and
``squeeze`` always join the thumb with other digits: by default one other digit or the whole
hand (``--chord-sizes`` counts the thumb: 2 5; 3 and 4 add two- and three-finger chords), and
the prompter shows their instructions before each block.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deftemg_acquisition.tasks.sequence import THUMB_SETS, GeneratorConfig, generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--set", dest="task_set", choices=["single", "multi", *THUMB_SETS], required=True
    )
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--prefix", default="seq", help="file name prefix")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--chord-sizes", type=int, nargs="+",
        help="digits per press, thumb included (default: 2 for multi; 2 5 for pinch and "
             "squeeze: thumb + one finger, or the whole hand)",
    )  # fmt: skip
    ap.add_argument("--chord-force", choices=["same", "independent"], default="same")
    defaults = GeneratorConfig()
    ap.add_argument(
        "--hold", type=float, nargs=2, default=list(defaults.hold_s), metavar=("MIN", "MAX"),
        help="press duration range, s",
    )  # fmt: skip
    ap.add_argument(
        "--gap", type=float, nargs=2, default=list(defaults.gap_s), metavar=("MIN", "MAX"),
        help="rest between presses, s",
    )  # fmt: skip
    ap.add_argument(
        "--lead-in", type=float, default=defaults.lead_in_s,
        help="s before the first prompt (keep >= the prompter's --lookahead)",
    )  # fmt: skip
    ap.add_argument("--per-block", type=int, default=15)
    ap.add_argument("--force", action="store_true", help="overwrite existing block files")
    ap.add_argument(
        "--seed", type=int, default=None,
        help="random seed (default: a new random one, printed and saved in each file)",
    )  # fmt: skip
    args = ap.parse_args()

    cfg = GeneratorConfig(
        task_set=args.task_set,
        reps=args.reps,
        chord_sizes=tuple(args.chord_sizes) if args.chord_sizes else None,
        chord_force=args.chord_force,
        hold_s=tuple(args.hold),
        gap_s=tuple(args.gap),
        lead_in_s=args.lead_in,
        per_block=args.per_block,
        seed=args.seed,
    )
    blocks = generate(cfg, args.prefix)
    existing = [args.out / f"{seq.name}.json" for seq in blocks]
    existing = [p for p in existing if p.exists()]
    if existing and not args.force:
        raise SystemExit(
            f"{len(existing)} block files already exist in {args.out} (e.g. {existing[0].name}); "
            "the default seed is random, so they would be replaced by different blocks. "
            "Use another --out, or --force"
        )
    for seq in blocks:
        path = args.out / f"{seq.name}.json"
        seq.save(path)
        print(f"{path}  {len(seq.prompts)} presses, {seq.duration_s:.0f} s")
    if blocks:
        print(f"seed {blocks[0].params['seed']} (regenerate with --seed)")


if __name__ == "__main__":
    main()
