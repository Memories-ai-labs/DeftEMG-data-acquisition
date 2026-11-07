"""Connect to a Wuji Glove and report what each stream actually sends (rate, shapes, clock).

    python -m deftemg_acquisition.acquisition.wuji.probe            # default streams, 10 s
    python -m deftemg_acquisition.acquisition.wuji.probe --all -s 30
    python -m deftemg_acquisition.acquisition.wuji.probe --simulate

Run it first on new hardware or after an SDK update: the simulated glove's sizes and rates
(``wuji.sim``) are placeholders to replace with what this prints.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .wuji_recorder import DEFAULT_STREAMS, STREAMS, WujiGloveRecorder


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sn", help="glove serial number (default: the only glove found)")
    ap.add_argument("--streams", nargs="+", default=list(DEFAULT_STREAMS), choices=sorted(STREAMS))
    ap.add_argument("--all", action="store_true", help="every known stream")
    ap.add_argument("-s", "--seconds", type=float, default=10.0)
    ap.add_argument("--simulate", action="store_true", help="probe the simulated glove")
    args = ap.parse_args(argv)

    glove = None
    if args.simulate:
        from .sim import SimulatedGlove

        glove = SimulatedGlove()
    rec = WujiGloveRecorder(sn=args.sn, streams=sorted(STREAMS) if args.all else args.streams,
                            glove=glove)  # fmt: skip
    info = rec.open()
    print("glove:", info)
    try:
        from deftemg_acquisition.clock import host_ns

        t0 = host_ns()
        rec.start_acquisition(0, t0)
        print(f"recording {args.seconds:g} s ...")
        time.sleep(args.seconds)
        arrays, meta = rec.stop_acquisition(host_ns())
    finally:
        rec.close()

    for stream, summary in meta["streams"].items():
        print(f"\n== {stream}: {summary}")
        host = arrays[f"{stream}_host_ns"]
        dev = arrays[f"{stream}_dev_us"]
        if len(host) > 1:
            jitter_ms = np.std(np.diff(host)) / 1e6
            # glove clock against the host clock over the recording (the clocks have different
            # epochs, so only the change in their difference means anything)
            off_ms = (dev * 1e3 - host) / 1e6
            drift = off_ms[-1] - off_ms[0]
            print(f"   arrival jitter {jitter_ms:.2f} ms; glove-host clock drift {drift:+.2f} ms "
                  f"over {(host[-1] - host[0]) / 1e9:.1f} s")  # fmt: skip
        for key, value in arrays.items():
            if key.startswith(f"{stream}_") and not key.endswith(("_host_ns", "_dev_us", "_seq")):
                print(f"   {key:40s} shape {value.shape}  range [{value.min():.3g}, "
                      f"{value.max():.3g}]" if value.size else f"   {key}: empty")  # fmt: skip


if __name__ == "__main__":
    main()
