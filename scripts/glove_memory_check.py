"""Glove recorder memory check: one long trial with a stand-in Wuji SDK at the glove's rates.

    pixi run python scripts/glove_memory_check.py fast [--no-cloud] [--data structured]
    pixi run python scripts/glove_memory_check.py real [--no-cloud] [--data structured]

Every subscribed stream (``TOPICS``, or all but ``tactile_point_cloud`` with ``--no-cloud``)
sends ``--rate`` frames a second for ``--minutes`` (default 15 min at 120 Hz), frames of the
glove's sizes. ``--data random`` fills them with random values (the worst case for
compression); ``--data structured`` with smooth values, closer to what a glove gives (contact
on part of the hand, the rest zero; palm taxels fixed in the wrist frame, finger taxels moving).

- ``fast``: the recorder only (``WujiGloveRecorder`` start, frames, stop), with made-up arrival
  times, so it runs as fast as frames can be made. Passes when the peak memory during the
  trial, less the peak before it, is at most 1.1 x the bytes of the arrays returned + 64 MB.
- ``real``: a whole session (simulated armband) in real time, frames from a feeder thread, the
  trial saved by ``AcquisitionSession.end_trial`` (``np.savez_compressed``); reports the save
  time. ``--unpaced`` sends the frames as fast as it can instead (a shorter trial, the same
  glove data).

Memory is the peak resident size (``ru_maxrss``) and, on macOS, also the peak physical
footprint, which counts memory the system compressed or swapped out (the resident size leaves
it out, so it reads low when memory is short); the check uses the larger rise. Prints one line
of results (sizes in MB of 2**20 bytes).
"""

import argparse
import ctypes
import os
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import numpy as np

from deftemg_acquisition.acquisition.wuji import WujiGloveRecorder, wuji_recorder
from deftemg_acquisition.acquisition.wuji.calibration import SDK_USER
from deftemg_acquisition.acquisition.wuji.wuji_recorder import CLOUD_POINTS, TOPICS
from deftemg_acquisition.clock import host_ns

TAXELS = 744  # 24 x 31, the tactile grid (an upper bound on the taxel count)
EMF = 6  # EMF tracking poses
ZONES = ("palm", "thumb", "index", "middle", "ring", "pinky")
MB = 2**20


class _RUsageInfoV4(ctypes.Structure):  # macOS <sys/resource.h> rusage_info_v4
    _fields_ = [("uuid", ctypes.c_uint8 * 16), ("values", ctypes.c_uint64 * 35)]


def peaks_mb() -> dict[str, float]:
    """Peak resident size, and on macOS the peak physical footprint, so far (MB)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        return {"rss": r / 2**10}  # KiB on Linux
    info = _RUsageInfoV4()
    if ctypes.CDLL("libc.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        raise OSError("proc_pid_rusage failed")
    return {"rss": r / MB, "footprint": info.values[28] / MB}  # ri_lifetime_max_phys_footprint


def memory(baseline: dict[str, float], peak: dict[str, float], nbytes: float) -> dict[str, Any]:
    """The peaks before and during the trial, the rise against the data returned."""
    out: dict[str, Any] = {}
    for key in baseline:
        out[f"{key}_baseline_mb"], out[f"{key}_peak_mb"] = baseline[key], peak[key]
    rise = max(peak[key] - baseline[key] for key in baseline)
    out.update(returned_mb=nbytes, rise_mb=rise, ratio=rise / nbytes)
    return out


class Frames:
    """The glove's frames, as the SDK gives them, one set per time step."""

    def __init__(self, kind: str, rate_hz: float):
        self.kind, self.rate_hz = kind, rate_hz
        self.rng = np.random.default_rng(0)
        layout = np.random.default_rng(1)
        self.taxel_phase = layout.uniform(0, 2 * np.pi, TAXELS)
        self.cloud_base = layout.uniform(-0.1, 0.1, (CLOUD_POINTS, 3)).astype(np.float32)
        self.cloud_finger = layout.random(CLOUD_POINTS) < 0.6  # taxels that move with fingers
        self.cloud_phase = layout.uniform(0, 2 * np.pi, CLOUD_POINTS)
        self.cloud_off = layout.random(CLOUD_POINTS) < 0.05  # never read (-1)

    def _values(self, n: int, t: float, phase: float = 0.0) -> list[float]:
        if self.kind == "random":
            return self.rng.standard_normal(n).tolist()
        return (0.3 * np.sin(2 * np.pi * 0.4 * t + phase + np.arange(n) * 0.7)).tolist()

    def _pose(self, t: float, i: int) -> NS:
        p, q = self._values(3, t, i), self._values(4, t, 10 + i)
        return NS(position=p, orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))

    def frame(self, topic: str, i: int) -> NS:
        t = i / self.rate_hz
        header = NS(seq=i, timestamp_us=int(t * 1e6), frame_id="")
        if topic == "tactile":
            if self.kind == "random":
                data = self.rng.random(TAXELS).tolist()
            else:  # contact on part of the hand, smooth; zero elsewhere
                data = np.maximum(0.0, np.sin(2 * np.pi * 0.3 * t + self.taxel_phase)).tolist()
            return NS(header=header, data=data)
        if topic == "hand_joint_angles":
            fingers = [NS(angles=self._values(5, t, f), confidence=1.0) for f in range(5)]
            return NS(header=header, fingers=fingers)
        if topic in ("emf_poses", "tip_poses"):
            n = EMF if topic == "emf_poses" else 5
            return NS(header=header, poses=[NS(pose=self._pose(t, k), confidence=1.0)
                                            for k in range(n)])  # fmt: skip
        if topic == "tactile_zones":
            values = self._values(len(ZONES), t)
            return NS(header=header, **dict(zip(ZONES, values, strict=True)))
        if topic == "tactile_point_cloud":
            if self.kind == "random":
                pts = self.rng.random((CLOUD_POINTS, 4), dtype=np.float32)
            else:
                pts = np.empty((CLOUD_POINTS, 4), np.float32)
                bend = 0.01 * np.sin(2 * np.pi * 0.4 * t + self.cloud_phase)
                pts[:, :3] = self.cloud_base + (self.cloud_finger * bend)[:, None]
                pressure = np.maximum(0.0, np.sin(2 * np.pi * 0.3 * t + self.cloud_phase) - 0.7)
                pts[:, 3] = np.where(self.cloud_off, -1.0, pressure)
            return NS(header=header, point_stride=16, data=pts.tobytes())
        raise KeyError(topic)


def install_fake_sdk(topics: tuple[str, ...]) -> dict[str, Any]:
    """A stand-in ``wuji_sdk`` with one glove offering ``topics``; the connected gloves."""
    found, connected = {"WG1": "right"}, {}

    class NotOffered(Exception):
        pass

    class Glove:
        def __init__(self, sn):
            self.serial_number, self.callbacks = sn, {}

        def hand_side(self):
            return NS(get=lambda: "right")

        def version(self):
            return NS(get=lambda: "1.0")

        def __getattr__(self, topic):
            if topic not in TOPICS:
                raise AttributeError(topic)
            if topic not in topics:
                raise NotOffered(topic)

            def subscribe(callback, on_error):
                self.callbacks[topic] = callback
                return NS(close=lambda: None, is_active=True)

            return lambda: NS(subscribe_with_callback=subscribe)

    class SdkManager:
        @staticmethod
        def instance():
            return SdkManager

        scan = staticmethod(lambda: [NS(sn=sn, device_type="glove") for sn in found])
        list_users = staticmethod(lambda: [{"user_id": "u1", "display_name": SDK_USER}])
        switch_user = staticmethod(lambda user_id: None)
        disconnect = staticmethod(lambda device_name: connected.pop(device_name))

        @staticmethod
        def connect(sn, device_name):
            connected[device_name] = Glove(sn)
            return connected[device_name]

    sys.modules["wuji_sdk"] = NS(SdkManager=SdkManager, DeviceType=NS(WujiGlove="glove"),
                                 WujiException=NotOffered)  # fmt: skip
    wuji_recorder.calibrate = lambda glove, side: {}
    return connected


def returned_bytes(rec: WujiGloveRecorder) -> list[Any]:
    """Wrap the recorder's ``stop_acquisition`` to note the bytes of the arrays it returns,
    then the peaks once it has returned (before anything is saved)."""
    noted: list[Any] = []
    stop = rec.stop_acquisition

    def wrapped(host_end_ns):
        arrays, meta = stop(host_end_ns)
        noted.extend([sum(a.nbytes for a in arrays.values()), peaks_mb()])
        return arrays, meta

    rec.stop_acquisition = wrapped
    return noted


def fast(args, topics: tuple[str, ...]) -> dict[str, Any]:
    install_fake_sdk(topics)
    rec = WujiGloveRecorder()
    rec.open()
    noted = returned_bytes(rec)
    frames = Frames(args.data, args.rate)
    n = int(args.minutes * 60 * args.rate)
    end = host_ns()
    start = end - int(args.minutes * 60e9)
    rec.start_acquisition(1, start)
    baseline = peaks_mb()
    t = time.perf_counter()
    for i in range(n):
        t_ns = start + int(i * 1e9 / args.rate)
        for (_, topic), buf in rec.buffers.items():
            buf.add(frames.frame(topic, i), t_ns)
    feed_s = time.perf_counter() - t
    t = time.perf_counter()
    arrays, meta = rec.stop_acquisition(end)
    stop_s = time.perf_counter() - t
    peak = peaks_mb()
    rec.close()
    assert all(s["n_frames"] == n for s in meta["streams"].values()), meta
    results = memory(baseline, peak, noted[0] / MB)
    limit = 1.1 * results["returned_mb"] + 64
    results.update(limit_mb=limit, ok=results["rise_mb"] <= limit, feed_s=feed_s, stop_s=stop_s)
    return results


def real(args, topics: tuple[str, ...]) -> dict[str, Any]:
    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband

    connected = install_fake_sdk(topics)
    tmp = Path(tempfile.mkdtemp())
    config = SessionConfig(data_dir=tmp, session_name="s", live_stream=False)
    rec = WujiGloveRecorder()
    noted = returned_bytes(rec)
    frames = Frames(args.data, args.rate)
    n = int(args.minutes * 60 * args.rate)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband()),
                            sources=[rec]) as session:  # fmt: skip
        glove = next(iter(connected.values()))
        stop, sent = threading.Event(), [0]

        def feed():  # paced (after the first n frames, with --unpaced) until stopped
            t0, i = time.perf_counter(), 0
            while not stop.is_set():  # the glove keeps streaming while the trial is saved
                for topic in topics:
                    glove.callbacks[topic](frames.frame(topic, i))
                i += 1
                sent[0] = i
                if args.unpaced and i <= n:
                    t0 = time.perf_counter() - i / args.rate
                    continue
                wait = t0 + i / args.rate - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)

        session.start_trial(label="glove_memory_check")
        baseline = peaks_mb()
        th = threading.Thread(target=feed, daemon=True)
        th.start()
        minute = 0
        while sent[0] < n:
            time.sleep(0.5)
            if int(sent[0] / args.rate / 60) > minute:
                minute = int(sent[0] / args.rate / 60)
                print(f"{minute} min: peaks {peaks_mb()} MB", flush=True)
        t = time.perf_counter()
        result = session.end_trial(task="glove_memory_check")
        save_s = time.perf_counter() - t
        stop.set()
        th.join()
    results = memory(baseline, peaks_mb(), noted[0] / MB)
    at_stop = memory(baseline, noted[1], noted[0] / MB)
    meta = result.sources["wuji_glove"]
    results.update(stop_ratio=at_stop["ratio"], save_s=save_s,
                   npz_mb=Path(result.npz_path).stat().st_size / MB,
                   n_frames={k: v["n_frames"] for k, v in meta["streams"].items()})  # fmt: skip
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=["fast", "real"])
    parser.add_argument("--no-cloud", action="store_true", help="leave out the point cloud")
    parser.add_argument("--data", choices=["random", "structured"], default="random")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--rate", type=float, default=120.0, help="frames per second")
    parser.add_argument("--unpaced", action="store_true", help="real: frames as fast as made")
    args = parser.parse_args()
    topics = tuple(t for t in TOPICS if not (args.no_cloud and t == "tactile_point_cloud"))
    results = (fast if args.mode == "fast" else real)(args, topics)
    head = f"{args.mode} cloud={'no' if args.no_cloud else 'yes'} data={args.data}"
    body = " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in results.items())  # fmt: skip
    print(f"{head} {args.minutes:g} min x {args.rate:g} Hz: {body}", flush=True)


if __name__ == "__main__":
    main()
