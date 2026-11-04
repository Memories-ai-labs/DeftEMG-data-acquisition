"""A simulated Wuji Glove with the SDK's object shapes, for trying the pipeline without one.

It offers the same resources as ``wuji_sdk.WujiGlove`` (``glove.tactile()`` ...), each with
``subscribe_with_callback(callback, on_error)`` calling back from a background thread with
frames shaped like the SDK's (``frame.header.seq / timestamp_us``, ``frame.data``, ...).
Sizes and rates are as measured on a right glove's recording (SDK MCAP): 120 Hz frames, 768
tactile values (-1 where there is no taxel), five fingers of five angles, five EMF poses. The
IMU comes at the frame rate here (about 1 kHz on the glove, in bursts). For the real frames,
replay a recording instead (``wuji.replay``).
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace as NS

RATE_HZ = 120.0
N_TAXELS = 768  # tactile frame length; -1.0 marks positions without a taxel
ZONE_SIZES = {"palm": 308, "thumb": 60, "index": 54, "middle": 60, "ring": 54, "pinky": 55}
N_VALID = sum(ZONE_SIZES.values())
N_FINGERS, N_ANGLES = 5, 5
N_SKELETON = 21  # MediaPipe landmarks
N_EMF = 5


def _pose(p: float) -> NS:
    return NS(position=[p, 0.0, 0.0], orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


def _frame_maker(stream: str):
    def tactile(t):
        press = max(0.0, math.sin(2 * math.pi * 0.5 * t))  # a press every 2 s
        return {"data": [press * ((i % 31) / 31) if i < N_VALID else -1.0
                         for i in range(N_TAXELS)]}  # fmt: skip

    def angles(t):
        bend = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t)
        return {"fingers": [NS(angles=[bend * (k + 1) / N_ANGLES] * N_ANGLES, confidence=1.0)
                            for k in range(N_FINGERS)]}  # fmt: skip

    def poses(n, attr):
        return lambda t: {attr: [NS(pose=_pose(0.01 * k + 0.001 * t), confidence=1.0)
                                 for k in range(n)]}  # fmt: skip

    def zones(t):
        return {z: [0.0] * n for z, n in ZONE_SIZES.items()}

    def imu(t):
        q = NS(x=0.0, y=0.0, z=0.0, w=1.0)
        return {"orientation": q, "angular_velocity": NS(x=0.0, y=0.0, z=math.sin(t)),
                "linear_acceleration": NS(x=0.0, y=0.0, z=9.81)}  # fmt: skip

    if stream in ("tactile", "tactile_binary", "tactile_residual"):
        return tactile
    if stream.startswith("imu_data_"):
        return imu
    return {
        "hand_joint_angles": angles,
        "hand_skeleton": poses(N_SKELETON, "joints"),
        "emf_poses": poses(N_EMF, "poses"),
        "tactile_zones": zones,
    }[stream]


class _Subscription:
    def __init__(self, make, callback, rate_hz: float):
        self.received_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(make, callback, rate_hz), daemon=True, name="sim-glove"
        )
        self._thread.start()

    @property
    def is_active(self) -> bool:
        return self._thread.is_alive()

    def _run(self, make, callback, rate_hz: float) -> None:
        t0 = time.monotonic()
        period = 1.0 / rate_hz
        seq = 0
        while not self._stop.wait(max(0.0, t0 + seq * period - time.monotonic())):
            t = time.monotonic() - t0
            header = NS(seq=seq, timestamp_us=int(time.time() * 1e6), frame_id="sim")
            callback(NS(header=header, **make(t)))
            self.received_count += 1
            seq += 1

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class _Resource:
    def __init__(self, stream: str, rate_hz: float):
        self._make = _frame_maker(stream)
        self._rate_hz = rate_hz

    def subscribe_with_callback(self, callback, on_error=None) -> _Subscription:
        return _Subscription(self._make, callback, self._rate_hz)


class _Value:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class SimulatedGlove:
    serial_number = "SIMGLOVE0001"
    device_name = "simulated"
    simulated = True

    def __init__(self, rate_hz: float = RATE_HZ, hand: str = "right"):
        self.rate_hz = rate_hz
        self.hand = hand

    def __getattr__(self, name: str):
        if name == "hand_side":
            return lambda: _Value(self.hand)
        if name == "version":
            return lambda: _Value("sim")
        try:
            _frame_maker(name)
        except KeyError:
            raise AttributeError(name) from None
        return lambda: _Resource(name, self.rate_hz)
