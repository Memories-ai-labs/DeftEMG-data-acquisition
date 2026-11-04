"""A Wuji Glove played back from an MCAP recording, with the SDK's object shapes.

The SDK's ``TopicRecorder`` writes glove streams to MCAP as JSON messages on topics such as
``/right_glove/tactile`` or ``/right_glove/imu_data/palm``. ``ReplayGlove`` offers the same
resources as ``wuji_sdk.WujiGlove`` (``glove.tactile()`` ...) and calls back with the recorded
frames, so a recording can stand in for the glove (``--wuji-replay``)::

    glove = ReplayGlove("session.mcap")
    WujiGloveRecorder(glove=glove)

Frames keep their recorded header (``seq``, ``timestamp_us``: the glove's clock) and come at
the pace they arrived when recorded (the MCAP log time), divided by ``speed``: bursts and
jitter included. All streams share one timeline, started by the first subscription, so they
stay aligned with each other. The recording plays once, or again and again with ``loop``.
Needs the ``mcap`` package (optional dependency).
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

_PREFIX = re.compile(r"^/(left|right)_glove/")
_LOOP_GAP_NS = 500_000_000  # pause between passes when looping, as joined armband trials


def stream_topic(stream: str, prefix: str) -> str:
    """MCAP topic of a ``WujiGlove`` resource: ``imu_data_palm`` -> ``<prefix>imu_data/palm``."""
    m = re.fullmatch(r"(imu_data|imu_raw)_(\w+)", stream)
    return prefix + (f"{m[1]}/{m[2]}" if m else stream)


def _decode(data: bytes) -> NS:
    return json.loads(data, object_hook=lambda d: NS(**d))


def _reader(f):
    try:
        from mcap.reader import make_reader
    except ImportError as e:  # optional dependency
        raise RuntimeError("glove replay needs the mcap package: pip install mcap") from e
    return make_reader(f)


class _Subscription:
    def __init__(self, glove: ReplayGlove, topic: str, callback, on_error):
        self.received_count = 0
        self._glove = glove
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(topic, callback, on_error), daemon=True, name="replay-glove"
        )
        self._thread.start()

    @property
    def is_active(self) -> bool:
        return self._thread.is_alive()

    def _run(self, topic: str, callback: Callable, on_error: Callable | None) -> None:
        g = self._glove
        t0 = g._start_clock()
        try:
            for n_pass in range(1_000_000_000 if g.loop else 1):
                offset_ns = n_pass * (g.duration_ns + _LOOP_GAP_NS)
                with open(g.path, "rb") as f:
                    for _, _, msg in _reader(f).iter_messages(topics=[topic]):
                        rel_ns = offset_ns + msg.log_time - g.start_ns
                        wait = t0 + rel_ns / g.speed / 1e9 - time.monotonic()
                        if self._stop.wait(max(0.0, wait)):
                            return
                        callback(_decode(msg.data))
                        self.received_count += 1
                if self._stop.is_set():
                    return
        except Exception as e:  # reported like an SDK stream error
            if on_error is not None:
                on_error(f"replay {topic}: {type(e).__name__}: {e}")
            else:
                raise

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class _Resource:
    def __init__(self, glove: ReplayGlove, topic: str):
        self._glove = glove
        self._topic = topic

    def subscribe_with_callback(self, callback, on_error=None) -> _Subscription:
        return _Subscription(self._glove, self._topic, callback, on_error)


class _Value:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class ReplayGlove:
    """A glove that plays an MCAP recording made with the SDK. ``speed`` > 1 plays faster."""

    simulated = True
    device_name = "replay"

    def __init__(self, path: str | Path, speed: float = 1.0, loop: bool = False):
        if speed <= 0:
            raise ValueError("replay speed must be > 0")
        self.path = Path(path)
        self.speed = float(speed)
        self.loop = loop
        with open(self.path, "rb") as f:
            summary = _reader(f).get_summary()
        if summary is None or summary.statistics is None:
            raise RuntimeError(f"{self.path}: no MCAP summary (unfinished recording?)")
        self.topics = {ch.topic: summary.statistics.channel_message_counts.get(cid, 0)
                       for cid, ch in summary.channels.items()}  # fmt: skip
        prefixes = {m[0] for t in self.topics if (m := _PREFIX.match(t))}
        if len(prefixes) != 1:
            raise RuntimeError(f"{self.path}: expected one glove's topics, found {prefixes}")
        self.prefix = prefixes.pop()
        self.hand = _PREFIX.match(self.prefix)[1]
        self.start_ns = summary.statistics.message_start_time
        self.duration_ns = summary.statistics.message_end_time - self.start_ns
        self.serial_number = f"replay:{self.path.name}"
        self._lock = threading.Lock()
        self._t0: float | None = None

    def _start_clock(self) -> float:
        """The shared timeline's start (monotonic s), set by the first subscription."""
        with self._lock:
            if self._t0 is None:
                self._t0 = time.monotonic()
            return self._t0

    def available_streams(self, names) -> list[str]:
        return [s for s in names if self.topics.get(stream_topic(s, self.prefix))]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "hand_side":
            return lambda: _Value(self.hand)
        if name == "version":
            return lambda: _Value("replay")
        topic = stream_topic(name, self.prefix)
        if not self.topics.get(topic):
            raise AttributeError(f"{self.path.name} has no {topic} messages")
        return lambda: _Resource(self, topic)
