"""The host clock used for every timestamp taken on the computer.

Host timestamps (EMG/IMU arrival, trial start/end, prompt draw times, ...) come from the
monotonic clock: it counts steadily and is never adjusted. The wall clock (``time.time_ns``)
is not used for them because the OS corrects it at any time, e.g. macOS slewing it by ~10 ms
over ~20 s to follow network time, which would bend the EMG timeline.

The wall clock is still needed for absolute dates and for the armband's ``TIME`` command, so
sessions and trials store :func:`anchor` pairs that convert between the two clocks.

Recordings made before this change used the wall clock throughout (``host_clock`` missing or
``"wall"``).
"""

from __future__ import annotations

import time

HOST_CLOCK = "monotonic"


def host_ns() -> int:
    """Current host time in ns (monotonic; meaningful only as differences and via anchors)."""
    return time.monotonic_ns()


def anchor() -> dict[str, int]:
    """A simultaneous (wall clock, host clock) reading, in ns.

    The wall clock is read between two host-clock reads and paired with their midpoint.
    """
    h1 = time.monotonic_ns()
    wall = time.time_ns()
    h2 = time.monotonic_ns()
    return {"wall_ns": wall, "host_ns": (h1 + h2) // 2}


def host_to_wall_ns(host: int, anchor_pair: dict[str, int]):
    """Host-clock ns -> wall-clock (Unix) ns, using an :func:`anchor` pair.

    Works on ints or numpy arrays.
    """
    return host + (anchor_pair["wall_ns"] - anchor_pair["host_ns"])
