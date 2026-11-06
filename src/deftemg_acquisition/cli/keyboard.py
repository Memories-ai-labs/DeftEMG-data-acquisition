"""Single-keypress input from the terminal (no Enter needed), for the CLI trial runner."""

from __future__ import annotations

import select
import sys
import termios
import tty
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def cbreak_keys(poll_s: float = 0.1) -> Iterator[Iterator[str | None]]:
    """Put the terminal in cbreak mode and yield an iterator of keypresses.

    The iterator yields ``None`` every ``poll_s`` with no key, so callers can do other work
    (e.g. show trial duration) between keys. Terminal settings are restored on exit.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    def keys() -> Iterator[str | None]:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], poll_s)
            yield sys.stdin.read(1) if ready else None

    try:
        tty.setcbreak(fd)
        yield keys()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
