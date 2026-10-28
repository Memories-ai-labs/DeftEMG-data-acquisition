"""Host-system helpers for long-running commands (recording sessions)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)


def keep_awake(display: bool = True) -> subprocess.Popen | None:
    """Stop macOS idle sleep while this process runs; returns the ``caffeinate`` process.

    Terminate the returned process to allow sleep again; ``caffeinate -w`` also exits by itself
    if this process dies. ``display=False`` lets the screen sleep but keeps the system awake
    (enough for a long job with no window). Closing a laptop lid still sleeps it unless it has
    power and an external display. Returns None where ``caffeinate`` is unavailable.
    """
    if sys.platform != "darwin" or shutil.which("caffeinate") is None:
        return None
    flags = ["-d", "-i"] if display else ["-i"]
    log.info("keeping the computer awake (caffeinate %s)", " ".join(flags))
    return subprocess.Popen(["caffeinate", *flags, "-w", str(os.getpid())])
