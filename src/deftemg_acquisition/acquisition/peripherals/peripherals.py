from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Peripheral(ABC):
    """An extra device recorded with every trial of an acquisition session. The armband is not
    one: it is the session's own device.

    ``name`` is the key in ``session.json`` and ``trials.jsonl`` and the prefix on its arrays.
    """

    name: str

    @abstractmethod
    def open(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def start_acquisition(self, index: int, host_start_ns: int) -> None:
        """Begin keeping the frames of trial ``index`` that arrive from ``host_start_ns``."""

    @abstractmethod
    def stop_acquisition(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """End the trial: arrays for the trial file and metadata for ``trials.jsonl``."""

    def save_files(self, session_dir: Path) -> None:
        """Write files to keep with the session, beside ``session.json``: called once, after
        ``open``, with the session folder made. None by default."""
        return

    def status(self) -> str | None:
        """A problem to show the operator now (e.g. no data arriving), or None if all is well.
        Asked about once a second from the GUI thread, so it must be quick."""
        return None
