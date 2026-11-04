"""Extra devices recorded alongside the armband, trial by trial (``AcquisitionSession``'s
``sources``).

A source streams on its own (its own threads, its own SDK) and stamps every sample with
:func:`deftemg_acquisition.clock.host_ns` on arrival, so it shares the armband's timeline. The
session calls it at the same points as the armband:

- ``open()`` once, after the armband is ready; returns info for ``session.json``
  (``"sources": {name: info}``). An error here aborts the session.
- ``start_trial(index, host_start_ns)`` / ``end_trial(host_end_ns)`` around each trial;
  ``end_trial`` returns ``(arrays, metadata)``. The arrays are saved into the trial ``.npz``
  as ``<name>_<key>``, the metadata into the trial's ``trials.jsonl`` line
  (``"sources": {name: metadata}``). An error here is logged and recorded as
  ``{"ok": false, "error": ...}``; the armband data is saved regardless.
- ``close()`` once, when the session closes.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TrialSource(Protocol):
    name: str  # prefix of the saved arrays and key in session.json / trials.jsonl

    def open(self) -> dict[str, Any]: ...

    def start_trial(self, index: int, host_start_ns: int) -> None: ...

    def end_trial(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    def close(self) -> None: ...
