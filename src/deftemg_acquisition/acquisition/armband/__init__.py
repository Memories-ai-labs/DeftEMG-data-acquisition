"""The WAVELETECH 18-channel EMG armband: serial driver, wire protocol, trial recorder, live
LSL streamer, and stand-ins (``sim``: a simulated armband; ``replay``: a recording played as
the armband).

The armband is the session's main device (``acquisition.session``); other devices join it as
sources (``acquisition.sources``, e.g. ``acquisition.wuji``).
"""

from .device import Armband, ChunkSink, DeviceError, find_port
from .protocol import DecodedStream, FilterStatus, decode
from .recorder import TrialRecorder

__all__ = [
    "Armband",
    "ChunkSink",
    "DecodedStream",
    "DeviceError",
    "FilterStatus",
    "TrialRecorder",
    "decode",
    "find_port",
]
