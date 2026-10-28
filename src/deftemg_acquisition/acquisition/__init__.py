"""Data acquisition for the WAVELETECH 18-channel EMG armband."""

from .device import Armband, DeviceError, find_port
from .protocol import DecodedStream, FilterStatus, decode
from .recorder import TrialRecorder
from .session import AcquisitionSession, SessionConfig, TrialResult

__all__ = [
    "AcquisitionSession",
    "Armband",
    "DecodedStream",
    "DeviceError",
    "FilterStatus",
    "SessionConfig",
    "TrialRecorder",
    "TrialResult",
    "decode",
    "find_port",
]
