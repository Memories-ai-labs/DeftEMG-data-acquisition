"""Wire protocol for the WAVELETECH 18-channel EMG armband (hardware manual v1.4).

Packets are fixed length and start with ``D2 D2 D2``, then a type byte (``AA`` = EMG,
``BB`` = IMU/device info), then a sequence number shared by both packet types that wraps
at 256. Packet length depends on whether the built-in filter is enabled:

- filter on  (default): 41 bytes, EMG is 18 x int16 big-endian, uV = reading / 7.5
- filter off:           59 bytes, EMG is 18 x int24 big-endian, uV = reading / 60
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

BAUD_RATE = 2_000_000
N_EMG_CHANNELS = 18
# Electrode positions around the band: the opening between ch18 and ch1 is about one electrode
# wide, as if one electrode were missing there.
BAND_SLOTS = N_EMG_CHANNELS + 1
EMG_RATE_HZ = 2000.0
IMU_RATE_HZ = 208.0

# No data for this long while streaming = the stream stalled (e.g. the computer slept).
STALL_GAP_S = 0.1

SYNC = b"\xd2\xd2\xd2"
TYPE_EMG = 0xAA
TYPE_IMU = 0xBB
HEADER_LEN = 5  # 3 sync bytes + type + seq

PACKET_LEN_FILTER_ON = 41
PACKET_LEN_FILTER_OFF = 59

EMG_UV_PER_COUNT_FILTER_ON = 1 / 7.5
EMG_UV_PER_COUNT_FILTER_OFF = 1 / 60
GYRO_RAD_S_PER_COUNT = 0.001225
ACC_M_S2_PER_COUNT = 0.0005982
# The manual's table says "Temp./0.01 C" but its worked example (253 -> 25.3 C) and real
# packets (0x016E = 366 -> 36.6 C on skin) both indicate 0.1 C per count.
TEMP_C_PER_COUNT = 0.1

FILTER_STATUS_RE = re.compile(rb"([01])-(50|60)-(20|40|60)")


def packet_len(filter_enabled: bool) -> int:
    return PACKET_LEN_FILTER_ON if filter_enabled else PACKET_LEN_FILTER_OFF


# --- Commands -------------------------------------------------------------------------------

LINE_END = b"\r\n"

START_ACQ = "StartDataAcq"
STOP_ACQ = "StopDataAcq"
GET_FILTER_STATUS = "GetFilterStatus"
# The manual writes "UnknownCommand", but firmware v1.4 hardware replies "Unknown Command".
# The stopped device also gives this reply to StopDataAcq.
UNKNOWN_COMMAND_REPLY = "Unknown Command"
UNKNOWN_COMMAND_REPLIES = (b"Unknown Command", b"UnknownCommand")


def set_filter_enabled_cmd(enabled: bool) -> str:
    return f"SetFilterEN-{int(enabled)}"


def set_powerline_freq_cmd(hz: int) -> str:
    if hz not in (50, 60):
        raise ValueError(f"power-line filter frequency must be 50 or 60, got {hz}")
    return f"SetPFFilterFreq-{hz}"


def set_highpass_freq_cmd(hz: int) -> str:
    if hz not in (20, 40, 60):
        raise ValueError(f"high-pass filter frequency must be 20, 40 or 60, got {hz}")
    return f"SetHPFilterFreq-{hz}"


def set_timestamp_cmd(unix_ms: int) -> str:
    """``TIME`` followed by the 64-bit Unix millisecond timestamp as 16 hex digits."""
    if not 0 <= unix_ms < 2**64:
        raise ValueError(f"timestamp out of range: {unix_ms}")
    return f"TIME{unix_ms:016X}"


@dataclass(frozen=True)
class FilterStatus:
    enabled: bool
    powerline_hz: int
    highpass_hz: int

    @classmethod
    def parse(cls, reply: bytes) -> FilterStatus:
        m = FILTER_STATUS_RE.search(reply)
        if m is None:
            raise ValueError(f"unrecognised filter status reply: {reply!r}")
        return cls(m.group(1) == b"1", int(m.group(2)), int(m.group(3)))


# --- Packet decoding ------------------------------------------------------------------------


def find_packets(buf: bytes | bytearray | np.ndarray, length: int) -> np.ndarray:
    """Return start offsets of complete packets in ``buf``.

    Walks the chain of fixed-length packets, re-synchronising on the next ``D2 D2 D2 {AA,BB}``
    header whenever the expected next header is missing (dropped bytes, command echoes). This
    rejects payload bytes that happen to look like a header, since those are skipped over as
    part of an accepted packet.
    """
    a = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    if a.size < length:
        return np.empty(0, dtype=np.int64)
    n = a.size - 3
    is_hdr = (
        (a[:n] == 0xD2)
        & (a[1 : n + 1] == 0xD2)
        & (a[2 : n + 2] == 0xD2)
        & ((a[3 : n + 3] == TYPE_EMG) | (a[3 : n + 3] == TYPE_IMU))
    )
    candidates = np.flatnonzero(is_hdr)
    candidates = candidates[candidates + length <= a.size]

    starts: list[int] = []
    next_expected = -1
    for c in candidates.tolist():
        if c < next_expected:
            continue  # inside the previously accepted packet
        starts.append(c)
        next_expected = c + length
    return np.asarray(starts, dtype=np.int64)


def unwrap_seq(seq: np.ndarray, prev: int | None = None) -> np.ndarray:
    """Unwrap the 8-bit sequence counter into a monotonically increasing int64 count.

    ``prev`` is the last unwrapped value from earlier data, to continue across chunks.
    """
    if seq.size == 0:
        return np.empty(0, np.int64)
    s = seq.astype(np.int64)
    first = int(s[0]) if prev is None else prev + ((int(s[0]) - prev) % 256 or 256)
    steps = np.diff(s) % 256
    steps[steps == 0] = 256  # a full wrap with no packets between is indistinguishable
    return np.concatenate([[first], first + np.cumsum(steps)])


def _be_signed(cols: np.ndarray) -> np.ndarray:
    """Combine big-endian byte columns (..., nbytes) into signed int64 values."""
    nbytes = cols.shape[-1]
    val = np.zeros(cols.shape[:-1], dtype=np.int64)
    for i in range(nbytes):
        val = (val << 8) | cols[..., i].astype(np.int64)
    sign_bit = 1 << (8 * nbytes - 1)
    return (val ^ sign_bit) - sign_bit


@dataclass
class DecodedStream:
    """Decoded contents of a byte stream. Sequence numbers are unwrapped across both types."""

    emg_uv: np.ndarray  # (n_emg, 18) float64, microvolts
    emg_counts: np.ndarray  # (n_emg, 18) int32, raw ADC counts
    emg_seq: np.ndarray  # (n_emg,) int64
    emg_offset: np.ndarray  # (n_emg,) int64, byte offset of the packet in the raw stream
    imu_seq: np.ndarray  # (n_imu,) int64
    imu_offset: np.ndarray  # (n_imu,) int64
    temperature_c: np.ndarray  # (n_imu,)
    gyro_rad_s: np.ndarray  # (n_imu, 3) x, y, z
    acc_m_s2: np.ndarray  # (n_imu, 3) x, y, z
    battery_mv: np.ndarray  # (n_imu,)
    battery_pct: np.ndarray  # (n_imu,)
    device_time_ms: np.ndarray  # (n_imu,) uint64, Unix ms from the device clock
    n_dropped: int  # packets missing according to the shared sequence counter

    def as_dict(self) -> dict[str, np.ndarray | int]:
        return dict(self.__dict__)


def decode(buf: bytes | bytearray, filter_enabled: bool = True) -> DecodedStream:
    length = packet_len(filter_enabled)
    a = np.frombuffer(bytes(buf), dtype=np.uint8)
    starts = find_packets(a, length)
    return _decode_packets(a, starts, length, filter_enabled, prev_seq=None)


def _decode_packets(
    a: np.ndarray, starts: np.ndarray, length: int, filter_enabled: bool, prev_seq: int | None
) -> DecodedStream:
    pkts = a[starts[:, None] + np.arange(length)].reshape(-1, length)

    seq_all = unwrap_seq(pkts[:, 4], prev_seq)
    chain = seq_all if prev_seq is None else np.concatenate([[prev_seq], seq_all])
    n_dropped = int(np.sum(np.diff(chain) - 1)) if chain.size > 1 else 0
    is_emg = pkts[:, 3] == TYPE_EMG
    is_imu = pkts[:, 3] == TYPE_IMU

    emg_payload = pkts[is_emg, HEADER_LEN:]
    if filter_enabled:
        emg_counts = _be_signed(emg_payload[:, :36].reshape(-1, N_EMG_CHANNELS, 2))
        emg_uv = emg_counts * EMG_UV_PER_COUNT_FILTER_ON
    else:
        emg_counts = _be_signed(emg_payload[:, :54].reshape(-1, N_EMG_CHANNELS, 3))
        emg_uv = emg_counts * EMG_UV_PER_COUNT_FILTER_OFF

    imu = pkts[is_imu, HEADER_LEN:]
    words = _be_signed(imu[:, :16].reshape(-1, 8, 2))  # bytes 1-16 as 8 signed int16
    ts = imu[:, 17:25].astype(np.uint64)
    device_time_ms = np.zeros(len(imu), dtype=np.uint64)
    for i in range(8):
        device_time_ms = (device_time_ms << np.uint64(8)) | ts[:, i]

    return DecodedStream(
        emg_uv=emg_uv.astype(np.float64),
        emg_counts=emg_counts.astype(np.int32),
        emg_seq=seq_all[is_emg],
        emg_offset=starts[is_emg],
        imu_seq=seq_all[is_imu],
        imu_offset=starts[is_imu],
        temperature_c=words[:, 0] * TEMP_C_PER_COUNT,
        gyro_rad_s=words[:, 1:4] * GYRO_RAD_S_PER_COUNT,
        acc_m_s2=words[:, 4:7] * ACC_M_S2_PER_COUNT,
        battery_mv=words[:, 7].astype(np.int64),
        battery_pct=imu[:, 16].astype(np.int64),
        device_time_ms=device_time_ms,
        n_dropped=n_dropped,
    )


class StreamDecoder:
    """Incremental :func:`decode` for live data: feed serial chunks as they arrive.

    Packets split across chunks are carried over to the next call, and sequence numbers stay
    unwrapped across calls. ``n_dropped`` in each result counts gaps since the previous call.
    Byte offsets in the results are relative to the carried-over buffer, not the whole stream.
    """

    def __init__(self, filter_enabled: bool = True):
        self.filter_enabled = filter_enabled
        self.length = packet_len(filter_enabled)
        self._carry = b""
        self._last_seq: int | None = None
        self.n_dropped_total = 0

    def reset(self) -> None:
        """Forget partial data and sequence history (e.g. after chunks were discarded)."""
        self._carry = b""
        self._last_seq = None

    @property
    def n_carried(self) -> int:
        """Bytes kept from earlier calls; the next result's offsets count them first."""
        return len(self._carry)

    def feed(self, data: bytes) -> DecodedStream:
        buf = self._carry + data
        a = np.frombuffer(buf, dtype=np.uint8)
        starts = find_packets(a, self.length)
        consumed = int(starts[-1]) + self.length if starts.size else 0
        # Anything that could still be the start of an incomplete packet is kept for next time.
        self._carry = buf[max(consumed, len(buf) - self.length + 1) :]
        out = _decode_packets(a, starts, self.length, self.filter_enabled, self._last_seq)
        if starts.size:
            self._last_seq = int(max(out.emg_seq.max(initial=-1), out.imu_seq.max(initial=-1)))
        self.n_dropped_total += out.n_dropped
        return out
