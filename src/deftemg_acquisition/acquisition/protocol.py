"""Wire protocol for the WAVELETECH 18-channel EMG armband (hardware manual v1.4).

Packets are fixed length and start with ``D2 D2 D2``, then a type byte (``AA`` = EMG,
``BB`` = IMU/device info), then a sequence number shared by both packet types that wraps
at 256. Packet length depends on whether the built-in filter is enabled:

- filter on  (default): 41 bytes, EMG is 18 x int16 big-endian, uV = reading / 7.5
- filter off:           59 bytes, EMG is 18 x int24 big-endian, uV = reading / 60
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
# A timing packet some Wi-Fi links add, of the same length: framed over, never decoded.
TYPE_MARK = 0xCC
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


def find_packets(
    buf: bytes | bytearray | np.ndarray,
    length: int,
    *,
    final: bool = True,
    kinds: tuple[int, ...] = (TYPE_EMG, TYPE_IMU),
) -> np.ndarray:
    """Return start offsets of complete packets in ``buf``.

    Walks the chain of fixed-length packets, re-synchronising on the next ``D2 D2 D2 {AA,BB}``
    header whenever the expected next header is missing (dropped bytes, command echoes). This
    rejects payload bytes that happen to look like a header, since those are skipped over as
    part of an accepted packet.

    Bytes lost mid-packet leave a block that is the head of one packet and the tail of a later
    one. It is left out (the sequence numbers count it as dropped), not decoded as one block of
    garbage, so a packet is kept only if it holds no other header and a header follows it
    (a command echo or lost bytes after it cost one packet, as a loss does). A genuine
    payload holds ``D2 D2 D2 {AA,BB,CC}`` about once in 10^8 packets.

    The last packet of ``buf`` has no successor to check yet. With ``final`` (the whole data)
    it is kept; without (a live stream), it is left for the next call until a header follows.
    ``kinds`` are the packet types returned; the others still frame the stream.
    """
    a = np.frombuffer(buf, dtype=np.uint8) if not isinstance(buf, np.ndarray) else buf
    if a.size < length:
        return np.empty(0, dtype=np.int64)
    n = a.size - 3
    sync = (a[:n] == 0xD2) & (a[1 : n + 1] == 0xD2) & (a[2 : n + 2] == 0xD2)
    kind = a[3 : n + 3]
    headers = np.flatnonzero(sync & ((kind == TYPE_EMG) | (kind == TYPE_IMU) | (kind == TYPE_MARK)))
    candidates = np.flatnonzero(sync & np.isin(kind, kinds))
    candidates = candidates[candidates + length <= a.size]
    # headers strictly inside (c, c + length): the packet at c was cut short
    inner = np.searchsorted(headers, candidates + length) - np.searchsorted(headers, candidates + 1)
    candidates = candidates[inner == 0]
    after = candidates + length
    seen = after + 4 <= a.size  # the successor's header is within buf
    candidates = candidates[np.isin(after, headers) | (final & ~seen)]

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


def _le(cols: np.ndarray, signed: bool = False) -> np.ndarray:
    """Combine little-endian byte columns (..., nbytes) into int64 values."""
    nbytes = cols.shape[-1]
    val = np.zeros(cols.shape[:-1], dtype=np.uint64)
    for i in range(nbytes):
        val |= cols[..., i].astype(np.uint64) << np.uint64(8 * i)
    if signed:
        return val.view(np.int64)
    return val.astype(np.int64)


MARKER_MAGIC = b"SY"
MARKER_FIELDS = (
    "offset", "version", "pair", "bridge_tsf_us", "band_count", "prev_type", "dongle_tsf_us",
    "no", "prev_seq", "bridge_cmd_rx", "bridge_cmd_ok", "bridge_cmd_fail", "bridge_cmd_drop",
    "bridge_gaps", "bridge_lost", "bridge_dropped_bytes", "n_band_packets",
)  # fmt: skip


def decode_markers(a: np.ndarray, length: int, emg_offset: np.ndarray, imu_offset: np.ndarray):
    """Decode the timing packets (type ``TYPE_MARK``) a wireless link adds to the stream.

    Layout (little-endian, offsets from the packet's first byte): ``SY`` at 5, version at 7,
    pair number at 8, the bridge's Wi-Fi clock in us at 10 (int64) and the count of band packets
    it had received at 18 (uint32), the type of the band packet before it at 22, the dongle's
    Wi-Fi clock in us at 23 (int64), a running marker number at 31 (uint32). Version 2+ adds the
    bridge's command counters (35: received u16, written u16, failed u8, dropped u8); version
    3+ in a 59-byte packet adds where loss happened (41: gaps u16, packets missing u16, bytes
    dropped u32). Counters a version lacks read 0. Packets without ``SY`` are ignored.

    Returns int64 arrays by ``MARKER_FIELDS``: ``offset`` is the byte position in the stream and
    ``n_band_packets`` the EMG and IMU packets that precede it in it, so a marker can be put on
    the host's clock through the chunk that carried it. ``prev_seq`` is its byte 4, the
    sequence number of the band packet before it, kept out of the band's own sequence check.
    """
    starts = find_packets(a, length, kinds=(TYPE_MARK,))
    if starts.size:
        magic = a[starts[:, None] + np.arange(5, 7)] == np.frombuffer(MARKER_MAGIC, np.uint8)
        starts = starts[magic.all(1)]
    pk = a[starts[:, None] + np.arange(length)] if starts.size else np.empty((0, length), np.uint8)
    n = len(starts)
    zeros = np.zeros(n, np.int64)
    version = pk[:, 7].astype(np.int64) if n else zeros
    fields: dict[str, np.ndarray] = dict.fromkeys(MARKER_FIELDS, zeros)
    fields.update(
        offset=starts.astype(np.int64),
        version=version,
        pair=pk[:, 8].astype(np.int64) if n else zeros,
        prev_type=pk[:, 22].astype(np.int64) if n else zeros,
        prev_seq=pk[:, 4].astype(np.int64) if n else zeros,
    )
    if n:
        fields["bridge_tsf_us"] = _le(pk[:, 10:18], signed=True)
        fields["band_count"] = _le(pk[:, 18:22])
        fields["dongle_tsf_us"] = _le(pk[:, 23:31], signed=True)
        fields["no"] = _le(pk[:, 31:35])
        v2 = (version >= 2) & (length >= 41)
        v3 = (version >= 3) & (length >= 59)
        for name, lo, hi, has in (
            ("bridge_cmd_rx", 35, 37, v2), ("bridge_cmd_ok", 37, 39, v2),
            ("bridge_cmd_fail", 39, 40, v2), ("bridge_cmd_drop", 40, 41, v2),
            ("bridge_gaps", 41, 43, v3), ("bridge_lost", 43, 45, v3),
            ("bridge_dropped_bytes", 45, 49, v3),
        ):  # fmt: skip
            fields[name] = np.where(has, _le(pk[:, lo:hi]), 0)
        band = np.sort(np.concatenate([emg_offset, imu_offset]))
        fields["n_band_packets"] = np.searchsorted(band, starts).astype(np.int64)
    return fields


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
    markers: dict[str, np.ndarray] = field(default_factory=dict)  # decode_markers; whole data only

    def as_dict(self) -> dict[str, np.ndarray | int]:
        d = {k: v for k, v in self.__dict__.items() if k != "markers"}
        d.update({f"marker_{k}": v for k, v in self.markers.items()})
        return d


def decode(buf: bytes | bytearray, filter_enabled: bool = True) -> DecodedStream:
    length = packet_len(filter_enabled)
    a = np.frombuffer(bytes(buf), dtype=np.uint8)
    starts = find_packets(a, length)
    out = _decode_packets(a, starts, length, filter_enabled, prev_seq=None)
    out.markers = decode_markers(a, length, out.emg_offset, out.imu_offset)
    return out


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
        starts = find_packets(a, self.length, final=False)
        consumed = int(starts[-1]) + self.length if starts.size else 0
        # Anything that could still be the start of an incomplete packet, or of a packet whose
        # successor's header hasn't arrived, is kept for next time.
        self._carry = buf[max(consumed, len(buf) - self.length - 3) :]
        out = _decode_packets(a, starts, self.length, self.filter_enabled, self._last_seq)
        if starts.size:
            self._last_seq = int(max(out.emg_seq.max(initial=-1), out.imu_seq.max(initial=-1)))
        self.n_dropped_total += out.n_dropped
        return out
