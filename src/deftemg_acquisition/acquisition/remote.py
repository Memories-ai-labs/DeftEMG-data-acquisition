"""The armband on another computer: its serial bytes over the network, as if it were plugged in.

On the computer with the armband, a session (``serve``, or ``acquire`` / ``prompter``, which
publish it too) sends every chunk it reads from the serial port, unchanged, as an LSL stream
(:class:`RawStreamer`: type ``RAW_STREAM_TYPE``, sharing the session's ``source_id``). On the
other computer :class:`RemoteArmband` reads that stream and hands the same chunks to its sinks,
as :class:`Armband`'s reader does:

    device = RemoteArmband("waveletech:/dev/cu.usbmodem1101")      # or the sending host's name
    config = SessionConfig(filter_enabled=device.resolve().enabled, ...)
    with AcquisitionSession(config, device=device) as session: ...  # records, streams, as usual

so recording (the ``.bin`` + ``.npz`` trials), the live streams, the effort meter and any other
sink work unchanged. The bytes are lossless: each chunk carries its offset in the sender's
stream, so a gap (the connection lost and recovered) is logged and counted, and shows in the
trial as dropped packets, as on a serial port.

Each chunk keeps the time it arrived at the sender's serial port, moved onto this computer's
clock by LSL's clock synchronisation (``time_correction``), not the time it arrived here: the
network's delay and jitter don't bend the EMG timeline, and it lines up with this computer's
prompts to within the clock synchronisation's error (well under 1 ms on a wired LAN).

The armband stays under the control of the computer it is plugged into: its filter setting is
the sender's (``filter_status``; ``configure_filters`` refuses another, as a replay does), and
commands (``TIME``, start / stop) are not sent across. A session here selects which bytes it
saves; trials need ``continuous`` (the default), as the sender streams throughout.
"""

from __future__ import annotations

import base64
import collections
import logging
import socket
import threading

import numpy as np
import pylsl

from deftemg_acquisition.clock import host_ns

from .armband import protocol as P
from .armband.device import Armband, DeviceError
from .armband.live import STREAM_PREFIX

log = logging.getLogger(__name__)

RAW_STREAM_TYPE = "ArmbandBytes"
RAW_STREAM_NAME = f"{STREAM_PREFIX}-Raw"
RAW_CHANNELS = ["offset", "data_base64"]  # byte offset of the chunk in the stream, its bytes


def lsl_minus_host_s() -> float:
    """LSL clock minus the host clock (``clock.host_ns``), in s."""
    return pylsl.local_clock() - host_ns() / 1e9


class RawStreamer:
    """Armband sink publishing every serial chunk unchanged over LSL (``RAW_STREAM_TYPE``).

    Runs on the reader thread: one LSL string sample per chunk (its byte offset and its bytes
    in base64), stamped with the chunk's receive time on the LSL clock. Pushing a sample only
    queues it for the connected inlets, so it never waits on the network.
    """

    def __init__(self, source_id: str, filter_status: P.FilterStatus, port: str | None = None):
        self.source_id = source_id
        self.n_bytes = 0
        self._offset = lsl_minus_host_s()
        info = pylsl.StreamInfo(RAW_STREAM_NAME, RAW_STREAM_TYPE, len(RAW_CHANNELS),
                                pylsl.IRREGULAR_RATE, pylsl.cf_string, source_id)  # fmt: skip
        desc = info.desc()
        desc.append_child_value("manufacturer", "WAVELETECH")
        desc.append_child_value("encoding", "offset, base64 serial bytes")
        desc.append_child_value("port", str(port or ""))
        desc.append_child_value("hostname", socket.gethostname())
        filt = desc.append_child("device_filter")
        filt.append_child_value("enabled", str(int(filter_status.enabled)))
        filt.append_child_value("powerline_hz", str(filter_status.powerline_hz))
        filt.append_child_value("highpass_hz", str(filter_status.highpass_hz))
        chans = desc.append_child("channels")
        for label in RAW_CHANNELS:
            chans.append_child("channel").append_child_value("label", label)
        self._outlet = pylsl.StreamOutlet(info, chunk_size=1)

    def __call__(self, t_ns: int, data: bytes) -> None:
        sample = [str(self.n_bytes), base64.b64encode(data).decode("ascii")]
        self.n_bytes += len(data)
        self._outlet.push_sample(sample, t_ns / 1e9 + self._offset)

    def have_consumers(self) -> bool:
        return self._outlet.have_consumers()


def _filter_status(info: pylsl.StreamInfo) -> P.FilterStatus:
    filt = info.desc().child("device_filter")
    if filt.empty():
        raise DeviceError(f"{info.source_id()}: no device_filter in the stream's description")
    return P.FilterStatus(
        filt.child_value("enabled") == "1",
        int(filt.child_value("powerline_hz")),
        int(filt.child_value("highpass_hz")),
    )


def find_raw_streams(source: str | None = None, timeout: float = 5.0) -> list[pylsl.StreamInfo]:
    """The armband byte streams on the network: from ``source`` (a session's ``source_id``, or
    the sending computer's host name) or, with None, all of them; waits up to ``timeout`` s for
    at least one."""
    pred = f"type='{RAW_STREAM_TYPE}'"
    if source:
        pred += f" and (source_id='{source}' or hostname='{source}')"
    found = pylsl.resolve_bypred(pred, 1, timeout)
    if found:  # others may answer a moment later
        found += pylsl.resolve_bypred(pred, 0, 0.5)
    unique = {(s.source_id(), s.hostname(), s.uid()): s for s in found}
    return list(unique.values())


class RemoteArmband(Armband):
    """An ``Armband`` whose bytes come from another computer's session over LSL.

    ``source``: the sending session's ``source_id`` (``serve`` prints it) or its host name;
    None takes the only armband stream on the network. The reader thread runs from
    :meth:`open` to :meth:`close` and hands chunks to the sinks while acquiring.

    ``n_gaps`` / ``n_bytes_lost`` count bytes the sender sent that never arrived (connection
    lost and recovered); ``latency_ms`` holds, per recent chunk, how long after it reached the
    sender's serial port it was handed to the sinks here.
    """

    RESOLVE_TIMEOUT_S = 10.0
    MAX_BUFFER_S = 60  # LSL keeps this much for us if we fall behind; older data is lost

    def __init__(self, source: str | None = None, timeout: float | None = None):
        super().__init__(f"remote:{source or 'any'}")
        self.source = source
        self.timeout = self.RESOLVE_TIMEOUT_S if timeout is None else timeout
        self.info: pylsl.StreamInfo | None = None
        self._inlet: pylsl.StreamInlet | None = None
        self._status: P.FilterStatus | None = None
        self.n_chunks = 0
        self.n_gaps = 0
        self.n_bytes_lost = 0
        self.latency_ms: collections.deque[float] = collections.deque(maxlen=20_000)
        self._next_offset: int | None = None
        self._last_t_ns = 0

    # --- the stream ----------------------------------------------------------------------------

    def resolve(self) -> P.FilterStatus:
        """Find the sender's stream (once) and return its armband filter setting."""
        if self.info is not None:
            return self._status
        found = find_raw_streams(self.source, self.timeout)
        what = f"from {self.source!r} " if self.source else ""
        if not found:
            raise DeviceError(
                f"no armband stream {what}on the network after {self.timeout:g} s: is the "
                "session on the other computer running, and LSL reaching it (README, "
                '"The armband on another computer")?'
            )
        if len(found) > 1:
            names = ", ".join(f"{s.source_id()} on {s.hostname()}" for s in found)
            raise DeviceError(f"several armband streams {what}({names}): pick one by source_id")
        # proc_clocksync: timestamps arrive on our LSL clock (time_correction applied)
        self._inlet = pylsl.StreamInlet(
            found[0], max_buflen=self.MAX_BUFFER_S, recover=True,
            processing_flags=pylsl.proc_clocksync,
        )  # fmt: skip
        self.info = self._inlet.info(timeout=self.timeout)  # with its description
        self._status = _filter_status(self.info)
        log.info("armband stream %s from %s: %s", self.info.source_id(), self.info.hostname(),
                 self._status)  # fmt: skip
        return self._status

    def describe(self) -> dict[str, str]:
        """Where the bytes come from (stored in ``session.json``)."""
        if self.info is None:
            return {}
        return {"remote_source_id": self.info.source_id(), "remote_host": self.info.hostname(),
                "remote_port": self.info.desc().child_value("port")}  # fmt: skip

    # --- Armband's interface, without a port --------------------------------------------------

    def open(self) -> None:
        self.resolve()
        self._inlet.open_stream(timeout=self.timeout)
        self._stop_reader.clear()
        self._reader = threading.Thread(target=self._read_loop, name="remote-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        if self._reader is None:
            return
        self.acquiring = False
        self._stop_reader.set()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        self._reader = None
        self._inlet.close_stream()
        if self.n_gaps:
            log.warning("remote armband: %d gaps, %d bytes lost", self.n_gaps, self.n_bytes_lost)

    def ensure_stopped(self) -> None:
        self.acquiring = False

    def filter_status(self) -> P.FilterStatus:
        return self.resolve()

    def configure_filters(self, powerline_hz: int, highpass_hz: int, enabled: bool | None = None):
        status = self.resolve()
        if enabled is not None and enabled != status.enabled:
            state = "on" if status.enabled else "off"
            raise DeviceError(f"the armband's filter is {state} on the computer it is plugged "
                              "into: set it there (or take its setting here)")  # fmt: skip
        return status

    def set_timestamp(self, unix_ms: int) -> None:
        pass  # the sender sets the armband's clock (from its wall clock) when it starts

    def send(self, cmd: str, expect: str | None = None, timeout: float = 1.0,
             allow_unknown: bool = False) -> bytes:  # fmt: skip
        return b""

    def start_acquisition(self) -> None:
        self.acquiring = True

    def stop_acquisition(self) -> None:
        self.acquiring = False

    # --- reading -----------------------------------------------------------------------------

    def _read_loop(self) -> None:
        assert self._inlet is not None
        lsl_minus_host = lsl_minus_host_s()
        stalled = False
        while not self._stop_reader.is_set():
            try:
                # Wait for one sample, then take what else has come: pull_chunk with a timeout
                # would wait for its whole buffer to fill (up to the timeout), adding latency.
                sample, stamp = self._inlet.pull_sample(timeout=0.05)
                samples, stamps = self._inlet.pull_chunk(timeout=0.0)
                if sample is not None:
                    samples, stamps = [sample, *samples], [stamp, *stamps]
            except pylsl.LostError:
                log.error("remote armband stream lost")
                return
            if not samples:
                idle_s = (host_ns() - self._last_t_ns) / 1e9
                if self.acquiring and self._last_t_ns and idle_s > 2.0 and not stalled:
                    stalled = True  # the sender stopped, or the network is down
                    log.warning("no data from the remote armband for %.0f s", idle_s)
                continue
            stalled = False
            for sample, stamp in zip(samples, stamps, strict=True):
                self._deliver(sample, stamp, lsl_minus_host)

    def _deliver(self, sample: list[str], stamp: float, lsl_minus_host: float) -> None:
        offset, data = int(sample[0]), base64.b64decode(sample[1])
        if self._next_offset is not None and offset != self._next_offset:
            if offset > self._next_offset:
                lost = offset - self._next_offset
                self.n_gaps += 1
                self.n_bytes_lost += lost
                log.warning("remote armband: %d bytes lost in transit", lost)
            else:
                log.warning("remote armband: the sender restarted its stream")
        self._next_offset = offset + len(data)
        now = host_ns()
        # The sender's receive time on our host clock; never before the previous chunk's (a new
        # clock-offset estimate can step it back), nor in the future (an offset error).
        t_ns = min(now, max(round((stamp - lsl_minus_host) * 1e9), self._last_t_ns))
        self._last_t_ns = t_ns
        if not self.acquiring:
            return
        self.n_chunks += 1
        for name, sink in self._sinks.items():
            try:
                sink(t_ns, data)
            except Exception:
                log.exception("sink %r failed", name)
        self.latency_ms.append((host_ns() - t_ns) / 1e6)

    def latency_summary(self) -> dict[str, float]:
        """Median, 95th percentile and maximum of ``latency_ms`` (ms); empty before any data."""
        if not self.latency_ms:
            return {}
        lat = np.asarray(self.latency_ms)
        return {"median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
                "max_ms": float(lat.max())}  # fmt: skip
