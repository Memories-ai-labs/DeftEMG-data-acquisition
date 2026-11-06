"""Serial connection to the armband: command/reply handling and a background reader thread."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Protocol

import serial
from serial.tools import list_ports

from deftemg_acquisition.clock import host_ns

from . import protocol as P

log = logging.getLogger(__name__)

# Called from the reader thread with (host receive time in ns, chunk of bytes); see clock.py.
ChunkSink = Callable[[int, bytes], None]


class DeviceError(RuntimeError):
    pass


class SerialLike(Protocol):
    def read(self, size: int = 1) -> bytes: ...
    def write(self, data: bytes) -> int | None: ...
    def reset_input_buffer(self) -> None: ...
    def close(self) -> None: ...
    @property
    def in_waiting(self) -> int: ...


def find_port() -> str:
    """Return the single USB serial port that looks like the armband, or raise."""
    # macOS: /dev/cu.usbmodem*; Linux: /dev/ttyACM* (USB CDC) or /dev/ttyUSB*.
    ports = [p.device for p in list_ports.comports() if _looks_usb(p.device)]
    # macOS exposes each device as both /dev/tty.* and /dev/cu.*; cu.* is the one to open.
    cu = [p for p in ports if "/cu." in p]
    ports = cu or ports
    if len(ports) != 1:
        raise DeviceError(f"could not pick a serial port automatically, found {ports}; pass one")
    return ports[0]


def _looks_usb(device: str) -> bool:
    name = device.lower()
    return "usb" in name or name.startswith("/dev/ttyacm")


class Armband:
    """One armband on one serial port.

    A reader thread runs for as long as the port is open. Every chunk it reads is checked for
    command replies and handed to every attached sink (see :meth:`attach_sink`), e.g. a trial
    recorder and a live streamer side by side. Sinks run on the reader thread and must be fast.
    """

    REPLY_BUFFER_MAX = 64 * 1024
    STALL_WARN_S = 0.5  # warn when streaming data stops for longer than this

    def __init__(self, port: str | None = None, *, ser: SerialLike | None = None):
        self.port = port
        self._ser = ser
        self._sinks: dict[str, ChunkSink] = {}
        self._reply_buf = bytearray()
        self._last_rx_ns = 0
        self._cond = threading.Condition()
        self._stop_reader = threading.Event()
        self._reader: threading.Thread | None = None
        self.acquiring = False

    # --- lifecycle ---------------------------------------------------------------------------

    def open(self) -> None:
        if self._ser is None:
            port = self.port or find_port()
            self.port = port
            self._ser = serial.Serial(
                port, baudrate=P.BAUD_RATE, bytesize=8, parity="N", stopbits=1, timeout=0.02
            )
        self._stop_reader.clear()
        self._reader = threading.Thread(target=self._read_loop, name="armband-reader", daemon=True)
        self._reader.start()

    def close(self) -> None:
        if self._ser is None:
            return
        try:
            if self.acquiring:
                self.stop_acquisition()
        finally:
            self._stop_reader.set()
            if self._reader is not None:
                self._reader.join(timeout=1.0)
            self._ser.close()
            self._ser = None

    def __enter__(self) -> Armband:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _read_loop(self) -> None:
        assert self._ser is not None
        while not self._stop_reader.is_set():
            try:
                data = self._ser.read(max(1, self._ser.in_waiting))
            except serial.SerialException as e:
                log.error("serial read failed: %s", e)
                return
            if not data:
                continue
            t_ns = host_ns()
            idle_s = (time.monotonic_ns() - self._last_rx_ns) / 1e9
            if self.acquiring and self._last_rx_ns and idle_s > self.STALL_WARN_S:
                log.warning(
                    "no data from the armband for %.1f s (computer asleep? cable or battery?)",
                    idle_s,
                )
            for name, sink in self._sinks.items():
                try:
                    sink(t_ns, data)
                except Exception:
                    log.exception("sink %r failed", name)
            with self._cond:
                self._last_rx_ns = time.monotonic_ns()
                self._reply_buf += data
                if len(self._reply_buf) > self.REPLY_BUFFER_MAX:
                    del self._reply_buf[: -self.REPLY_BUFFER_MAX]
                self._cond.notify_all()

    # --- streaming sink ----------------------------------------------------------------------

    def attach_sink(self, name: str, sink: ChunkSink) -> None:
        # Replace the dict instead of mutating it so the reader thread can iterate lock-free.
        self._sinks = {**self._sinks, name: sink}

    def detach_sink(self, name: str) -> None:
        self._sinks = {k: v for k, v in self._sinks.items() if k != name}

    # --- commands ----------------------------------------------------------------------------

    def send(
        self,
        cmd: str,
        expect: str | None = None,
        timeout: float = 1.0,
        allow_unknown: bool = False,
    ) -> bytes:
        """Send ``cmd`` + CRLF. If ``expect`` is given, wait for it to appear in the reply.

        Returns the bytes received after sending, up to and including the expected reply (or
        the "Unknown Command" reply if ``allow_unknown``). Always wait for a reply before
        sending the next command: the device drops commands that arrive while it is busy.
        """
        if self._ser is None:
            raise DeviceError("port is not open")
        with self._cond:
            self._reply_buf.clear()
        log.debug("-> %s", cmd)
        self._ser.write(cmd.encode("ascii") + P.LINE_END)
        if expect is None:
            return b""
        target = expect.encode("ascii")
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                idx = self._reply_buf.find(target)
                if idx >= 0:
                    return bytes(self._reply_buf[: idx + len(target)])
                for unknown in P.UNKNOWN_COMMAND_REPLIES:
                    if (u := self._reply_buf.find(unknown)) >= 0:
                        if allow_unknown:
                            return bytes(self._reply_buf[: u + len(unknown)])
                        raise DeviceError(f"device rejected {cmd!r} ({unknown.decode()})")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeviceError(f"no {expect!r} reply to {cmd!r} within {timeout}s")
                self._cond.wait(remaining)

    def _wait_for_quiet(self, quiet_s: float = 0.1, timeout: float = 2.0) -> bool:
        """Wait until no bytes have arrived for ``quiet_s`` (counting from this call).

        Returns False on timeout.
        """
        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._cond:
                idle = (time.monotonic_ns() - max(self._last_rx_ns, start_ns)) / 1e9
            if idle >= quiet_s:
                return True
            time.sleep(quiet_s - idle)
        return False

    def ensure_stopped(self) -> None:
        """Bring the device into the acquisition-stopped state from any state.

        The device powers on streaming and then echoes StopDataAcq. If it is already stopped
        it answers "Unknown Command" instead, slowly enough that a command sent before that
        reply is dropped, so the reply is awaited either way before the stream must go quiet.
        """
        self.send(P.STOP_ACQ, expect=P.STOP_ACQ, timeout=2.0, allow_unknown=True)
        if not self._wait_for_quiet():
            raise DeviceError("device is still streaming after StopDataAcq")
        self.acquiring = False
        assert self._ser is not None
        self._ser.reset_input_buffer()
        with self._cond:
            self._reply_buf.clear()

    def filter_status(self) -> P.FilterStatus:
        """Query filter settings. Only answered while acquisition is stopped."""
        deadline = time.monotonic() + 1.0
        self.send(P.GET_FILTER_STATUS)
        with self._cond:
            while (m := P.FILTER_STATUS_RE.search(self._reply_buf)) is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DeviceError(f"no reply to GetFilterStatus: {bytes(self._reply_buf)!r}")
                self._cond.wait(remaining)
            return P.FilterStatus.parse(m.group(0))

    def configure_filters(
        self, powerline_hz: int, highpass_hz: int, enabled: bool | None = None
    ) -> P.FilterStatus:
        """Apply filter settings (device must be stopped) and return the verified status.

        ``enabled`` switches all built-in filtering on or off; the power-line notch has no
        switch of its own. None leaves it as it is. SetFilterEN is only sent when the state
        has to change, so a device on its power-up default (on) is never sent it.
        """
        for cmd in (P.set_powerline_freq_cmd(powerline_hz), P.set_highpass_freq_cmd(highpass_hz)):
            self.send(cmd, expect=cmd)
        status = self.filter_status()
        if enabled is not None and status.enabled != enabled:
            cmd = P.set_filter_enabled_cmd(enabled)
            self.send(cmd, expect=cmd)
            status = self.filter_status()
        if (status.powerline_hz, status.highpass_hz) != (powerline_hz, highpass_hz) or (
            enabled is not None and status.enabled != enabled
        ):
            raise DeviceError(f"filter settings did not take effect: {status}")
        return status

    def set_timestamp(self, unix_ms: int) -> None:
        cmd = P.set_timestamp_cmd(unix_ms)
        if self.acquiring:
            # While streaming the device applies TIME but sends no echo (observed on hardware;
            # the manual doesn't say). Its clock then jumps to ``unix_ms`` in the IMU packets.
            self.send(cmd)
        else:
            self.send(cmd, expect=cmd)

    def start_acquisition(self) -> None:
        self.send(P.START_ACQ, expect=P.START_ACQ)
        self.acquiring = True

    def stop_acquisition(self) -> None:
        # While streaming, the echo is interleaved with packets that were already in flight.
        self.send(P.STOP_ACQ, expect=P.STOP_ACQ)
        self.acquiring = False
        self._wait_for_quiet()
