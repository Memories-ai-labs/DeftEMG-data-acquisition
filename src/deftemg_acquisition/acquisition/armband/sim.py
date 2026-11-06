"""A simulated armband serial port, for tests and for trying the pipeline without hardware.

It follows the command rules in the manual: powers on streaming; while streaming it only
answers StopDataAcq and TIME...; while stopped it answers StopDataAcq (and anything invalid)
with "Unknown Command", and applies TIME... while streaming without echoing it, as observed
on real hardware.
With the filter off (``SetFilterEN-0``) it sends the 59-byte raw format, with a 60 Hz
power-line hum added to the EMG.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import protocol as P


class SimulatedArmband:
    """Drop-in for ``serial.Serial``: ``Armband("sim", ser=SimulatedArmband())``."""

    # On real hardware an "Unknown Command" reply takes a while, and commands that arrive in
    # the meantime are silently dropped.
    UNKNOWN_REPLY_DELAY_S = 0.05
    HUM_UV = 30.0  # power-line hum amplitude while the filter is off

    def __init__(self, seed: int = 0, imu_every: int = 10, streaming: bool = True):
        self.lock = threading.Lock()
        self.out = bytearray()
        self.streaming = streaming
        self.seq = 0
        self.filters = [1, 50, 60]
        self.commands: list[str] = []  # commands the device acted on (dropped ones excluded)
        self.dropped_commands: list[str] = []
        self._pending_reply: tuple[float, bytes] | None = None
        self.imu_every = imu_every  # one IMU packet per this many EMG packets (~200 Hz)
        self._rng = np.random.default_rng(seed)
        self._n_emg = 0
        self._last = time.monotonic()
        self._clock_base_ms = 0  # device clock = base + time since it was set
        self._clock_set_at = time.monotonic()
        self._stall_until = 0.0

    def stall(self, seconds: float) -> None:
        """Send nothing for ``seconds``, like a sleeping host: the sequence counter pauses while
        the device clock keeps running (as observed on hardware)."""
        with self.lock:
            self._stall_until = time.monotonic() + seconds

    @property
    def in_waiting(self) -> int:
        return len(self.out)

    def _device_ms(self) -> int:
        return self._clock_base_ms + int((time.monotonic() - self._clock_set_at) * 1000)

    def _emg_payloads(self, n: int) -> np.ndarray:
        """EMG payloads of the next ``n`` packets, one row of bytes each."""
        t = (self._n_emg + np.arange(n)) / P.EMG_RATE_HZ
        # Background noise plus a ~1 s contraction every 3 s, centred on a different channel
        # each time so it moves around the band.
        phase = t % 3.0
        burst = np.sin(np.pi * np.clip(phase - 1.0, 0, 1))[:, None] ** 2
        centre = (5 * (t // 3.0) % P.N_EMG_CHANNELS)[:, None]
        ch = np.arange(P.N_EMG_CHANNELS)[None, :]
        spread = np.exp(-0.5 * ((ch - centre) / 1.5) ** 2)
        uv = self._rng.normal(size=(n, P.N_EMG_CHANNELS)) * (3 + 40 * burst * spread)
        if self.filters[0]:
            return np.clip(uv * 7.5, -32768, 32767).astype(">i2").view(np.uint8).reshape(n, -1)
        uv += self.HUM_UV * np.sin(2 * np.pi * 60 * t)[:, None]  # what the notch removes
        counts = np.clip(uv * 60, -(2**23), 2**23 - 1).astype(">i4")
        return counts.view(np.uint8).reshape(n, -1, 4)[:, :, 1:].reshape(n, -1)  # 24-bit

    def _imu_packet(self) -> bytes:
        t = self._n_emg / P.EMG_RATE_HZ
        gyro = np.array([np.sin(t), np.cos(0.7 * t), 0.2 * np.sin(2 * t)]) / P.GYRO_RAD_S_PER_COUNT
        acc = np.array([0.5 * np.sin(0.3 * t), 0.5 * np.cos(0.3 * t), 9.81]) / P.ACC_M_S2_PER_COUNT
        words = np.concatenate([[345], gyro, acc, [4012]]).astype(">i2").tobytes()
        head = P.SYNC + bytes([P.TYPE_IMU, self.seq & 0xFF]) + words
        head += bytes([82]) + self._device_ms().to_bytes(8, "big")
        return head + bytes(P.packet_len(bool(self.filters[0])) - len(head))

    def _generate(self) -> None:
        now = time.monotonic()
        if not self.streaming or now < self._stall_until:
            self._last = now
            return
        n = int((now - self._last) * P.EMG_RATE_HZ)
        if n == 0:
            return
        self._last += n / P.EMG_RATE_HZ
        for row in self._emg_payloads(n):
            self.out += P.SYNC + bytes([P.TYPE_EMG, self.seq & 0xFF]) + row.tobytes()
            self.seq += 1
            self._n_emg += 1
            if self._n_emg % self.imu_every == 0:
                self.out += self._imu_packet()
                self.seq += 1

    def read(self, size: int = 1) -> bytes:
        time.sleep(0.002)  # USB CDC delivers data in small bursts
        with self.lock:
            self._generate()
            if self._pending_reply and time.monotonic() >= self._pending_reply[0]:
                self.out += self._pending_reply[1]
                self._pending_reply = None
            data = bytes(self.out[:size])
            del self.out[:size]
            return data

    def write(self, data: bytes) -> int:
        cmd = data.decode().strip()
        with self.lock:
            self._generate()
            if self._pending_reply is not None:
                self.dropped_commands.append(cmd)
                return len(data)
            self.commands.append(cmd)
            reply = None
            if cmd.startswith("TIME"):
                self._clock_base_ms, self._clock_set_at = int(cmd[4:], 16), time.monotonic()
                reply = None if self.streaming else cmd  # no echo while streaming
            elif self.streaming:
                if cmd == P.STOP_ACQ:
                    self.streaming, reply = False, cmd
            elif cmd == P.START_ACQ:
                self.streaming, reply = True, cmd
                self._last = time.monotonic()
            elif cmd.startswith("SetFilterEN-"):
                self.filters[0], reply = int(cmd.split("-")[1]), cmd
            elif cmd.startswith("SetPFFilterFreq-"):
                self.filters[1], reply = int(cmd.split("-")[1]), cmd
            elif cmd.startswith("SetHPFilterFreq-"):
                self.filters[2], reply = int(cmd.split("-")[1]), cmd
            elif cmd == P.GET_FILTER_STATUS:
                reply = "-".join(map(str, self.filters))
            else:  # includes StopDataAcq while already stopped, as on the real device
                reply = P.UNKNOWN_COMMAND_REPLY
            if reply == P.UNKNOWN_COMMAND_REPLY:
                due = time.monotonic() + self.UNKNOWN_REPLY_DELAY_S
                self._pending_reply = (due, reply.encode() + P.LINE_END)
            elif reply:
                self.out += reply.encode() + P.LINE_END
        return len(data)

    def reset_input_buffer(self) -> None:
        with self.lock:
            self.out.clear()

    def close(self) -> None:
        pass
