"""Trial-level control of an armband recording session.

This is the piece a cueing / prompt UI should drive: it knows nothing about keyboards.

    with AcquisitionSession(SessionConfig(port="/dev/cu.usbmodem1101")) as session:
        for task in ["rest", "pinch", "grip"]:
            show_cue(task)
            session.start_trial(label=task)
            wait_for_end()
            result = session.end_trial()
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from deftemg_acquisition.analysis.filtering import EmgFilter
from deftemg_acquisition.clock import HOST_CLOCK, anchor, host_ns
from deftemg_acquisition.system import keep_awake

from .device import Armband, DeviceError
from .protocol import STALL_GAP_S, FilterStatus
from .recorder import TrialRecorder

log = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    port: str | None = None  # None: auto-detect the single USB serial port
    powerline_hz: int = 60
    highpass_hz: int = 60
    # False: the armband's built-in filters (notch and high-pass together) off, raw 24-bit EMG
    filter_enabled: bool = True
    # With filter_enabled False, causal filters on the live EMG stream only (recordings stay
    # raw): high-pass in Hz (None: off) and a notch from live.LIVE_NOTCHES ("none", "50", "60",
    # "comb50", "comb60", "comb50all", "comb60all"); None: powerline_hz and every harmonic
    # ("comb60all" / "comb50all"), as filter-emg filters recordings by default.
    live_highpass_hz: float | None = 20.0
    live_notch: str | None = None
    data_dir: Path = Path("data/raw")
    # True: stream from session open to close and save only inside trials, so the live view
    # works between trials. False: StartDataAcq/StopDataAcq around every trial.
    continuous: bool = True
    live_stream: bool = True  # publish over LSL for the viewer (see live.py)
    session_name: str | None = None  # default: the start time, YYYYMMDD-HHMMSS
    metadata: dict[str, Any] = field(default_factory=dict)  # subject id, arm, notes, ...


@dataclass
class TrialResult:
    index: int
    label: str | None
    device_timestamp_ms: int  # value sent with TIME... (wall-clock Unix ms) at trial start
    host_start_ns: int  # host clock (see clock.py) when recording of this trial began
    host_end_ns: int  # host clock when recording of this trial ended
    n_emg: int
    n_imu: int
    n_dropped: int
    raw_path: str
    npz_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    n_stalls: int = 0  # times no data arrived for more than STALL_GAP_S
    stalled_s: float = 0.0  # total time without data
    host_clock: str = HOST_CLOCK
    # (wall_ns, host_ns) pairs at trial start (when TIME was set) and end; see clock.py
    anchor_start: dict[str, int] = field(default_factory=dict)
    anchor_end: dict[str, int] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return (self.host_end_ns - self.host_start_ns) / 1e9


class AcquisitionSession:
    """Owns one armband connection and records numbered trials into a session folder.

    Opening the session stops any ongoing acquisition and configures the built-in filters.
    Each trial sets the device clock to host Unix ms, records, then saves
    ``trial_NNN[_label].bin/.npz`` and appends a line to ``trials.jsonl``.

    In continuous mode (default) the device streams from :meth:`open` to :meth:`close` and a
    trial only selects which bytes are saved. Otherwise each trial sends StartDataAcq and
    StopDataAcq. If ``live_stream`` is set, all data and trial markers also go out over LSL.
    """

    def __init__(self, config: SessionConfig, device: Armband | None = None):
        self.config = config
        self.device = device or Armband(config.port)
        name = config.session_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.session_dir = Path(config.data_dir) / name
        self.filter_status: FilterStatus | None = None
        self.trials: list[TrialResult] = []
        self._recorder: TrialRecorder | None = None
        self._pending: dict[str, Any] = {}
        self.live = None
        self._caffeinate: subprocess.Popen | None = None

    # --- lifecycle ---------------------------------------------------------------------------

    def open(self) -> None:
        # USB is suspended while the computer sleeps, and the armband then stops sending.
        self._caffeinate = keep_awake()
        self.device.open()
        try:
            self.device.ensure_stopped()
            self.filter_status = self.device.configure_filters(
                self.config.powerline_hz, self.config.highpass_hz, self.config.filter_enabled
            )
            if self.config.live_stream:
                # imported lazily: pulls in liblsl
                from .live import LiveStreamer, describe_filter, live_filter_config

                live = self.live_emg_filter()  # None: the armband has filtered already
                emg_filter = None
                if live is not None:
                    emg_filter = live_filter_config(live.highpass_hz, live.notch)
                    log.info("live EMG stream filters: %s", describe_filter(emg_filter))
                    if emg_filter.notch_hz not in (None, self.config.powerline_hz):
                        log.warning(
                            "live notch at %g Hz, but powerline_hz is %s: check the mains "
                            "frequency",
                            emg_filter.notch_hz,
                            self.config.powerline_hz,
                        )
                self.live = LiveStreamer(
                    f"waveletech:{self.device.port}",
                    self.filter_status.enabled,
                    emg_filter=emg_filter,
                )
                self.live.start()
                self.device.attach_sink("live", self.live)
            if self.config.continuous:
                self.device.set_timestamp(anchor()["wall_ns"] // 1_000_000)
                self.device.start_acquisition()
        except Exception:
            self._shutdown()
            raise
        log.info("armband ready on %s: %s", self.device.port, self.filter_status)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "port": self.device.port,
            "opened_at": datetime.now().isoformat(),
            "filter_status": asdict(self.filter_status),
            "continuous": self.config.continuous,
            "host_clock": HOST_CLOCK,
            "clock_anchor": anchor(),
            "metadata": self.config.metadata,
        }
        (self.session_dir / "session.json").write_text(json.dumps(info, indent=2))

    def live_emg_filter(self) -> EmgFilter | None:
        """The causal filter on live EMG (stream, viewer, effort meter), or None when the
        armband filters already (``filter_enabled``) or before :meth:`open`."""
        if self.filter_status is None or self.filter_status.enabled:
            return None
        notch = self.config.live_notch or f"comb{self.config.powerline_hz}all"
        return EmgFilter(self.config.live_highpass_hz, notch, zero_phase=False)

    def close(self) -> None:
        try:
            if self.in_trial:
                self.end_trial()
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        try:
            self.device.close()  # stops acquisition if running
        finally:
            if self.live is not None:
                self.live.stop()
                self.live = None
            if self._caffeinate is not None:
                self._caffeinate.terminate()
                self._caffeinate = None

    def __enter__(self) -> AcquisitionSession:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- trials ------------------------------------------------------------------------------

    @property
    def in_trial(self) -> bool:
        return self._recorder is not None

    @property
    def next_index(self) -> int:
        return len(self.trials) + 1

    def start_trial(self, label: str | None = None, **metadata: Any) -> int:
        """Sync the device clock to the host, start streaming, and begin recording.

        Returns the trial index. ``metadata`` is stored with the trial (e.g. cue onset time).
        """
        if self.in_trial:
            raise DeviceError("a trial is already running")
        recorder = TrialRecorder()
        clock_anchor = anchor()  # the device clock is set from this wall-clock reading
        device_ts_ms = clock_anchor["wall_ns"] // 1_000_000
        self.device.set_timestamp(device_ts_ms)  # accepted while streaming, too
        self.device.attach_sink("trial", recorder)
        host_start_ns = host_ns()
        if not self.config.continuous:
            try:
                self.device.start_acquisition()
            except Exception:
                self.device.detach_sink("trial")
                raise
        self._recorder = recorder
        self._pending = {
            "index": self.next_index,
            "label": label,
            "device_timestamp_ms": device_ts_ms,
            "host_start_ns": host_start_ns,
            "anchor_start": clock_anchor,
            "metadata": metadata,
        }
        if self.live is not None:
            self.live.push_marker(f"trial_start|{self.next_index}|{label or ''}")
        log.info("trial %d started (%s)", self.next_index, label or "unlabelled")
        return self.next_index

    def push_marker(self, marker: str) -> None:
        """Send an event marker over LSL (no-op without live streaming)."""
        if self.live is not None:
            self.live.push_marker(marker)

    def end_trial(self, arrays: dict[str, Any] | None = None, **metadata: Any) -> TrialResult:
        """Stop recording, save the trial, and return its summary.

        ``arrays`` are saved into the trial's ``.npz`` (e.g. prompt timings); ``metadata`` is
        stored in ``trials.jsonl`` alongside what was given to :meth:`start_trial`.
        """
        if self._recorder is None:
            raise DeviceError("no trial is running")
        recorder, pending = self._recorder, self._pending
        try:
            if not self.config.continuous:
                self.device.stop_acquisition()
        finally:
            host_end_ns = host_ns()
            anchor_end = anchor()
            self.device.detach_sink("trial")
            self._recorder = None
            self._pending = {}
            if self.live is not None:
                self.live.push_marker(f"trial_end|{pending['index']}")

        pending["metadata"] = {**pending["metadata"], **metadata}
        stem = self.session_dir / f"trial_{pending['index']:03d}"
        if pending["label"]:
            stem = stem.with_name(f"{stem.name}_{_slug(pending['label'])}")
        enabled = self.filter_status.enabled if self.filter_status else True
        files = recorder.save(
            stem,
            filter_enabled=enabled,
            extra={
                "device_timestamp_ms": pending["device_timestamp_ms"],
                "host_start_ns": pending["host_start_ns"],
                "host_end_ns": host_end_ns,
                "host_clock": HOST_CLOCK,
                "anchor_wall_ns": pending["anchor_start"]["wall_ns"],
                "anchor_host_ns": pending["anchor_start"]["host_ns"],
                "anchor_end_wall_ns": anchor_end["wall_ns"],
                "anchor_end_host_ns": anchor_end["host_ns"],
                **(arrays or {}),
            },
        )
        decoded = files.data
        gaps_s = np.diff(decoded["emg_host_time_ns"]) / 1e9
        stalls = gaps_s[gaps_s > STALL_GAP_S]
        result = TrialResult(
            anchor_end=anchor_end,
            n_stalls=len(stalls),
            stalled_s=round(float(stalls.sum()), 3),
            host_end_ns=host_end_ns,
            n_emg=len(decoded["emg_seq"]),
            n_imu=len(decoded["imu_seq"]),
            n_dropped=int(decoded["n_dropped"]),
            raw_path=str(files.raw),
            npz_path=str(files.npz),
            **pending,
        )
        self.trials.append(result)
        with open(self.session_dir / "trials.jsonl", "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        log.info(
            "trial %d saved: %d EMG / %d IMU packets, %d dropped, %.2fs",
            result.index,
            result.n_emg,
            result.n_imu,
            result.n_dropped,
            result.duration_s,
        )
        if result.n_stalls:
            log.warning(
                "trial %d: data stalled %d times (%.1f s in total); see check-timing",
                result.index,
                result.n_stalls,
                result.stalled_s,
            )
        return result


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")
