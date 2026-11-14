"""Record a Wuji Glove alongside the armband, as a session source (``acquisition.sources``).

The glove streams through ``wuji-sdk`` (its own USB discovery, decoding and threads). Each
subscribed stream's callback stamps the frame with :func:`host_ns` on arrival, the armband's
clock, and keeps it while a trial is running. At the end of a trial the frames become arrays
in the trial ``.npz`` (the session prefixes them with the source name, ``glove_``)::

    glove_<stream>_host_ns   (n,)   host clock when the frame arrived (as emg_host_time_ns)
    glove_<stream>_dev_us    (n,)   the frame header's timestamp_us (glove clock)
    glove_<stream>_seq       (n,)   the frame header's sequence number
    glove_<stream>_<field>   (n, …) the frame's values, e.g. glove_tactile_data

Streams are named after the SDK's ``WujiGlove`` resources (``STREAMS``).

With ``live=True`` each stream is also published over LSL as it arrives, whether or not a
trial is running (``LiveGlove``): one stream per glove stream, type ``Glove_<stream>``, named
``Waveletech glove <stream>``, each frame one sample of its fields flattened in order (e.g.
``hand_joint_angles``: 25 angles, then 5 confidences), stamped with its arrival time on the LSL
clock like the armband's samples. The live viewer shows them (``viz.glove``). Live, the
recorder also subscribes to the ``LIVE_STREAMS`` the glove offers, for the viewer only.
"""

import logging
import os
import threading
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import numpy as np
import pylsl

from deftemg_acquisition.acquisition.lsl import STREAM_PREFIX, make_outlet
from deftemg_acquisition.acquisition.peripherals.peripherals import Peripheral
from deftemg_acquisition.clock import host_ns

from .calibration import calibrate, use_user

log = logging.getLogger(__name__)

SOURCE_NAME = "glove"
DEVICE_NAME = "deftemg_glove"  # local alias for the SDK connection


def _disable_sdk_telemetry() -> None:
    """Stop wuji-sdk collecting or uploading telemetry. It reads these when first imported."""
    os.environ["WUJI_TELEMETRY_DISABLE"] = "1"  # no collection, no local storage, no upload
    os.environ["WUJI_TELEMETRY_NO_UPLOAD"] = "1"
    os.environ["WUJI_DEVICE_TELEMETRY_DISABLE"] = "1"


def _quat(q) -> list[float]:
    return [q.x, q.y, q.z, q.w]


def _vec(v) -> list[float]:
    return [v.x, v.y, v.z]


def _data(frame) -> dict[str, Any]:
    return {"data": frame.data}


def _zones(frame) -> dict[str, Any]:
    return {z: getattr(frame, z) for z in ("palm", "thumb", "index", "middle", "ring", "pinky")}


def _joint_angles(frame) -> dict[str, Any]:
    # firmware order, fingers thumb..pinky, each finger's angles concatenated
    return {
        "angles": [a for finger in frame.fingers for a in finger.angles],
        "confidence": [finger.confidence for finger in frame.fingers],
    }


def _poses(items, pose=lambda x: x.pose) -> dict[str, Any]:
    return {
        "position": [pose(x).position for x in items],
        "orientation": [_quat(pose(x).orientation) for x in items],
        "confidence": [x.confidence for x in items],
    }


def _imu(frame) -> dict[str, Any]:
    return {
        "orientation": _quat(frame.orientation),
        "gyro": _vec(frame.angular_velocity),
        "acc": _vec(frame.linear_acceleration),
    }


CLOUD_POINTS = 549  # taxels in the glove's tactile point cloud


def _point_cloud(frame) -> dict[str, Any]:
    """Taxel positions (m, wrist frame) and pressures (-1: no reading), ``CLOUD_POINTS`` of
    them (NaN pads a shorter cloud). Each point starts with float32 x, y, z, pressure."""
    stride = frame.point_stride // 4
    pts = np.frombuffer(bytes(frame.data), np.float32).reshape(-1, stride)[:CLOUD_POINTS, :4]
    out = np.full((CLOUD_POINTS, 4), np.nan, np.float32)
    out[: len(pts)] = pts
    return {"position": out[:, :3], "pressure": out[:, 3]}


# WujiGlove resource name -> converter of its frames
STREAMS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "tactile": _data,  # calibrated tactile values
    "tactile_binary": _data,  # 24x31 contact grid (after tactile calibration)
    "tactile_residual": _data,  # 24x31 signed contact signal (after tactile calibration)
    "tactile_zones": _zones,
    "hand_joint_angles": _joint_angles,
    "hand_skeleton": lambda f: _poses(f.joints),  # MediaPipe-order joints
    "emf_poses": lambda f: _poses(f.poses),  # raw EMF tracking poses
    "tip_poses": lambda f: _poses(f.poses),  # fingertip poses, wrist frame, thumb..pinky
    "tactile_point_cloud": _point_cloud,  # taxels in 3D (wrist frame) with their pressure
    **{f"imu_data_{part}": _imu for part in ("palm", "thumb", "index", "middle", "ring", "pinky")},
}

TOPICS = (
    "tactile",
    "hand_joint_angles",
    "emf_poses",
    "tactile_zones",
    "tactile_point_cloud",
    "tip_poses",
)


class StreamBuffer:
    """One stream's frames during a trial. ``add`` runs on the SDK's callback thread."""

    def __init__(
        self, name: str, convert: Callable[[Any], dict[str, Any]], topic: str | None = None
    ):
        self.name = name
        self.topic = topic or name  # the STREAMS name, e.g. "tactile" for "left_tactile"
        self.convert = convert
        self._lock = threading.Lock()
        self._recording = False
        self._start_ns = 0
        self._clear()
        self.n_received = 0  # whole session
        self.n_errors = 0  # frames that could not be converted, and SDK errors

    def _clear(self) -> None:
        self._host: list[int] = []
        self._dev: list[int] = []
        self._seq: list[int] = []
        self._fields: dict[str, list] = {}

    def start(self, host_start_ns: int) -> None:
        with self._lock:
            self._clear()
            self._start_ns = host_start_ns
            self._recording = True

    def add(self, frame: Any, t_ns: int) -> None:
        with self._lock:
            self.n_received += 1
            if not self._recording:
                return
            try:
                fields = self.convert(frame)
            except Exception:
                self.n_errors += 1
                if self.n_errors == 1:
                    log.exception("glove %s: could not convert a frame", self.name)
                return
            header = frame.header
            self._host.append(t_ns)
            self._dev.append(header.timestamp_us)
            self._seq.append(header.seq)
            for key, value in fields.items():
                self._fields.setdefault(key, []).append(value)

    def error(self, message: str) -> None:
        with self._lock:
            self.n_errors += 1
        log.warning("glove %s: %s", self.name, message)

    def stop(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """End the trial: arrays of the frames that arrived in [start, end] and a summary."""
        with self._lock:
            self._recording = False
            host = np.asarray(self._host, dtype=np.int64)
            keep = (host >= self._start_ns) & (host <= host_end_ns)
            arrays = {
                f"{self.name}_host_ns": host[keep],
                f"{self.name}_dev_us": np.asarray(self._dev, dtype=np.int64)[keep],
                f"{self.name}_seq": np.asarray(self._seq, dtype=np.int64)[keep],
            }
            for key, values in self._fields.items():
                arrays[f"{self.name}_{key}"] = np.asarray(values, dtype=np.float32)[keep]
            n = int(keep.sum())
            span_s = (host_end_ns - self._start_ns) / 1e9
            gaps = np.diff(arrays[f"{self.name}_host_ns"]) / 1e9
            seq = arrays[f"{self.name}_seq"]
            summary = {
                "n_frames": n,
                "rate_hz": round(n / span_s, 1) if span_s > 0 else 0.0,
                "max_gap_s": round(float(gaps.max()), 4) if len(gaps) else None,
                "n_seq_skipped": int(np.clip(np.diff(seq) - 1, 0, None).sum()) if n > 1 else 0,
                "n_errors": self.n_errors,
            }
            self._clear()
            return arrays, summary


LIVE_STREAM_TYPE = "Glove"


def live_stream_type(stream: str) -> str:
    """LSL type of a glove stream, e.g. ``Glove_tactile``."""
    return f"{LIVE_STREAM_TYPE}_{stream}"


def flatten_fields(fields: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """A frame's converted fields as one row: channel labels and float32 values."""
    labels: list[str] = []
    parts = []
    for key, value in fields.items():
        a = np.asarray(value, dtype=np.float32).ravel()
        labels += [key] if np.ndim(value) == 0 else [f"{key}_{i}" for i in range(a.size)]
        parts.append(a)
    return labels, np.concatenate(parts) if parts else np.zeros(0, np.float32)


class LiveGlove:
    """Publishes glove frames over LSL, one outlet per stream, made at its first frame (when
    its size is known). Called on the SDK's callback threads; a failure never stops recording."""

    def __init__(self, source_id: str):
        self.source_id = source_id
        self._outlets: dict[str, pylsl.StreamOutlet] = {}
        self._sizes: dict[str, int] = {}
        self._lsl_offset = pylsl.local_clock() - host_ns() / 1e9
        self.n_errors = 0

    def push(self, buf: StreamBuffer, frame: Any, t_ns: int) -> None:
        try:
            labels, x = flatten_fields(buf.convert(frame))
            outlet = self._outlets.get(buf.name)
            if outlet is None:
                outlet = make_outlet(
                    f"{STREAM_PREFIX} glove {buf.name}", live_stream_type(buf.topic), labels,
                    [""] * len(labels), pylsl.IRREGULAR_RATE, pylsl.cf_float32,
                    self.source_id, manufacturer="Wuji",
                )  # fmt: skip
                self._outlets[buf.name] = outlet
                self._sizes[buf.name] = len(labels)
            if len(x) != self._sizes[buf.name]:
                raise ValueError(f"frame size {len(x)}, stream has {self._sizes[buf.name]}")
            outlet.push_sample(x, t_ns / 1e9 + self._lsl_offset)
        except Exception:
            self.n_errors += 1
            if self.n_errors == 1:
                log.exception("glove %s: could not publish a frame live", buf.name)


class WujiGloveRecorder(Peripheral):
    """

    Wuji Glove Acquisition Session Source.

    Args:
        sn: single or multiple gloves.
    """

    name = SOURCE_NAME

    def __init__(self, sn: str | Iterable[str] | None = None):
        self.sn = (sn,) if isinstance(sn, str) else tuple(sn) if sn is not None else None
        self.gloves: list[Any] = []
        self.buffers: dict[tuple[str, str], StreamBuffer] = {}  # by (sn, topic)
        self.subscriptions: list[Any] = []
        self.calibration: dict[str, Any] = {}
        self.manager = None  # set when this recorder made the SDK connection
        self.lsl_publisher = LiveGlove(source_id=SOURCE_NAME)

    def initialize(self) -> list[Any]:
        _disable_sdk_telemetry()  # before the import: the SDK reads these at import time
        from wuji_sdk import DeviceType, SdkManager  # the optional [wuji] extra: only to connect

        manager = SdkManager.instance()
        use_user(manager)
        devices = [d for d in manager.scan() if d.device_type == DeviceType.WujiGlove]
        if self.sn is not None:
            missing = [sn for sn in self.sn if sn not in {d.sn for d in devices}]
            if missing:
                found = ", ".join(d.sn for d in devices) or "none"
                raise RuntimeError(f"wuji: glove {', '.join(missing)} not found; found: {found}")
            devices = [d for d in devices if d.sn in self.sn]
        if not devices:
            raise RuntimeError("wuji: no glove found")
        self.manager = manager
        self.gloves = [
            manager.connect(sn=d.sn, device_name=f"{DEVICE_NAME}_{d.sn}") for d in devices
        ]
        log.info("connected wuji glove(s): %s", ", ".join(d.sn for d in devices))
        return self.gloves

    def subscribe(self) -> None:
        _disable_sdk_telemetry()  # before the import: the SDK reads these at import time
        from wuji_sdk import WujiException

        self.buffers = {}
        self.subscriptions = []
        try:
            for glove in self.gloves:
                sn = glove.serial_number
                for topic in TOPICS:
                    buf = StreamBuffer(f"{sn}_{topic}", STREAMS[topic], topic)
                    try:
                        sub = getattr(glove, topic)().subscribe_with_callback(
                            callback=partial(self._on_frame, buf), on_error=buf.error
                        )
                    except WujiException as e:
                        log.warning("glove %s: %s not available, skipped (%s)", sn, topic, e)
                        continue
                    self.buffers[(sn, topic)] = buf
                    self.subscriptions.append(sub)
        except Exception:
            for sub in self.subscriptions:
                sub.close()
            self.subscriptions = []
            raise

    def open(self) -> dict[str, Any]:
        self.initialize()
        try:
            for glove in self.gloves:
                side = glove.hand_side().get()
                self.calibration[glove.serial_number] = calibrate(glove, side)
            self.subscribe()
        except BaseException:  # also Ctrl+C during the calibration: disconnect the gloves
            self.close()
            raise
        info = {}
        for glove in self.gloves:
            sn = glove.serial_number
            info[sn] = {
                "hand_side": glove.hand_side().get(),
                "firmware": glove.version().get(),
                "streams": [t for s, t in self.buffers if s == sn],
                "calibration": self.calibration[sn],
            }
        return info

    def close(self) -> None:
        for sub in self.subscriptions:
            try:
                sub.close()
            except Exception:
                log.exception("glove: closing a subscription failed")
        self.subscriptions = []
        if self.manager is not None:
            try:
                for glove in self.gloves:
                    self.manager.disconnect(f"{DEVICE_NAME}_{glove.serial_number}")
            finally:
                self.manager = None
                self.gloves = []

    def _on_frame(self, buf: StreamBuffer, frame: Any) -> None:
        t_ns = host_ns()  # stamp first: the arrival time is the timeline
        buf.add(frame, t_ns)
        self.lsl_publisher.push(buf, frame, t_ns)

    def start_acquisition(self, index: int, host_start_ns: int) -> None:
        for buf in self.buffers.values():
            buf.start(host_start_ns)
        self.acquiring = True

    def stop_acquisition(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays: dict[str, np.ndarray] = {}
        streams: dict[str, Any] = {}
        for buf in self.buffers.values():
            a, summary = buf.stop(host_end_ns)
            arrays.update(a)
            streams[buf.name] = summary
            if summary["n_frames"] == 0:
                log.warning("glove %s: no frames in this trial", buf.name)
        self.acquiring = False
        return arrays, {"streams": streams}
