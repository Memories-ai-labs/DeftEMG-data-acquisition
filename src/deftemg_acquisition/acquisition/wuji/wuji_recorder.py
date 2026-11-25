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
import mmap
import operator
import os
import threading
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pylsl

from deftemg_acquisition.acquisition.lsl import STREAM_PREFIX, make_outlet
from deftemg_acquisition.acquisition.peripherals.peripherals import Peripheral
from deftemg_acquisition.clock import host_ns

from .calibration import calibrate, use_user

log = logging.getLogger(__name__)

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


CHUNK_FRAMES = 3600  # rows per buffer chunk: 30 s at the glove's 120 Hz


def _int64(value: Any) -> np.int64:
    """A header value as stored in the trial's int64 arrays; raises if it can't be one.

    Converted inside a list, as each value was when the trial's values were kept in lists: a
    numpy scalar out of int64's range raises there (as an array or a bare scalar it would
    wrap)."""
    a = np.asarray([value], dtype=np.int64)
    if a.shape != (1,):
        raise ValueError(f"not a single integer: {value!r}")
    return a[0]


def _empty_chunk(shape: tuple[int, ...], dtype: type) -> np.ndarray:
    """An uninitialised array in its own anonymous memory map: its pages are taken only as
    rows are written, and go back to the system as soon as the array is freed (memory freed
    by ``malloc`` may stay with the process, so copying the chunks out at ``stop`` would hold
    the data twice)."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if not nbytes:
        return np.empty(shape, dtype)
    if os.name == "nt":
        memory = mmap.mmap(-1, nbytes)
    else:  # an anonymous map is shared by default, and would be shared with any child process
        memory = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE)
    return np.frombuffer(memory, dtype).reshape(shape)


def _gather(chunks: list[Any], keep: np.ndarray, n_keep: int, chunk_frames: int) -> np.ndarray:
    """The kept rows of one array's chunks, as one array of exactly ``n_keep`` rows. Each chunk
    is freed (its list entry set to None) once copied, so the copy needs about one chunk more
    than the data."""
    out = np.empty((n_keep, *chunks[0].shape[1:]), chunks[0].dtype)
    pos = 0
    for j in range(len(chunks)):
        rows = keep[j * chunk_frames : (j + 1) * chunk_frames]
        k = int(np.count_nonzero(rows))
        if k == len(rows):
            out[pos : pos + k] = chunks[j][:k]
        elif k:
            np.compress(rows, chunks[j][: len(rows)], axis=0, out=out[pos : pos + k])
        pos += k
        chunks[j] = None
    return out


class StreamBuffer:
    """One stream's frames during a trial. ``add`` runs on the SDK's callback thread.

    Each array (host times, header times and sequence numbers, and every field) is kept in
    chunks of ``chunk_frames`` rows, allocated as needed (``_empty_chunk``) and filled in
    place, so a trial holds about its data's size in memory. ``stop`` copies the kept rows into
    arrays of their exact size, freeing each chunk as it goes, without holding up ``add``.

    A frame whose shape differs from the trial's first frame fails the stream for that trial:
    its frames are freed, later ones are not kept, and ``stop`` raises. A frame that can't be
    converted, whose header ``seq`` or ``timestamp_us`` is not an int64, or that can't be
    stored (no memory for a chunk; a host time that is not an int64), is dropped and counted in
    ``n_errors``."""

    def __init__(
        self,
        name: str,
        convert: Callable[[Any], dict[str, Any]],
        topic: str | None = None,
        chunk_frames: int = CHUNK_FRAMES,
    ):
        chunk_frames = operator.index(chunk_frames)  # a whole number of rows, not 4.0
        if chunk_frames < 1:
            raise ValueError(f"chunk_frames must be at least 1, not {chunk_frames}")
        self.name = name
        self.topic = topic or name  # the STREAMS name, e.g. "tactile" for "left_tactile"
        self.convert = convert
        self.chunk_frames = chunk_frames
        self._lock = threading.Lock()
        self._recording = False
        self._start_ns = 0
        self._clear()
        self.n_received = 0  # whole session
        self.n_errors = 0  # frames that could not be converted, and SDK errors
        self._logged: set[str] = set()  # the kinds of dropped frame logged (the first of each)
        self.last_ns = 0  # host time of the last frame, in a trial or not (0: none yet)

    def _clear(self) -> None:
        # chunks of each array; a row counts once all of its arrays are written (_n)
        self._host: list[np.ndarray] = []
        self._dev: list[np.ndarray] = []
        self._seq: list[np.ndarray] = []
        self._fields: dict[str, list[np.ndarray]] = {}  # in the trial's first frame's order
        self._n = 0
        self._shapes: dict[str, tuple[int, ...]] | None = None  # the trial's first frame
        self._failed: str | None = None

    def start(self, host_start_ns: int) -> None:
        with self._lock:
            self._clear()
            self._start_ns = host_start_ns
            self._recording = True

    def _drop(self, what: str) -> None:
        self.n_errors += 1
        if what not in self._logged:
            self._logged.add(what)
            log.exception("glove %s: could not %s", self.name, what)

    def add(self, frame: Any, t_ns: int) -> None:
        with self._lock:
            self.n_received += 1
            self.last_ns = t_ns
            if not self._recording or self._failed:
                return
            try:
                fields = self.convert(frame)
                values = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
            except Exception:
                self._drop("convert a frame")
                return
            shapes = {key: array.shape for key, array in values.items()}
            if self._shapes is None:
                self._shapes = shapes
            elif shapes != self._shapes:
                failed = f"a frame of shape {shapes}, the trial's first frame {self._shapes}"
                log.error("glove %s: %s; the stream fails this trial", self.name, failed)
                self._clear()
                self._failed = failed
                return
            try:  # before any write: a bad header drops the frame, not the trial
                header = frame.header
                dev, seq = _int64(header.timestamp_us), _int64(header.seq)
            except Exception:
                self._drop("read a frame's header")
                return
            try:  # a row counts only once all of its arrays are written
                chunk, row = divmod(self._n, self.chunk_frames)
                if chunk == len(self._host):  # allocate every array's next chunk, then keep them
                    rows = self.chunk_frames
                    host_chunk, dev_chunk, seq_chunk = (
                        _empty_chunk((rows,), np.int64) for _ in range(3)
                    )
                    field_chunks = {
                        key: _empty_chunk((rows, *shape), np.float32)
                        for key, shape in self._shapes.items()
                    }
                    self._host.append(host_chunk)
                    self._dev.append(dev_chunk)
                    self._seq.append(seq_chunk)
                    for key, array in field_chunks.items():
                        self._fields.setdefault(key, []).append(array)
                self._host[chunk][row] = t_ns
                self._dev[chunk][row] = dev
                self._seq[chunk][row] = seq
                for key, array in values.items():
                    self._fields[key][chunk][row] = array
            except Exception:
                self._drop("store a frame")
                return
            self._n += 1  # the row is complete

    def error(self, message: str) -> None:
        with self._lock:
            self.n_errors += 1
        log.warning("glove %s: %s", self.name, message)

    def stop(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """End the trial: arrays of the frames that arrived in [start, end] and a summary."""
        with self._lock:  # hand the chunks over; the copy runs without the lock
            self._recording = False
            failed, n, start_ns, n_errors = self._failed, self._n, self._start_ns, self.n_errors
            host_chunks = list(self._host)  # which rows to keep, whatever the fields are named
            columns: dict[str, list[Any]] = {
                "host_ns": self._host,
                "dev_us": self._dev,
                "seq": self._seq,
            }
            if n:  # (a field's chunks may exist with no rows: a frame that failed to store)
                columns.update(self._fields)
            self._clear()
        if failed:
            raise ValueError(failed)
        host = np.concatenate(host_chunks)[:n] if n else np.empty(0, np.int64)
        del host_chunks
        return self._arrays(columns, host, n, start_ns, host_end_ns, n_errors)

    def _arrays(
        self,
        columns: dict[str, list[Any]],
        host: np.ndarray,
        n: int,
        start_ns: int,
        host_end_ns: int,
        n_errors: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        keep = (host >= start_ns) & (host <= host_end_ns)
        n_keep = int(np.count_nonzero(keep))
        del host
        arrays = {}
        for key, chunks in columns.items():
            arrays[f"{self.name}_{key}"] = (
                _gather(chunks, keep, n_keep, self.chunk_frames) if n else np.empty(0, np.int64)
            )
        span_s = (host_end_ns - start_ns) / 1e9
        gaps = np.diff(arrays[f"{self.name}_host_ns"]) / 1e9
        seq = arrays[f"{self.name}_seq"]
        summary = {
            "n_frames": n_keep,
            "rate_hz": round(n_keep / span_s, 1) if span_s > 0 else 0.0,
            "max_gap_s": round(float(gaps.max()), 4) if len(gaps) else None,
            "n_seq_skipped": int(np.clip(np.diff(seq) - 1, 0, None).sum()) if n_keep > 1 else 0,
            "n_errors": n_errors,
        }
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

    name = "wuji_glove"

    def __init__(self, sn: str | Iterable[str] | None = None):
        self.sn = (sn,) if isinstance(sn, str) else tuple(sn) if sn is not None else None
        self.gloves: list[Any] = []
        self.buffers: dict[tuple[str, str], StreamBuffer] = {}  # by (sn, topic)
        self.subscriptions: list[Any] = []
        self.calibration: dict[str, Any] = {}
        self.hand_models: dict[str, tuple[str, str]] = {}
        self.manager = None  # set when this recorder made the SDK connection
        self.lsl_publisher = LiveGlove(source_id=f"{self.name}_{self.sn}")

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
        self.gloves = []
        try:
            for device in devices:
                self.gloves.append(
                    manager.connect(sn=device.sn, device_name=f"{DEVICE_NAME}_{device.sn}")
                )
        except BaseException:  # a later glove failed: disconnect the ones already connected
            self.close()
            raise
        log.info("connected wuji glove(s): %s", ", ".join(d.sn for d in devices))
        return self.gloves

    def check_streams(self, max_silence_s: float = 1.0) -> list[str]:
        """Streams with no frame for ``max_silence_s`` s, or none since subscribing."""
        now = host_ns()
        return [
            buffer.name
            for buffer in self.buffers.values()
            if not buffer.last_ns or (now - buffer.last_ns) / 1e9 > max_silence_s
        ]

    def status(self) -> str | None:
        """The gloves whose streams are silent (``check_streams``) or whose SDK subscription
        has stopped, e.g. ``WG1…: no data`` (every stream) or ``WG1…: no tactile, tip_poses``."""
        silent = set(self.check_streams())
        missing: dict[str, list[str]] = {}
        for i, ((sn, topic), buf) in enumerate(self.buffers.items()):
            sub = self.subscriptions[i] if i < len(self.subscriptions) else None
            if buf.name in silent or not getattr(sub, "is_active", True):
                missing.setdefault(sn, []).append(topic)
        parts = []
        for sn, topics in missing.items():
            n = sum(1 for s, _ in self.buffers if s == sn)
            parts.append(f"{sn}: no data" if len(topics) == n else f"{sn}: no {', '.join(topics)}")
        return "; ".join(parts) or None

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
                sn, side = glove.serial_number, glove.hand_side().get()
                self.calibration[sn] = calibrate(glove, side)
                urdf = self.calibration[sn].get("calibrated_urdf")
                if urdf:
                    name = f"{self.name}_{sn}_{side}_hand.urdf"
                    self.hand_models[sn] = (name, Path(urdf).read_text())
            self.subscribe()
            info = {}
            for glove in self.gloves:
                sn = glove.serial_number
                info[sn] = {
                    "hand_side": glove.hand_side().get(),
                    "firmware": glove.version().get(),
                    "streams": [t for s, t in self.buffers if s == sn],
                    "calibration": self.calibration[sn],
                }
                if sn in self.hand_models:
                    info[sn]["hand_model_file"] = self.hand_models[sn][0]
            return info
        except BaseException:  # also Ctrl+C during the calibration: disconnect the gloves
            self.close()
            raise

    def save_files(self, session_dir: Path) -> None:
        """Each glove's calibrated hand model, as ``<name>_<sn>_<side>_hand.urdf``."""
        for name, urdf in self.hand_models.values():
            (session_dir / name).write_text(urdf)

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
            try:
                a, summary = buf.stop(host_end_ns)
            except Exception as e:
                log.exception("glove %s: stopping the stream failed", buf.name)
                streams[buf.name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                continue
            arrays.update(a)
            if summary["n_frames"] == 0:
                summary = {"ok": False, "error": "no frames in this trial", **summary}
                log.warning("glove %s: no frames in this trial", buf.name)
            streams[buf.name] = summary

        self.acquiring = False
        failed = sorted(name for name, s in streams.items() if s.get("ok") is False)

        metadata: dict[str, Any] = {"streams": streams}
        if failed:
            metadata["ok"] = False
            metadata["error"] = f"streams failed: {', '.join(failed)}"
        return arrays, metadata
