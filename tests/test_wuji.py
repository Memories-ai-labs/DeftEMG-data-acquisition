"""Wuji Glove as a session source: buffering, converters, failure isolation (no glove needed)."""

import logging
import mmap
import os
import sys
import threading
import time
from collections.abc import Callable
from functools import partial
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pytest

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband
from deftemg_acquisition.acquisition.wuji import StreamBuffer, WujiGloveRecorder, wuji_recorder
from deftemg_acquisition.acquisition.wuji.wuji_recorder import STREAMS
from deftemg_acquisition.clock import host_ns


def _frame(seq, data):
    return NS(header=NS(seq=seq, timestamp_us=1000 * seq, frame_id=""), data=data)


def test_stream_buffer_keeps_only_the_trial_window():
    buf = StreamBuffer("tactile", STREAMS["tactile"])
    buf.add(_frame(0, [0.0, 0.0]), t_ns=50)  # before any trial: not kept
    buf.start(100)
    buf.add(_frame(1, [1.0, 2.0]), t_ns=90)  # arrived before the trial's start time
    buf.add(_frame(2, [3.0, 4.0]), t_ns=150)
    buf.add(_frame(4, [5.0, 6.0]), t_ns=200)
    buf.add(NS(header=NS(seq=5, timestamp_us=0)), t_ns=210)  # no data: counted as an error
    buf.add(_frame(6, [7.0, 8.0]), t_ns=400)  # after the end time
    arrays, summary = buf.stop(300)
    assert arrays["tactile_host_ns"].tolist() == [150, 200]
    assert arrays["tactile_seq"].tolist() == [2, 4]
    assert arrays["tactile_dev_us"].tolist() == [2000, 4000]
    assert arrays["tactile_data"].shape == (2, 2)
    assert summary["n_frames"] == 2 and summary["n_seq_skipped"] == 1
    assert summary["n_errors"] == 1 and buf.n_received == 6
    buf.add(_frame(7, [0.0, 0.0]), t_ns=500)  # between trials: not kept
    assert buf.stop(600)[1]["n_frames"] == 0


def test_a_non_numeric_frame_is_an_error_and_the_columns_stay_aligned():
    buf = StreamBuffer("tactile", STREAMS["tactile"])
    buf.start(0)
    buf.add(_frame(1, [1.0, 2.0]), t_ns=10)
    buf.add(_frame(2, ["nope", "nope"]), t_ns=20)
    buf.add(_frame(3, [3.0, 4.0]), t_ns=30)
    arrays, summary = buf.stop(100)
    assert arrays["tactile_data"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert arrays["tactile_host_ns"].tolist() == [10, 30]
    assert summary["n_errors"] == 1


def test_a_ragged_stream_fails_its_trial_and_the_others_are_kept():
    rec = WujiGloveRecorder()
    ragged = StreamBuffer("A_ragged", STREAMS["tactile"])
    good = StreamBuffer("A_tactile", STREAMS["tactile"])
    rec.buffers = {("A", "ragged"): ragged, ("A", "tactile"): good}  # the failing one first
    rec.start_acquisition(0, 0)
    ragged.add(_frame(1, [1.0, 2.0]), t_ns=10)
    ragged.add(_frame(2, [3.0]), t_ns=20)
    assert ragged._host == [] and ragged._fields == {}  # freed as soon as it fails
    ragged.add(_frame(3, [6.0, 7.0]), t_ns=25)  # not kept once failed
    assert ragged._host == []
    good.add(_frame(1, [4.0, 5.0]), t_ns=10)
    arrays, meta = rec.stop_acquisition(30)
    assert arrays["A_tactile_data"].tolist() == [[4.0, 5.0]]
    assert not any(key.startswith("A_ragged") for key in arrays)
    assert meta["streams"]["A_tactile"]["n_frames"] == 1
    assert meta["streams"]["A_ragged"]["ok"] is False
    assert meta["streams"]["A_ragged"]["error"].startswith("ValueError")
    assert not ragged._recording and not good._recording  # neither keeps recording

    rec.start_acquisition(1, 100)  # the next trial starts afresh
    ragged.add(_frame(4, [8.0]), t_ns=110)
    arrays, meta = rec.stop_acquisition(200)
    assert arrays["A_ragged_data"].tolist() == [[8.0]]


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _Resource:
    def __init__(self, glove, topic):
        self.glove, self.topic = glove, topic

    def subscribe_with_callback(self, callback, on_error):
        self.glove.callbacks[self.topic] = callback
        return NS(close=lambda: None)


class _NotOffered(Exception):
    pass


class _FakeGlove:
    def __init__(self, sn, side, offers):
        self.serial_number, self.side, self.offers, self.callbacks = sn, side, offers, {}

    def hand_side(self):
        return _Value(self.side)

    def version(self):
        return _Value("1.0")

    def __getattr__(self, topic):
        if topic not in STREAMS:
            raise AttributeError(topic)
        if topic not in self.offers:
            raise _NotOffered(topic)
        return lambda: _Resource(self, topic)


@pytest.fixture
def fake_sdk(monkeypatch):
    """A stand-in for ``wuji_sdk``: ``scan`` finds the gloves of ``found`` ({sn: side}), each
    offering the streams of ``offers``."""
    found: dict[str, str] = {}
    offers = set(STREAMS)
    connected: dict[str, _FakeGlove] = {}
    users = [{"user_id": "", "display_name": "Default"}]
    current = [""]

    class SdkManager:
        @staticmethod
        def instance():
            return SdkManager

        @staticmethod
        def scan():
            return [NS(sn=sn, device_type="glove") for sn in found]

        @staticmethod
        def connect(sn, device_name):
            connected[device_name] = _FakeGlove(sn, found[sn], offers)
            return connected[device_name]

        @staticmethod
        def disconnect(device_name):
            del connected[device_name]

        @staticmethod
        def list_users():
            return users

        @staticmethod
        def create_user(display_name):
            users.append({"user_id": f"u{len(users)}", "display_name": display_name})
            return users[-1]

        @staticmethod
        def switch_user(user_id):
            current[0] = user_id

    sdk = NS(SdkManager=SdkManager, DeviceType=NS(WujiGlove="glove"), WujiException=_NotOffered)
    monkeypatch.setitem(sys.modules, "wuji_sdk", sdk)
    monkeypatch.setattr(wuji_recorder, "calibrate", lambda glove, side: {"urdf": f"{side}.urdf"})
    return NS(found=found, offers=offers, connected=connected, users=users, current=current)


def test_glove_sn_selects_among_the_gloves_found(fake_sdk):
    fake_sdk.found.update({"A": "left", "B": "left"})
    fake_sdk.offers.intersection_update({"tactile", "tactile_zones"})
    rec = WujiGloveRecorder(sn="A")
    assert rec.open() == {"A": {"hand_side": "left", "firmware": "1.0",
                                "streams": ["tactile", "tactile_zones"],  # those offered
                                "calibration": {"urdf": "left.urdf"}}}  # fmt: skip
    assert [g.serial_number for g in fake_sdk.connected.values()] == ["A"]
    rec.close()
    assert not fake_sdk.connected


def test_sdk_telemetry_is_disabled_before_the_sdk_is_imported(fake_sdk, monkeypatch):
    for name in (
        "WUJI_TELEMETRY_DISABLE",
        "WUJI_TELEMETRY_NO_UPLOAD",
        "WUJI_DEVICE_TELEMETRY_DISABLE",
    ):
        monkeypatch.delenv(name, raising=False)
    fake_sdk.found.update({"A": "left"})
    rec = WujiGloveRecorder()
    rec.open()
    rec.close()
    for name in (
        "WUJI_TELEMETRY_DISABLE",
        "WUJI_TELEMETRY_NO_UPLOAD",
        "WUJI_DEVICE_TELEMETRY_DISABLE",
    ):
        assert os.environ[name] == "1"


def test_every_session_calibrates_under_one_sdk_user(fake_sdk):
    fake_sdk.found.update({"A": "left"})
    for _ in range(2):
        rec = WujiGloveRecorder()
        rec.open()
        rec.close()
    assert [u["display_name"] for u in fake_sdk.users] == ["Default", "deftemg"]
    assert fake_sdk.current == ["u1"]


def test_the_calibrated_hand_model_is_kept_as_it_was(fake_sdk, monkeypatch, tmp_path):
    urdf = tmp_path / "sdk" / "left_hand.urdf"
    urdf.parent.mkdir()
    urdf.write_text("<robot name='A'/>")
    monkeypatch.setattr(wuji_recorder, "calibrate",
                        lambda glove, side: {"calibrated_urdf": str(urdf)})  # fmt: skip
    fake_sdk.found.update({"A": "left"})
    rec = WujiGloveRecorder()
    info = rec.open()
    assert info["A"]["hand_model_file"] == "wuji_glove_A_left_hand.urdf"
    urdf.write_text("<robot name='next calibration'/>")  # the SDK replaces its file
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    rec.save_files(session_dir)
    assert (session_dir / "wuji_glove_A_left_hand.urdf").read_text() == "<robot name='A'/>"
    rec.close()


def test_a_failed_connection_disconnects_the_glove_already_open(fake_sdk, monkeypatch):
    import wuji_sdk

    fake_sdk.found.update({"A": "left", "B": "left"})
    connect = wuji_sdk.SdkManager.connect

    def fail_the_second(sn, device_name):
        if sn == "B":
            raise RuntimeError("usb")
        return connect(sn, device_name)

    monkeypatch.setattr(wuji_sdk.SdkManager, "connect", fail_the_second)
    with pytest.raises(RuntimeError, match="usb"):
        WujiGloveRecorder().open()
    assert not fake_sdk.connected  # glove A was connected, then disconnected


def test_a_failure_while_reading_the_glove_closes_it(fake_sdk, monkeypatch):
    fake_sdk.found.update({"A": "left"})
    fake_sdk.offers.intersection_update({"tactile"})

    def fail(self):
        raise RuntimeError("firmware")

    monkeypatch.setattr(_FakeGlove, "version", fail)
    with pytest.raises(RuntimeError, match="firmware"):
        WujiGloveRecorder().open()
    assert not fake_sdk.connected


def test_glove_sn_not_found_is_an_error(fake_sdk):
    fake_sdk.found.update({"A": "left"})
    with pytest.raises(RuntimeError, match="glove C not found; found: A"):
        WujiGloveRecorder(sn=["A", "C"]).open()
    assert not fake_sdk.connected


def test_gloves_are_recorded_by_serial_number(fake_sdk):
    fake_sdk.found.update({"A": "left", "B": "left"})  # the same side: kept apart all the same
    fake_sdk.offers.intersection_update({"tactile"})
    rec = WujiGloveRecorder()
    info = rec.open()
    assert [info[sn]["streams"] for sn in "AB"] == [["tactile"], ["tactile"]]
    rec.start_acquisition(0, host_ns())
    for i, glove in enumerate(fake_sdk.connected.values()):
        glove.callbacks["tactile"](_frame(1, [float(i)]))
    arrays, meta = rec.stop_acquisition(host_ns())
    assert arrays["A_tactile_data"].tolist() == [[0.0]]
    assert arrays["B_tactile_data"].tolist() == [[1.0]]
    assert sorted(meta["streams"]) == ["A_tactile", "B_tactile"]
    rec.close()
    assert not fake_sdk.connected


def _session(tmp_path, sources):
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    return AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband()),
                              sources=sources)  # fmt: skip


class _Failing:
    name = "glove"
    closed = False

    def open(self):
        return {}

    def start_acquisition(self, index, host_start_ns):
        raise RuntimeError("start bug")

    def stop_acquisition(self, host_end_ns):
        raise RuntimeError("glove unplugged")

    def close(self):
        self.closed = True


def test_failing_source_never_loses_the_emg(tmp_path):
    src = _Failing()
    with _session(tmp_path, [src]) as session:
        session.start_trial()
        time.sleep(0.2)
        result = session.end_trial()
    assert len(np.load(result.npz_path)["emg_uv"]) > 0
    assert result.sources["glove"] == {"ok": False, "error": "RuntimeError: glove unplugged"}
    assert src.closed


def test_a_stream_with_no_frames_is_not_ok():
    rec = WujiGloveRecorder()
    silent = StreamBuffer("A_silent", STREAMS["tactile"])
    good = StreamBuffer("A_tactile", STREAMS["tactile"])
    rec.buffers = {("A", "silent"): silent, ("A", "tactile"): good}
    rec.start_acquisition(0, 0)
    good.add(_frame(1, [1.0, 2.0]), t_ns=10)
    arrays, meta = rec.stop_acquisition(20)
    assert meta["ok"] is False and meta["error"] == "streams failed: A_silent"
    assert meta["streams"]["A_silent"]["ok"] is False
    assert meta["streams"]["A_silent"]["n_frames"] == 0
    assert "ok" not in meta["streams"]["A_tactile"]
    assert arrays["A_tactile_data"].tolist() == [[1.0, 2.0]]  # still saved
    assert arrays["A_silent_host_ns"].tolist() == []


def test_the_prompter_names_a_source_that_failed(tmp_path):
    import pyqtgraph as pg

    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    seq = Sequence("a", "single", [Prompt(0.1, 0.1, ("thumb",), ("low",))])
    pg.mkQApp()
    with _session(tmp_path, [_Failing()]) as session:
        win = TaskWindow(session, [(tmp_path / "a.json", seq)], tail_s=0.1)
        assert "saved without" not in win.prompter.message
        win.start_block()
        win.end_block(completed=False)
        assert "Last block saved without:\nglove: RuntimeError: glove unplugged" in (
            win.prompter.message
        )
        win.close()


def test_the_glove_status_names_silent_streams():
    rec = WujiGloveRecorder()
    tactile = StreamBuffer("A_tactile", STREAMS["tactile"], "tactile")
    poses = StreamBuffer("A_tip_poses", STREAMS["tactile"], "tip_poses")
    rec.buffers = {("A", "tactile"): tactile, ("A", "tip_poses"): poses}
    assert rec.status() == "A: no data"  # nothing since subscribing
    tactile.add(_frame(1, [1.0]), t_ns=host_ns())  # between trials: still counts
    assert rec.status() == "A: no tip_poses"
    poses.add(_frame(1, [1.0]), t_ns=host_ns())
    assert rec.status() is None
    tactile.last_ns -= int(2e9)  # silent for 2 s
    assert rec.status() == "A: no tactile"
    tactile.add(_frame(2, [1.0]), t_ns=host_ns())
    rec.subscriptions = [NS(is_active=True), NS(is_active=False)]  # the SDK stopped one
    assert rec.status() == "A: no tip_poses"


class _Unwell(_Failing):
    problem: str | None = "no data"

    def start_acquisition(self, index, host_start_ns):
        pass

    def stop_acquisition(self, host_end_ns):
        return {}, {}

    def status(self):
        return self.problem


def test_the_session_collects_the_sources_status(tmp_path):
    class Broken(_Unwell):
        name = "other"

        def status(self):
            raise RuntimeError("bug")

    src = _Unwell()
    with _session(tmp_path, [src, Broken()]) as session:
        assert session.source_status() == {
            "glove": "no data",
            "other": "status check failed: RuntimeError: bug",
        }
        src.problem = None
        assert session.source_status() == {"other": "status check failed: RuntimeError: bug"}


def test_the_prompter_shows_a_source_problem_during_a_block(tmp_path):
    import pyqtgraph as pg

    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    seq = Sequence("a", "single", [Prompt(0.1, 0.1, ("thumb",), ("low",))])
    pg.mkQApp()
    src = _Unwell()
    src.problem = None
    with _session(tmp_path, [src]) as session:
        win = TaskWindow(session, [(tmp_path / "a.json", seq)], tail_s=0.1)
        win.check_sources()
        assert win.prompter.warning == ""
        win.start_block()
        src.problem = "A: no data"  # the glove is unplugged during the block
        win.check_sources()
        assert win.running and win.prompter.warning == "glove: A: no data"
        win.prompter.grab()  # the band is drawn
        src.problem = None
        win.check_sources()
        assert win.prompter.warning == ""
        win.close()


def test_a_source_saves_its_files_with_the_session(tmp_path):
    class Saving(_Failing):
        def save_files(self, session_dir):
            (session_dir / "glove_model.urdf").write_text("<robot/>")

    with _session(tmp_path, [Saving()]) as session:
        path = session.session_dir / "glove_model.urdf"
        assert path.read_text() == "<robot/>"


def test_source_open_failure_aborts_the_session(tmp_path):
    class NoGlove(_Failing):
        def open(self):
            raise RuntimeError("no glove found")

    src = NoGlove()
    with pytest.raises(RuntimeError, match="no glove found"):
        with _session(tmp_path, [src]):
            pass


def test_ctrl_c_while_opening_shuts_the_armband_down(tmp_path):
    class Stopping(_Failing):
        def open(self):
            raise KeyboardInterrupt

    session = _session(tmp_path, [Stopping()])
    with pytest.raises(KeyboardInterrupt):
        session.open()  # __enter__ would not run __exit__ if this escaped open()
    assert session.device._ser is None


def test_ctrl_c_quits_calibration():
    import asyncio
    import signal
    import threading

    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from deftemg_acquisition.acquisition.wuji.calibration import calibrate

    class Glove:
        serial_number = "A"

        async def calibrate(self, **_kwargs):
            await asyncio.sleep(30)
            return {"calibrated_urdf": "unused"}

    pg.mkQApp()
    QtCore.QTimer.singleShot(50, lambda: signal.raise_signal(signal.SIGINT))
    with pytest.raises(KeyboardInterrupt):
        calibrate(Glove(), "left", timeout_s=1)
    for thread in threading.enumerate():
        if thread.name == "wuji-calibrate":
            thread.join(timeout=2)
            assert not thread.is_alive()


def test_point_cloud_converter():
    from deftemg_acquisition.acquisition.wuji.wuji_recorder import CLOUD_POINTS

    pts = np.arange(10 * 5, dtype=np.float32).reshape(10, 5)  # x, y, z, pressure, extra
    frame = NS(point_stride=20, data=list(pts.tobytes()))  # the SDK gives a list of bytes
    out = STREAMS["tactile_point_cloud"](frame)
    assert out["position"].shape == (CLOUD_POINTS, 3)
    assert out["position"][:10].tolist() == pts[:, :3].tolist()
    assert out["pressure"][:10].tolist() == pts[:, 3].tolist()
    assert np.isnan(out["position"][10:]).all()  # a shorter cloud is padded
    big = np.zeros((CLOUD_POINTS + 7, 4), np.float32)
    assert STREAMS["tactile_point_cloud"](NS(point_stride=16, data=big.tobytes()))[
        "pressure"].shape == (CLOUD_POINTS,)  # fmt: skip


def test_glove_panel_selectors():
    from deftemg_acquisition.acquisition.wuji.wuji_recorder import flatten_fields
    from deftemg_acquisition.viz.glove import ZONES, field_columns, joint_angles_deg, zone_means

    labels, row = flatten_fields({"thumb": [0.2, -1.0, 0.4], "palm": [-1.0, -1.0],
                                  "index": [1.0]})  # fmt: skip
    assert field_columns(labels, "thumb").tolist() == [0, 1, 2]
    means = zone_means(labels)(row[None, :])[0]
    assert means[ZONES.index("thumb")] == pytest.approx(0.3)  # -1: no reading, left out
    assert means[ZONES.index("palm")] == 0.0 and means[ZONES.index("index")] == 1.0
    labels, row = flatten_fields({"angles": np.full(25, np.pi / 2), "confidence": [1.0] * 5})
    assert joint_angles_deg(labels)(row[None, :]).tolist() == [[pytest.approx(90.0)] * 25]


# --- StreamBuffer against the list-per-frame version it replaced ---------------------------------

log = logging.getLogger(__name__)


# The StreamBuffer before chunked buffers, verbatim (renamed): the behaviour the new one keeps.
class _ReferenceStreamBuffer:
    """One stream's frames during a trial. ``add`` runs on the SDK's callback thread.

    A frame whose shape differs from the trial's first frame fails the stream for that trial:
    its frames are freed, later ones are not kept, and ``stop`` raises."""

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
        self.last_ns = 0  # host time of the last frame, in a trial or not (0: none yet)

    def _clear(self) -> None:
        self._host: list[int] = []
        self._dev: list[int] = []
        self._seq: list[int] = []
        self._fields: dict[str, list[np.ndarray]] = {}
        self._shapes: dict[str, tuple[int, ...]] | None = None  # the trial's first frame
        self._failed: str | None = None

    def start(self, host_start_ns: int) -> None:
        with self._lock:
            self._clear()
            self._start_ns = host_start_ns
            self._recording = True

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
                self.n_errors += 1
                if self.n_errors == 1:
                    log.exception("glove %s: could not convert a frame", self.name)
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
            header = frame.header
            self._host.append(t_ns)
            self._dev.append(header.timestamp_us)
            self._seq.append(header.seq)
            for key, array in values.items():
                self._fields.setdefault(key, []).append(array)

    def error(self, message: str) -> None:
        with self._lock:
            self.n_errors += 1
        log.warning("glove %s: %s", self.name, message)

    def stop(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """End the trial: arrays of the frames that arrived in [start, end] and a summary."""
        with self._lock:
            self._recording = False
            try:
                if self._failed:
                    raise ValueError(self._failed)
                return self._arrays(host_end_ns)
            finally:
                self._clear()

    def _arrays(self, host_end_ns: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        host = np.asarray(self._host, dtype=np.int64)
        keep = (host >= self._start_ns) & (host <= host_end_ns)
        arrays = {
            f"{self.name}_host_ns": host[keep],
            f"{self.name}_dev_us": np.asarray(self._dev, dtype=np.int64)[keep],
            f"{self.name}_seq": np.asarray(self._seq, dtype=np.int64)[keep],
        }
        for key, values in self._fields.items():
            arrays[f"{self.name}_{key}"] = np.stack(values)[keep]
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
        return arrays, summary


_ZONE_NAMES = ("palm", "thumb", "index", "middle", "ring", "pinky")
_RESIZABLE = {"tactile", "tactile_binary", "tactile_residual", "tactile_zones",
              "hand_joint_angles", "hand_skeleton", "emf_poses", "tip_poses"}  # fmt: skip


def _glove_frame(topic, rng, seq, size=None):
    """A frame of ``topic`` with random values, as the SDK gives it; ``size`` changes the
    shape of its fields (``_RESIZABLE`` topics only)."""
    header = NS(seq=seq, timestamp_us=int(rng.integers(-(2**40), 2**40)), frame_id="")

    def floats(n):
        return rng.standard_normal(n).tolist()

    def xyz(names="xyz"):
        return NS(**dict(zip(names, floats(len(names)), strict=True)))

    def pose():
        return NS(position=floats(3), orientation=xyz("xyzw"))

    if topic.startswith("tactile") and topic not in ("tactile_zones", "tactile_point_cloud"):
        return NS(header=header, data=floats(size or 5))
    if topic == "tactile_zones":  # scalar zones; ``size`` turns the palm into a list
        zones = {z: float(rng.standard_normal()) for z in _ZONE_NAMES}
        if size:
            zones["palm"] = floats(size)
        return NS(header=header, **zones)
    if topic == "hand_joint_angles":
        fingers = [NS(angles=floats(5), confidence=float(rng.random())) for _ in range(size or 5)]
        return NS(header=header, fingers=fingers)
    if topic in ("hand_skeleton", "emf_poses", "tip_poses"):
        items = [NS(pose=pose(), confidence=float(rng.random())) for _ in range(size or 3)]
        return NS(header=header, **{"joints" if topic == "hand_skeleton" else "poses": items})
    if topic == "tactile_point_cloud":  # a shorter cloud (NaN padded) half the time
        n = wuji_recorder.CLOUD_POINTS - int(rng.integers(0, 2)) * 40
        pts = rng.standard_normal((n, 5)).astype(np.float32)
        return NS(header=header, point_stride=20, data=list(pts.tobytes()))
    if topic.startswith("imu_data_"):
        return NS(header=header, orientation=xyz("xyzw"), angular_velocity=xyz(),
                  linear_acceleration=xyz())  # fmt: skip
    raise KeyError(topic)


def _unconvertible(seq):
    return NS(header=NS(seq=seq, timestamp_us=0, frame_id=""))  # no values: every converter fails


def _observe(buf, op):
    """Apply one op to a buffer; what can be seen of it afterwards."""
    kind, *args = op
    out: Any = None
    if kind == "start":
        buf.start(args[0])
    elif kind == "add":
        buf.add(args[0], t_ns=args[1])
    elif kind == "error":
        buf.error(args[0])
    else:
        try:
            arrays, summary = buf.stop(args[0])
        except Exception as e:
            out = ("raised", type(e), str(e))
        else:
            out = ("ok", summary, [
                (key, a.dtype.str, a.shape, a.tobytes(), a.flags.c_contiguous, a.flags.writeable)
                for key, a in arrays.items()
            ])  # fmt: skip
            for a in arrays.values():  # the same values, NaNs included
                assert np.array_equal(a, a.copy(), equal_nan=a.dtype.kind == "f")
    return out, buf.n_received, buf.last_ns, buf.n_errors


def _differential(topic, ops, chunk_frames=4):
    """Run ``ops`` on the old and the new StreamBuffer side by side; everything matches."""
    ref = _ReferenceStreamBuffer(f"A_{topic}", STREAMS[topic], topic)
    new = StreamBuffer(f"A_{topic}", STREAMS[topic], topic, chunk_frames=chunk_frames)
    n_stops = 0
    for i, op in enumerate(ops):
        expected, got = _observe(ref, op), _observe(new, op)
        assert got == expected, f"op {i} {op[0]}"
        n_stops += op[0] == "stop"
    return n_stops


def _trial(topic, rng, times, start=1000, end=2000, seq0=0):
    ops = [("start", start)]
    ops += [("add", _glove_frame(topic, rng, seq0 + i), t) for i, t in enumerate(times)]
    return ops + [("stop", end)]


def _inside(n, start=1000, step=7):
    return [start + 1 + step * i for i in range(n)]


def _scenario(name, topic, rng):
    frame = partial(_glove_frame, topic, rng)
    if name == "no frames":
        return [("start", 1000), ("stop", 2000)]
    if name == "frames received but none kept":
        return _trial(topic, rng, [10, 999, 2001, 5000])
    if name in ("one frame", "exactly one chunk", "one chunk plus one", "several chunks"):
        n = {"one frame": 1, "exactly one chunk": 4, "one chunk plus one": 5}.get(name, 13)
        return _trial(topic, rng, _inside(n))
    if name == "frames before start and after end":
        return [("add", frame(0), 500), *_trial(topic, rng, [900, 1000, 1500, 2000, 2001], seq0=1),
                ("add", frame(9), 2500), ("start", 3000), ("add", frame(10), 2999),
                ("add", frame(11), 3000), ("stop", 3000)]  # fmt: skip
    if name == "non-monotonic host times":
        return _trial(topic, rng, [1500, 1200, 1900, 999, 1800, 1100, 2001, 1300, 1300, 1000])
    if name == "a conversion error mid-trial":
        ops = _trial(topic, rng, _inside(9))
        ops.insert(3, ("add", _unconvertible(2), 1010))
        ops.insert(6, ("add", _unconvertible(5), 1030))  # where the second chunk starts
        return ops
    if name == "error() calls":
        ops = [("error", "before"), *_trial(topic, rng, _inside(6))]
        ops.insert(3, ("error", "during"))
        ops.insert(5, ("add", _unconvertible(9), 1020))
        return ops + [("error", "after"), *_trial(topic, rng, _inside(3))]
    if name == "back-to-back trials":
        ops = (
            _trial(topic, rng, _inside(5))
            + _trial(topic, rng, _inside(4, start=2000), 2000, 3000)
            + _trial(topic, rng, _inside(9), 2500)
        )
        return ops + [("start", 4000), ("start", 4100), ("add", frame(1), 4200),
                      ("stop", 5000), ("stop", 6000), ("stop", 4000)]  # fmt: skip
    size = 2 if topic != "hand_joint_angles" else 4
    if name == "a shape change mid-chunk":
        ops = _trial(topic, rng, _inside(7))
        ops.insert(3, ("add", frame(2, size=size), 1010))
        return ops
    if name == "a shape change at a chunk boundary":
        ops = _trial(topic, rng, _inside(7))
        ops.insert(5, ("add", frame(4, size=size), 1028))
        return ops
    if name == "a failed trial, then a good one":
        ops = _trial(topic, rng, _inside(6))
        ops.insert(2, ("add", frame(1, size=size), 1004))
        ops.insert(4, ("add", _unconvertible(3), 1009))  # not even converted once failed
        return ops + _trial(topic, rng, _inside(6, start=3000), 3000, 4000)
    raise KeyError(name)


_SCENARIOS = ["no frames", "frames received but none kept", "one frame", "exactly one chunk",
              "one chunk plus one", "several chunks", "frames before start and after end",
              "non-monotonic host times", "a conversion error mid-trial", "error() calls",
              "back-to-back trials", "a shape change mid-chunk",
              "a shape change at a chunk boundary", "a failed trial, then a good one"]  # fmt: skip


@pytest.mark.parametrize("topic", sorted(STREAMS))
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_chunked_stream_buffer_matches_the_list_version(scenario, topic):
    if "shape" in scenario and topic not in _RESIZABLE:
        pytest.skip("this stream's converter always gives the same shape")
    assert _differential(topic, _scenario(scenario, topic, np.random.default_rng(7))) >= 1


@pytest.mark.parametrize("topic", sorted(STREAMS))
def test_chunked_stream_buffer_fuzz(topic):
    """Seeded random trials: frame counts across chunk boundaries, host times in and out of
    order and of the window, conversion errors, SDK errors, shape changes, stray frames."""
    for seed in range(25):
        rng = np.random.default_rng(seed)
        ops, seq, t0 = [], 0, 0
        for _ in range(int(rng.integers(1, 5))):
            for _ in range(int(rng.integers(0, 3))):  # between trials
                ops.append(("add", _glove_frame(topic, rng, seq), t0 + int(rng.integers(0, 50))))
                seq += 1
            start, end = t0 + 100, t0 + 100 + int(rng.integers(0, 400))
            trial: list[tuple] = []
            for _ in range(int(rng.choice([0, 1, 3, 4, 5, 8, 12, 17, 33]))):
                r = rng.random()
                t = int(rng.integers(start - 30, end + 30))
                if r < 0.05:
                    trial.append(("add", _unconvertible(seq), t))
                elif r < 0.08:
                    trial.append(("error", "sdk"))
                elif r < 0.1 and topic in _RESIZABLE:
                    trial.append(("add", _glove_frame(topic, rng, seq, size=2), t))
                else:
                    trial.append(("add", _glove_frame(topic, rng, seq), t))
                seq += int(rng.integers(-1, 3))  # mostly in order, some skipped or repeated
            if rng.random() < 0.5:  # host times in order (else as drawn)
                trial.sort(key=lambda op: op[2] if op[0] == "add" else 0)
            ops += [("start", start), *trial, ("stop", end)]
            t0 = end + int(rng.integers(0, 100))
        _differential(topic, ops, chunk_frames=int(rng.choice([1, 2, 4, 5])))


def test_chunks_are_filled_in_place():
    buf = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
    assert StreamBuffer("B", STREAMS["tactile"]).chunk_frames == 3600  # 30 s at 120 Hz
    buf.start(0)
    for i in range(5):
        buf.add(_frame(i, [float(i), 0.5]), t_ns=10 + i)
    assert len(buf._host) == 2 and len(buf._fields["data"]) == 2  # one array per 4 frames
    assert buf._fields["data"][0].shape == (4, 2) and buf._fields["data"][0].dtype == np.float32
    for chunk in (buf._host[0], buf._fields["data"][1]):  # given back to the system when freed
        base = chunk
        while isinstance(base, np.ndarray):
            base = base.base
        assert isinstance(base, memoryview) and isinstance(base.obj, mmap.mmap)
    arrays, _ = buf.stop(100)
    assert arrays["A_tactile_data"][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert buf._host == [] and buf._fields == {}  # handed over and freed
    with pytest.raises(ValueError, match="chunk_frames"):
        StreamBuffer("C", STREAMS["tactile"], chunk_frames=0)


def test_stop_copies_outside_the_lock(monkeypatch):
    """The SDK callback is not held up while a trial's arrays are put together."""
    buf = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
    gather = wuji_recorder._gather
    locked = []

    def check(*args, **kwargs):
        locked.append(buf._lock.locked())
        return gather(*args, **kwargs)

    monkeypatch.setattr(wuji_recorder, "_gather", check)
    buf.start(0)
    for i in range(6):
        buf.add(_frame(i, [1.0]), t_ns=10 + i)
    buf.stop(100)
    assert locked and not any(locked)


@pytest.mark.parametrize(
    "header",
    [
        NS(seq=2**63, timestamp_us=0),
        NS(seq=1, timestamp_us=None),
        NS(seq="x", timestamp_us=0),
        NS(seq=[1, 2], timestamp_us=0),
        NS(seq=[1], timestamp_us=0),
        NS(seq=np.uint64(2**63), timestamp_us=0),
        NS(seq=1, timestamp_us=np.uint64(2**64 - 1)),
        NS(seq=np.float64(1e30), timestamp_us=0),
        NS(seq=1, timestamp_us=-(2**63) - 1),
        None,
    ],  # fmt: skip
)
def test_a_bad_header_drops_its_frame(header, caplog):
    """A header the arrays can't hold (or no header) drops the frame as a conversion error,
    when it arrives; the list version kept it and failed the whole stream at ``stop``."""
    bad = NS(data=[9.0, 9.0]) if header is None else NS(header=header, data=[9.0, 9.0])
    for first in (False, True):  # also the trial's first frame, which sets the shapes
        new = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
        ref = _ReferenceStreamBuffer("A_tactile", STREAMS["tactile"])
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger=wuji_recorder.__name__):
            for buf in (new, ref):
                buf.start(0)
            for i in range(7):
                if i == (0 if first else 3):
                    new.add(bad, t_ns=20 + i)  # never raises on the SDK's thread
                    ref.error("stands in for the dropped frame")
                frame = _frame(i, [float(i), 2.0 * i])
                for buf in (new, ref):
                    buf.add(frame, t_ns=10 + i)
            new.add(bad, t_ns=99)
            ref.error("stands in for the dropped frame")
        logged = [r for r in caplog.records if r.name == wuji_recorder.__name__]
        assert len(logged) == 1
        assert "header" in logged[0].getMessage()
        assert new.n_received == ref.n_received + 2 and new.last_ns == 99
        (got, got_summary), (expected, summary) = new.stop(200), ref.stop(200)
        assert got_summary == summary and summary["n_errors"] == 2
        assert list(got) == list(expected)
        for key in expected:
            assert got[key].dtype == expected[key].dtype
            assert np.array_equal(got[key], expected[key])

    ref = _ReferenceStreamBuffer("A_tactile", STREAMS["tactile"])
    ref.start(0)
    ref.add(_frame(1, [1.0, 2.0]), t_ns=5)
    if header is None:
        with pytest.raises(AttributeError):
            ref.add(bad, t_ns=10)
    else:
        ref.add(bad, t_ns=10)
        with pytest.raises((OverflowError, TypeError, ValueError)):
            ref.stop(100)


def test_add_while_stop_runs_keeps_rows_aligned():
    """Frames keep arriving on another thread while trials stop and start: every array kept
    has one row per frame, host, header and values from the same frame."""
    buf = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=8)
    counter = [0]
    done = threading.Event()

    def feed():
        while not done.is_set():
            i = counter[0]
            buf.add(_frame(i, [float(i), -float(i)]), t_ns=10**6 + i)
            counter[0] = i + 1
            time.sleep(0.00005)  # as the SDK, between frames (the lock is not fair)

    th = threading.Thread(target=feed)
    th.start()
    try:
        seen = 0
        for _ in range(20):
            buf.start(0)
            target = counter[0] + 50
            while counter[0] < target:
                time.sleep(0.0005)
            arrays, summary = buf.stop(2**62)
            seq = arrays["A_tactile_seq"]
            assert len(seq) == summary["n_frames"] > 0
            assert np.array_equal(seq, np.arange(seq[0], seq[0] + len(seq)))  # none lost
            assert seq[0] >= seen  # none from before this trial's start
            assert np.array_equal(arrays["A_tactile_host_ns"], 10**6 + seq)
            assert np.array_equal(arrays["A_tactile_dev_us"], 1000 * seq)
            assert np.array_equal(arrays["A_tactile_data"][:, 0], seq.astype(np.float32))
            assert np.array_equal(arrays["A_tactile_data"][:, 1], -seq.astype(np.float32))
            seen = int(seq[-1]) + 1
    finally:
        done.set()
        th.join()


def test_a_field_with_no_values_matches_the_list_version():
    frames = [("add", _frame(i, []), 10 + i) for i in range(6)]
    assert _differential("tactile", [("start", 0), *frames, ("stop", 100)]) == 1


def test_stop_frees_each_chunk_and_reports_the_errors_it_was_given(monkeypatch):
    """Every chunk is let go once copied; an error counted while the copy runs belongs to the
    next trial, not to the summary of this one."""
    buf = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
    gather = wuji_recorder._gather
    handed: list[list[Any]] = []

    def check(chunks, *args, **kwargs):
        handed.append(chunks)
        buf.error("arrives while stop copies")
        out = gather(chunks, *args, **kwargs)
        assert chunks and all(c is None for c in chunks)
        return out

    monkeypatch.setattr(wuji_recorder, "_gather", check)
    buf.start(0)
    for i in range(10):
        buf.add(_frame(i, [1.0]), t_ns=10 + i)
    buf.error("during the trial")
    arrays, summary = buf.stop(100)
    assert len(handed) == 4 and all(len(c) == 3 for c in handed)  # host, dev, seq, data
    assert summary["n_errors"] == 1 and buf.n_errors == 5
    assert arrays["A_tactile_seq"].tolist() == list(range(10))


class _Unwritable(np.ndarray):
    """A chunk whose writes fail while ``armed``."""

    armed = False

    def __setitem__(self, key, value):
        if _Unwritable.armed:
            raise ValueError("write failed")
        super().__setitem__(key, value)


class _FlakyChunks:
    """``_empty_chunk``, but its ``fail_at``-th call raises, and field chunks (float32) are
    ``_Unwritable``."""

    def __init__(self, real):
        self.real, self.calls, self.fail_at = real, 0, None

    def __call__(self, shape, dtype):
        self.calls += 1
        if self.calls == self.fail_at:
            raise MemoryError("no memory for a chunk")
        chunk = self.real(shape, dtype)
        return chunk.view(_Unwritable) if np.dtype(dtype) == np.float32 else chunk


def _store_badly(new, flaky, how, t_ns):
    """One add of ``new`` that fails while storing the frame, as ``how``."""
    frame = _frame(99, [9.0, 9.0])
    if how == "host time":
        new.add(frame, t_ns=2**63)  # not an int64
    elif how == "field write":
        _Unwritable.armed = True
        try:
            new.add(frame, t_ns=t_ns)
        finally:
            _Unwritable.armed = False
    else:  # the k-th chunk this add allocates: 1 a host time chunk, 4 the field's chunk
        flaky.fail_at = flaky.calls + {"no memory, host": 1, "no memory, field": 4}[how]
        new.add(frame, t_ns=t_ns)
        flaky.fail_at = None


@pytest.mark.parametrize(
    ("how", "at"),
    [("host time", 0), ("host time", 2), ("host time", 4),
     ("no memory, host", 0), ("no memory, host", 4),
     ("no memory, field", 0), ("no memory, field", 4),
     ("field write", 0), ("field write", 2), ("field write", 4)],
)  # fmt: skip
def test_a_frame_that_cant_be_stored_is_dropped(how, at, monkeypatch, caplog):
    """A frame that fails part-way through being stored (no memory for a chunk, a value the
    arrays can't hold) is dropped and counted, never raised on the SDK's thread, and leaves
    no partial row: the frames after it line up as if it had never come."""
    flaky = _FlakyChunks(wuji_recorder._empty_chunk)
    monkeypatch.setattr(wuji_recorder, "_empty_chunk", flaky)
    new = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
    ref = _ReferenceStreamBuffer("A_tactile", STREAMS["tactile"])

    def same(got, expected):
        (arrays, summary), (ref_arrays, ref_summary) = got, expected
        assert summary == ref_summary
        assert list(arrays) == list(ref_arrays)
        for key in ref_arrays:
            assert arrays[key].dtype == ref_arrays[key].dtype
            assert np.array_equal(arrays[key], ref_arrays[key])

    with caplog.at_level(logging.ERROR, logger=wuji_recorder.__name__):
        for buf in (new, ref):
            buf.start(0)
        for i in range(7):  # chunks of 4: frame 4 starts the second chunk
            if i == at:
                errors = new.n_errors
                _store_badly(new, flaky, how, t_ns=20 + i)
                assert new.n_errors == errors + 1 and new._n == i  # no row kept
                ref.error("stands in for the dropped frame")
            frame = _frame(i, [float(i), 2.0 * i])
            for buf in (new, ref):
                buf.add(frame, t_ns=10 + i)
        same(new.stop(200), ref.stop(200))

        for buf in (new, ref):  # a trial whose only frame fails: no field arrays at all
            buf.start(300)
        _store_badly(new, flaky, how, t_ns=310)
        ref.error("stands in for the dropped frame")
        same(new.stop(400), ref.stop(400))
    logged = [r for r in caplog.records if r.name == wuji_recorder.__name__]
    assert len(logged) == 1 and "store a frame" in logged[0].getMessage()


def test_a_field_named_like_a_header_array_matches_the_list_version():
    """A converter's field called ``host_ns``, ``dev_us`` or ``seq`` replaces that array in the
    trial, as before, and never changes which frames are kept."""

    def convert(frame):
        return {"host_ns": [frame.header.seq, 0.5], "seq": [1.0], "data": frame.data}

    new = StreamBuffer("A_x", convert, chunk_frames=4)
    ref = _ReferenceStreamBuffer("A_x", convert)
    ops = [("start", 1000)]
    ops += [("add", _frame(i, [float(i)]), t) for i, t in enumerate([5, 1001, 1500, 3000, 1999])]
    ops += [("add", _frame(5 + i, [1.0]), 1100 + i) for i in range(5)] + [("stop", 2000)]
    for i, op in enumerate(ops):
        assert _observe(new, op) == _observe(ref, op), f"op {i} {op[0]}"


def test_header_values_the_list_version_kept_are_kept_the_same():
    """Header values of other types, as the list version converted them (a fraction is cut)."""
    ops = [("start", 0)]
    for i, (seq, t_us) in enumerate(
        [
            (np.uint64(5), np.int32(7)),
            (np.float64(6.7), 2.5),
            (True, "9"),
            (np.int64(-(2**63)), np.uint64(2**63 - 1)),
        ]
    ):
        ops.append(("add", NS(header=NS(seq=seq, timestamp_us=t_us), data=[1.0 * i]), 10 + i))
    assert _differential("tactile", [*ops, ("stop", 100)]) == 1


def test_the_first_drop_of_each_kind_is_logged(caplog):
    """An SDK error does not hide a later kind of dropped frame from the log; every drop is
    counted, and each kind is logged once."""
    buf = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
    buf.start(0)
    with caplog.at_level(logging.ERROR, logger=wuji_recorder.__name__):
        buf.error("the SDK reported an error")
        for i in range(2):
            buf.add(_frame(i, [1.0]), t_ns=2**63)  # can't be stored
            buf.add(NS(header=NS(seq="x", timestamp_us=0), data=[1.0]), t_ns=10)
            buf.add(_unconvertible(i), t_ns=10)
    logged = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert buf.n_errors == 7
    assert len(logged) == 3
    for what in ("store a frame", "read a frame's header", "convert a frame"):
        assert sum(what in m for m in logged) == 1


def test_chunk_frames_is_a_whole_number():
    assert StreamBuffer("A", STREAMS["tactile"], chunk_frames=np.int64(4)).chunk_frames == 4
    with pytest.raises(TypeError):
        StreamBuffer("A", STREAMS["tactile"], chunk_frames=4.0)


def test_a_trial_started_while_the_last_one_is_copied_keeps_its_own_start(monkeypatch):
    """``start`` may run while ``stop`` copies (another thread): the stopped trial's summary
    uses the start it had."""
    buf = StreamBuffer("A_tactile", STREAMS["tactile"], chunk_frames=4)
    gather = wuji_recorder._gather

    def start_next(*args, **kwargs):
        buf.start(10**9)
        return gather(*args, **kwargs)

    buf.start(0)
    for i in range(10):
        buf.add(_frame(i, [1.0]), t_ns=10**8 + i)
    monkeypatch.setattr(wuji_recorder, "_gather", start_next)
    arrays, summary = buf.stop(10**9 // 2)  # a 0.5 s trial
    assert summary["n_frames"] == 10 and summary["rate_hz"] == 20.0
    assert buf._recording and buf._start_ns == 10**9

    monkeypatch.setattr(wuji_recorder, "_gather", gather)  # before the frames to keep are chosen
    arrays_of = buf._arrays

    def start_first(*args, **kwargs):
        buf.start(10**8 + 5)
        return arrays_of(*args, **kwargs)

    buf.start(0)
    for i in range(10):
        buf.add(_frame(i, [1.0]), t_ns=10**8 + i)
    monkeypatch.setattr(buf, "_arrays", start_first)
    arrays, summary = buf.stop(10**9 // 2)
    assert arrays["A_tactile_seq"].tolist() == list(range(10)) and summary["rate_hz"] == 20.0


def test_a_text_block_records_the_glove(fake_sdk, tmp_path):
    """A text-prompt block run in the prompter window with a glove: its trial has the glove's
    arrays and the glove's ``sources`` metadata, as lane blocks do."""
    import json

    import pyqtgraph as pg

    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.textblock import TextBlock, TextPrompt

    fake_sdk.found["A"] = "right"
    fake_sdk.offers.intersection_update({"tactile", "tactile_point_cloud"})
    prompts = [TextPrompt("Ball, power grasp", "ball", grasp="Power", highlight="hand")]
    block = TextBlock("ball", "grasp", "self_paced", prompts)
    rng = np.random.default_rng(0)
    pg.mkQApp()
    with _session(tmp_path, [WujiGloveRecorder()]) as session:
        win = TaskWindow(session, [(tmp_path / "ball.json", block)], tail_s=0.1)
        win.start_block()
        (glove,) = fake_sdk.connected.values()
        for seq in range(3):
            for topic in ("tactile", "tactile_point_cloud"):
                glove.callbacks[topic](_glove_frame(topic, rng, seq))
        win.end_block(completed=False)
        win.close()
    (rec,) = [json.loads(line) for line in (tmp_path / "s" / "trials.jsonl").open()]
    assert rec["metadata"]["task"] == "text_prompter"
    meta = rec["sources"]["wuji_glove"]
    assert meta["ok"] is True
    assert {k: v["n_frames"] for k, v in meta["streams"].items()} == {
        "A_tactile": 3, "A_tactile_point_cloud": 3}  # fmt: skip
    data = np.load(rec["npz_path"])
    assert data["wuji_glove_A_tactile_data"].shape == (3, 5)
    assert data["wuji_glove_A_tactile_point_cloud_pressure"].shape == (
        3, wuji_recorder.CLOUD_POINTS)  # fmt: skip
    assert data["wuji_glove_A_tactile_seq"].tolist() == [0, 1, 2]
    assert "text_go_host_ns" in data.files  # the block's own arrays are there too
