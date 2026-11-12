"""Wuji Glove as a session source: buffering, converters, failure isolation (no glove needed)."""

import sys
import time
from types import SimpleNamespace as NS

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


def test_every_session_calibrates_under_one_sdk_user(fake_sdk):
    fake_sdk.found.update({"A": "left"})
    for _ in range(2):
        rec = WujiGloveRecorder()
        rec.open()
        rec.close()
    assert [u["display_name"] for u in fake_sdk.users] == ["Default", "deftemg"]
    assert fake_sdk.current == ["u1"]


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


def test_source_open_failure_aborts_the_session(tmp_path):
    class NoGlove(_Failing):
        def open(self):
            raise RuntimeError("no glove found")

    src = NoGlove()
    with pytest.raises(RuntimeError, match="no glove found"):
        with _session(tmp_path, [src]):
            pass


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
