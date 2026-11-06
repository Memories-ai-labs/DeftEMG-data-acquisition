"""Wuji Glove as a session source: buffering, saving with the armband, failure isolation."""

import json
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband
from deftemg_acquisition.acquisition.wuji import StreamBuffer, WujiGloveRecorder
from deftemg_acquisition.acquisition.wuji.sim import N_TAXELS, SimulatedGlove
from deftemg_acquisition.acquisition.wuji.wuji_recorder import STREAMS


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


def test_converters_shape_sdk_frames():
    glove = SimulatedGlove()
    rec = WujiGloveRecorder(streams=["hand_joint_angles", "emf_poses", "imu_data_palm"],
                            glove=glove)  # fmt: skip
    rec.open()
    try:
        rec.start_trial(1, 0)
        time.sleep(0.2)
        arrays, meta = rec.end_trial(2**62)
    finally:
        rec.close()
    n = meta["streams"]["hand_joint_angles"]["n_frames"]
    assert n > 5
    assert arrays["hand_joint_angles_angles"].shape == (n, 25)
    assert arrays["hand_joint_angles_confidence"].shape == (n, 5)
    m = len(arrays["emf_poses_host_ns"])
    assert arrays["emf_poses_position"].shape == (m, 5, 3)
    assert arrays["emf_poses_orientation"].shape == (m, 5, 4)
    assert arrays["imu_data_palm_acc"].shape[1:] == (3,)


def test_unknown_stream_is_rejected():
    with pytest.raises(ValueError, match="unknown glove streams"):
        WujiGloveRecorder(streams=["tactile", "nope"])


def _session(tmp_path, sources):
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    return AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband()),
                              sources=sources)  # fmt: skip


def test_glove_saved_with_the_armband_trial(tmp_path):
    glove = WujiGloveRecorder(streams=["tactile", "hand_joint_angles"],
                              glove=SimulatedGlove(rate_hz=200))  # fmt: skip
    with _session(tmp_path, [glove]) as session:
        session.start_trial(label="press")
        time.sleep(0.6)
        result = session.end_trial()
    d = np.load(result.npz_path)
    t = d["glove_tactile_host_ns"]
    assert d["glove_tactile_data"].shape == (len(t), N_TAXELS)
    assert len(t) > 60  # ~120 frames at 200 Hz
    # on the armband's timeline, inside the trial
    assert result.host_start_ns <= t[0] and t[-1] <= result.host_end_ns
    assert d["emg_host_time_ns"][0] < t[-1] and t[0] < d["emg_host_time_ns"][-1]
    assert len(d["emg_uv"]) > 0
    rec = json.loads((tmp_path / "s" / "trials.jsonl").read_text().splitlines()[0])
    assert rec["sources"]["glove"]["ok"]
    assert rec["sources"]["glove"]["streams"]["tactile"]["n_frames"] == len(t)
    info = json.loads((tmp_path / "s" / "session.json").read_text())
    assert info["sources"]["glove"]["sn"] == "SIMGLOVE0001"
    assert info["sources"]["glove"]["simulated"] is True


class _Failing:
    name = "glove"
    closed = False

    def open(self):
        return {}

    def start_trial(self, index, host_start_ns):
        raise RuntimeError("start bug")

    def end_trial(self, host_end_ns):
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


def test_prompter_block_records_the_glove(tmp_path):
    import pyqtgraph as pg
    from PyQt5 import QtCore

    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    seq = Sequence("quick", "single", [Prompt(0.2, 0.3, ("index",), ("high",))])
    app = pg.mkQApp()
    glove = WujiGloveRecorder(streams=["tactile"], glove=SimulatedGlove(rate_hz=200))
    with _session(tmp_path, [glove]) as session:
        win = TaskWindow(session, [(tmp_path / "q.json", seq)], tail_s=0.2)
        win.show()
        win.start_block()
        deadline = time.monotonic() + 5
        while win.running and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        result = session.trials[0]
        win.close()
    d = np.load(result.npz_path)
    assert len(d["glove_tactile_host_ns"]) > 50
    assert "prompt_onset_host_ns" in d.files  # prompts, EMG and glove in one file


def test_live_glove_reaches_the_viewer_panels():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pylsl
    import pyqtgraph as pg

    from deftemg_acquisition.viz.glove import glove_panels

    pg.mkQApp()
    panels = glove_panels(hand=False)
    joints, zones = panels
    for _ in range(5):
        for pn in panels:
            pn.poll()
    assert not any(pn.widget.isVisible() for pn in panels)  # no glove streaming: hidden
    rec = WujiGloveRecorder(streams=["tactile"], live=True,
                            glove=SimulatedGlove(rate_hz=200))  # fmt: skip
    rec.open()
    assert set(rec.live_buffers) == {"hand_joint_angles", "tactile_zones",
                                     "tactile_point_cloud", "tip_poses"}  # fmt: skip
    for pn in panels:
        pn.widget.show()  # as inside a shown viewer window
        pn.widget.setVisible(False)
    try:
        deadline = time.monotonic() + 10
        while min(pn.inlet.n for pn in panels) < 20 and time.monotonic() < deadline:
            for pn in panels:
                pn.poll()
            time.sleep(0.02)
    finally:
        rec.close()
    assert all(pn.widget.isVisible() for pn in panels)
    assert len(joints.inlet.labels) == 30  # 25 angles, 5 confidences
    for pn in panels:
        pn.draw(pylsl.local_clock(), 5.0)
    t, y = joints.curves[0].xData, joints.curves[0].yData  # as set, not as clipped to view
    assert len(t) >= 20 and t.max() <= 0.5 and t.min() >= -5.0  # on the EMG time axis
    assert np.abs(y).max() > 1.0  # degrees, not radians
    assert len(zones.curves) == 6
    assert rec._live is None and rec.live_buffers == {}  # closed


def test_live_only_streams_are_not_saved_with_trials():
    rec = WujiGloveRecorder(streams=["tactile"], live=True, glove=SimulatedGlove())
    info = rec.open()
    try:
        assert info["streams"] == ["tactile"]
        assert "tactile_point_cloud" in info["live_streams"]
        rec.start_trial(1, 0)
        time.sleep(0.2)
        arrays, meta = rec.end_trial(2**62)
        assert rec.live_buffers["tactile_point_cloud"].n_received > 5
        assert rec._live.n_errors == 0  # every live frame published
    finally:
        rec.close()
    assert list(meta["streams"]) == ["tactile"]
    assert all(k.startswith("tactile_") for k in arrays)


def test_point_cloud_and_tip_poses_converters():
    from deftemg_acquisition.acquisition.wuji.wuji_recorder import CLOUD_POINTS

    pts = np.arange(10 * 5, dtype=np.float32).reshape(10, 5)  # x, y, z, pressure, extra
    frame = NS(point_stride=20, data=list(pts.tobytes()))  # as replayed: a list of bytes
    out = STREAMS["tactile_point_cloud"](frame)
    assert out["position"].shape == (CLOUD_POINTS, 3)
    assert out["position"][:10].tolist() == pts[:, :3].tolist()
    assert out["pressure"][:10].tolist() == pts[:, 3].tolist()
    assert np.isnan(out["position"][10:]).all()  # a shorter cloud is padded
    big = np.zeros((CLOUD_POINTS + 7, 4), np.float32)
    assert STREAMS["tactile_point_cloud"](NS(point_stride=16, data=big.tobytes()))[
        "pressure"].shape == (CLOUD_POINTS,)  # fmt: skip
    tips = SimulatedGlove().tip_poses()
    got = []
    sub = tips.subscribe_with_callback(got.append)
    time.sleep(0.05)
    sub.close()
    assert np.asarray(STREAMS["tip_poses"](got[0])["position"]).shape == (5, 3)


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
