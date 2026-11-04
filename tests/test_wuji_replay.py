"""Wuji Glove replay from an SDK-style MCAP (synthetic recordings written here)."""

import argparse
import json
import time

import numpy as np
import pytest

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig, wuji
from deftemg_acquisition.acquisition.sim import SimulatedArmband
from deftemg_acquisition.acquisition.wuji import WujiGloveRecorder
from deftemg_acquisition.acquisition.wuji.replay import ReplayGlove, stream_topic

mcap_writer = pytest.importorskip("mcap.writer")

RATE_HZ = 120
N_TAXELS = 768


def _header(seq, t_us, frame_id="r_wrist"):
    return {"seq": seq, "timestamp_us": t_us, "frame_id": frame_id}


def _pose(k, t):
    return {"pose": {"position": [0.01 * k, t, 0.0],
                     "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
            "confidence": 0.9}  # fmt: skip


def _imu(seq, t_us):
    v = {"x": 0.1, "y": 0.2, "z": 9.8}
    return {"header": _header(seq, t_us, "r_palm_imu_link"),
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "orientation_covariance": [0.0] * 9, "angular_velocity": v,
            "angular_velocity_covariance": [0.0] * 9, "linear_acceleration": v,
            "linear_acceleration_covariance": [0.0] * 9}  # fmt: skip


def write_glove_mcap(path, seconds=1.0, hand="right", seq0=1000, dev0_us=5_000_000):
    """An MCAP laid out as the SDK's TopicRecorder writes it: JSON messages per glove topic."""
    p = f"/{hand}_glove/"
    log0 = 1_700_000_000_000_000_000
    with open(path, "wb") as f:
        w = mcap_writer.Writer(f)
        w.start()
        ch = {}
        for topic in ("tactile", "hand_joint_angles", "emf_poses", "imu_data/palm"):
            sid = w.register_schema(name=p + topic, encoding="jsonschema", data=b"{}")
            ch[topic] = w.register_channel(topic=p + topic, message_encoding="json",
                                           schema_id=sid)  # fmt: skip
        n = int(seconds * RATE_HZ)
        for i in range(n):
            t_ns = log0 + round(i * 1e9 / RATE_HZ)
            dev_us = dev0_us + round(i * 1e6 / RATE_HZ)
            hdr = _header(seq0 + i, dev_us)
            msgs = {
                "tactile": {"header": hdr, "data": [i / n] * 591 + [-1.0] * (N_TAXELS - 591)},
                "hand_joint_angles": {
                    "header": hdr,
                    "fingers": [{"angles": [i / n] * 5, "confidence": 0.95} for _ in range(5)],
                },
                "emf_poses": {"header": hdr, "poses": [_pose(k, i / n) for k in range(5)]},
                "imu_data/palm": _imu(seq0 + i, dev_us),
            }
            for topic, msg in msgs.items():
                w.add_message(ch[topic], log_time=t_ns, publish_time=t_ns,
                              data=json.dumps(msg).encode())  # fmt: skip
        w.finish()
    return path


def test_stream_topic_names():
    assert stream_topic("tactile", "/right_glove/") == "/right_glove/tactile"
    assert stream_topic("imu_data_palm", "/left_glove/") == "/left_glove/imu_data/palm"


def test_replay_gives_the_recorded_frames(tmp_path):
    path = write_glove_mcap(tmp_path / "g.mcap", seconds=0.5)
    glove = ReplayGlove(path, speed=4.0)
    assert glove.hand == "right" and glove.prefix == "/right_glove/"
    rec = WujiGloveRecorder(streams=["tactile", "hand_joint_angles", "emf_poses",
                                     "imu_data_palm"], glove=glove)  # fmt: skip
    info = rec.open()
    assert info["hand_side"] == "right" and info["replay"] == str(path)
    try:
        rec.start_trial(1, 0)
        t = time.monotonic()
        time.sleep(0.5 / 4 + 0.3)  # the whole recording at 4x, and some slack
        arrays, meta = rec.end_trial(2**62)
    finally:
        rec.close()
    n = 60
    assert meta["streams"]["tactile"]["n_frames"] == n
    assert arrays["tactile_data"].shape == (n, N_TAXELS)
    assert arrays["tactile_seq"].tolist() == list(range(1000, 1000 + n))  # header kept
    assert arrays["tactile_dev_us"][0] == 5_000_000
    assert arrays["hand_joint_angles_angles"].shape == (n, 25)
    assert arrays["emf_poses_position"].shape == (n, 5, 3)
    assert arrays["imu_data_palm_acc"].shape == (n, 3)
    # paced: spread over the recording's length / speed, not all at once
    span_s = (arrays["tactile_host_ns"][-1] - arrays["tactile_host_ns"][0]) / 1e9
    assert 0.08 < span_s < 0.3, span_s
    assert time.monotonic() - t < 2


def test_replay_loops(tmp_path):
    path = write_glove_mcap(tmp_path / "g.mcap", seconds=0.1)
    rec = WujiGloveRecorder(streams=["tactile"], glove=ReplayGlove(path, speed=5.0, loop=True))
    rec.open()
    try:
        rec.start_trial(1, 0)
        time.sleep(0.6)  # each pass: 0.1 s + 0.5 s gap, at 5x
        _, meta = rec.end_trial(2**62)
    finally:
        rec.close()
    assert meta["streams"]["tactile"]["n_frames"] > 12  # more than one pass


def test_missing_stream_is_reported(tmp_path):
    glove = ReplayGlove(write_glove_mcap(tmp_path / "g.mcap", seconds=0.1))
    with pytest.raises(AttributeError, match="tactile_binary"):
        WujiGloveRecorder(streams=["tactile_binary"], glove=glove).open()
    assert glove.available_streams(["tactile", "tactile_binary"]) == ["tactile"]


def test_cli_options_make_a_replay_source(tmp_path):
    path = write_glove_mcap(tmp_path / "g.mcap", seconds=0.1)
    ap = argparse.ArgumentParser()
    wuji.add_arguments(ap)
    (src,) = wuji.sources_from_args(ap.parse_args(["--wuji-replay", str(path),
                                                   "--wuji-replay-speed", "2"]))  # fmt: skip
    assert isinstance(src.glove, ReplayGlove) and src.glove.speed == 2.0
    assert wuji.sources_from_args(ap.parse_args([])) == []


def test_replayed_glove_saved_with_the_armband_trial(tmp_path):
    path = write_glove_mcap(tmp_path / "g.mcap", seconds=2.0)
    glove = WujiGloveRecorder(streams=["tactile"], glove=ReplayGlove(path))
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband()),
                            sources=[glove]) as session:  # fmt: skip
        session.start_trial()
        time.sleep(0.5)
        result = session.end_trial()
    d = np.load(result.npz_path)
    t = d["glove_tactile_host_ns"]
    assert len(t) > 40 and d["glove_tactile_data"].shape == (len(t), N_TAXELS)
    assert result.host_start_ns <= t[0] and t[-1] <= result.host_end_ns
    assert len(d["emg_uv"]) > 0
    info = json.loads((tmp_path / "s" / "session.json").read_text())
    assert info["sources"]["glove"]["replay"] == str(path)
