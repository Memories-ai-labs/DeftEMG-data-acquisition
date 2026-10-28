import json
import time

import numpy as np
import pytest

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig, decode
from deftemg_acquisition.acquisition import protocol as P
from deftemg_acquisition.acquisition.device import _looks_usb
from deftemg_acquisition.acquisition.sim import SimulatedArmband


def emg_packet(seq: int, counts: list[int]) -> bytes:
    body = b"".join(int(c).to_bytes(2, "big", signed=True) for c in counts)
    return P.SYNC + bytes([P.TYPE_EMG, seq & 0xFF]) + body


def imu_packet(seq: int, device_ms: int) -> bytes:
    words = [366, 0, -4, -3, 1433, 12212, 10620, 4100]
    body = b"".join(w.to_bytes(2, "big", signed=True) for w in words)
    body += bytes([87]) + device_ms.to_bytes(8, "big") + bytes(11)
    return P.SYNC + bytes([P.TYPE_IMU, seq & 0xFF]) + body


def test_command_formats():
    assert P.set_timestamp_cmd(0x000001A056048660) == "TIME000001A056048660"
    assert P.set_powerline_freq_cmd(60) == "SetPFFilterFreq-60"
    assert P.set_highpass_freq_cmd(60) == "SetHPFilterFreq-60"
    assert P.set_filter_enabled_cmd(False) == "SetFilterEN-0"
    assert P.FilterStatus.parse(b"junk1-60-20\r\n") == P.FilterStatus(True, 60, 20)


def test_decode_filter_on_with_garbage_and_drop():
    counts = list(range(-9, 9))
    stream = b"StartDataAcq\r\n" + emg_packet(254, counts) + imu_packet(255, 1234)
    stream += emg_packet(1, counts)  # seq 0 dropped across the wrap
    stream += emg_packet(2, counts)[:20]  # truncated tail
    d = decode(stream)
    assert len(d.emg_seq) == 2 and len(d.imu_seq) == 1
    np.testing.assert_array_equal(d.emg_seq, [254, 257])
    assert d.n_dropped == 1
    np.testing.assert_allclose(d.emg_uv[0], np.array(counts) / 7.5)
    assert d.device_time_ms[0] == 1234
    assert d.temperature_c[0] == pytest.approx(36.6)
    assert d.gyro_rad_s[0, 1] == pytest.approx(-4 * 0.001225)
    assert d.battery_mv[0] == 4100 and d.battery_pct[0] == 87


def test_decode_filter_off_24bit():
    counts = [-(2**23), 2**23 - 1] + [0] * 16
    body = b"".join(c.to_bytes(3, "big", signed=True) for c in counts)
    pkt = P.SYNC + bytes([P.TYPE_EMG, 7]) + body
    assert len(pkt) == P.PACKET_LEN_FILTER_OFF
    d = decode(pkt * 3, filter_enabled=False)
    assert d.emg_counts.shape == (3, 18)
    assert d.emg_counts[0, 0] == -(2**23) and d.emg_counts[0, 1] == 2**23 - 1


def test_stream_decoder_matches_batch_decode():
    rng = np.random.default_rng(1)
    stream = b"junk" + b"".join(
        imu_packet(i, i) if i % 10 == 9 else emg_packet(i, rng.integers(-500, 500, 18))
        for i in range(600)
    )
    stream = stream[:5000] + stream[5000 + 41 :]  # lose one packet mid-stream
    whole = decode(stream)
    dec = P.StreamDecoder()
    cuts = np.sort(rng.integers(0, len(stream), 80))
    chunks = np.split(np.frombuffer(stream, np.uint8), cuts)
    parts = [dec.feed(c.tobytes()) for c in chunks]
    for field in ("emg_uv", "emg_seq", "imu_seq", "device_time_ms"):
        got = np.concatenate([getattr(d, field) for d in parts])
        np.testing.assert_array_equal(got, getattr(whole, field))
    assert dec.n_dropped_total == whole.n_dropped == 1
    dec = P.StreamDecoder()
    dec.feed(stream[:100])
    tail = 100 - (4 + 2 * 41)  # after the junk and two whole packets: a packet's start
    assert dec.n_carried == tail and dec.feed(b"").n_dropped == 0
    dec.reset()
    assert dec.n_carried == 0


def run_trial(tmp_path, continuous, live_stream=False):
    sim = SimulatedArmband()
    config = SessionConfig(
        data_dir=tmp_path, session_name="s1", continuous=continuous, live_stream=live_stream
    )
    session = AcquisitionSession(config, device=Armband("sim", ser=sim))
    session.open()
    assert session.filter_status == P.FilterStatus(True, 60, 60)
    time.sleep(0.2)
    session.start_trial(label="pinch index")
    time.sleep(0.3)
    result = session.end_trial()
    return sim, session, result


@pytest.mark.parametrize("continuous", [True, False])
def test_session_trial_flow(tmp_path, continuous):
    sim, session, result = run_trial(tmp_path, continuous)
    session.close()

    sent = sim.commands
    assert sent[:4] == ["StopDataAcq", "SetPFFilterFreq-60", "SetHPFilterFreq-60",
                        "GetFilterStatus"]  # fmt: skip
    if continuous:  # stream from open; TIME at trial start while streaming; stop at close
        assert sent[4].startswith("TIME") and sent[5] == "StartDataAcq"
        assert sent[6].startswith("TIME") and sent[7:] == ["StopDataAcq"]
    else:
        assert sent[4].startswith("TIME") and sent[5:] == ["StartDataAcq", "StopDataAcq"]
    trial_ts = int(sent[6 if continuous else 4][4:], 16)
    assert trial_ts == result.device_timestamp_ms

    assert result.label == "pinch index" and result.n_dropped == 0
    assert 450 < result.n_emg < 800 and result.n_imu > 40  # ~0.3 s, not the 0.2 s before it
    data = np.load(result.npz_path)
    assert data["emg_uv"].shape == (result.n_emg, 18)
    # In continuous mode the first packets may predate TIME taking effect (no echo to wait
    # for), but the clock was already synced at open, so it is within a few ms either way.
    assert abs(int(data["device_time_ms"][0]) - trial_ts) < 50
    assert np.all(np.diff(data["emg_host_time_ns"]) >= 0)
    log = [json.loads(line) for line in (tmp_path / "s1" / "trials.jsonl").read_text().splitlines()]
    assert log[0]["npz_path"].endswith("trial_001_pinch-index.npz")


def test_live_stream_over_lsl(tmp_path):
    pylsl = pytest.importorskip("pylsl")
    sim = SimulatedArmband()
    config = SessionConfig(data_dir=tmp_path, session_name="s1", live_stream=True)
    with AcquisitionSession(config, device=Armband("sim", ser=sim)) as session:
        emg_info = pylsl.resolve_byprop("type", "EMG", timeout=5)
        mk_info = pylsl.resolve_byprop("type", "Markers", timeout=5)
        assert emg_info and mk_info
        emg = pylsl.StreamInlet(emg_info[0])
        markers = pylsl.StreamInlet(mk_info[0])
        emg.open_stream(timeout=5)
        markers.open_stream(timeout=5)
        time.sleep(0.2)
        session.start_trial(label="grip")
        time.sleep(0.3)
        session.end_trial()
        time.sleep(0.2)
        samples, _ = emg.pull_chunk(timeout=1.0, max_samples=100_000)
        got_markers, _ = markers.pull_chunk(timeout=1.0)
    assert len(samples) > 800 and len(samples[0]) == 18
    assert [m[0] for m in got_markers] == ["trial_start|1|grip", "trial_end|1"]


def test_open_when_device_already_stopped(tmp_path):
    # Regression: the stopped device answers StopDataAcq slowly with "Unknown Command" and drops
    # commands sent before that reply, which left the power-line filter at its 50 Hz default.
    sim = SimulatedArmband(streaming=False)
    sim.filters = [1, 50, 60]
    config = SessionConfig(data_dir=tmp_path, session_name="s1", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=sim)) as session:
        assert session.filter_status == P.FilterStatus(True, 60, 60)
    assert sim.dropped_commands == []


def test_session_with_device_filter_off(tmp_path):
    sim = SimulatedArmband()
    config = SessionConfig(
        data_dir=tmp_path, session_name="raw", filter_enabled=False, live_stream=False
    )
    with AcquisitionSession(config, device=Armband("sim", ser=sim)) as session:
        assert session.filter_status == P.FilterStatus(False, 60, 60)
        time.sleep(0.2)
        session.start_trial()
        time.sleep(0.3)
        result = session.end_trial()
    assert sim.commands[3:6] == ["GetFilterStatus", "SetFilterEN-0", "GetFilterStatus"]
    assert result.n_dropped == 0 and 450 < result.n_emg < 800 and result.n_imu > 40
    data = np.load(result.npz_path)
    assert abs(int(data["device_time_ms"][0]) - result.device_timestamp_ms) < 50
    # Raw 24-bit EMG at 1/60 uV per count, still carrying the hum the notch would remove.
    emg = data["emg_uv"]
    t = np.arange(len(emg)) / P.EMG_RATE_HZ
    basis = np.stack([np.sin(2 * np.pi * 60 * t), np.cos(2 * np.pi * 60 * t), t**0], axis=1)
    coef = np.linalg.lstsq(basis, emg, rcond=None)[0]
    assert np.median(np.hypot(coef[0], coef[1])) == pytest.approx(sim.HUM_UV, rel=0.2)
    info = json.loads((tmp_path / "raw" / "session.json").read_text())
    assert info["filter_status"]["enabled"] is False

    # The next session with default settings turns the filter back on.
    config = SessionConfig(data_dir=tmp_path, session_name="filtered", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=sim)) as session:
        assert session.filter_status == P.FilterStatus(True, 60, 60)
    assert "SetFilterEN-1" in sim.commands


def test_donning_ids_follow_the_answers(tmp_path):
    from deftemg_acquisition.acquisition.subjects import ask_donning

    def session(name, wall, trials=1, simulated=False, **meta):
        (tmp_path / name).mkdir()
        info = {"opened_at": "2026-10-16T09:00:00", "clock_anchor": {"wall_ns": wall},
                "metadata": {**meta, **({"simulated": True} if simulated else {})}}  # fmt: skip
        (tmp_path / name / "session.json").write_text(json.dumps(info))
        (tmp_path / name / "trials.jsonl").write_text("{}\n" * trials)

    quiet = {"say": lambda _: None}
    answers = iter(["", "S01"])  # an empty ID is asked again
    first = ask_donning(tmp_path, ask=lambda _: next(answers), **quiet)
    assert (first.subject, first.donning, first.session_in_donning) == ("S01", 1, 1)
    session("a", 1, **first.metadata())
    same = ask_donning(tmp_path, "S01", ask=lambda _: "n", **quiet)
    assert (same.donning, same.session_in_donning, same.previous_session) == (1, 2, "a")
    assert same.donning_started == first.donning_started
    session("b", 2, **same.metadata())
    session("c", 3, trials=0, subject="S01", donning=9)  # opened and quit: ignored
    session("d", 4, simulated=True, subject="S01", donning=7)  # simulated: its own history
    answers = iter(["maybe", "y"])  # until a yes / no
    redon = ask_donning(tmp_path, "S01", ask=lambda _: next(answers), **quiet)
    assert (redon.donning, redon.session_in_donning, redon.previous_session) == (2, 1, "b")
    assert ask_donning(tmp_path, "S01", redonned=True, simulated=True, **quiet).donning == 8
    assert ask_donning(tmp_path, "S02", ask=lambda _: "y", **quiet).donning == 1  # per subject
    # Sessions from before donning ids: no question, a new donning.
    session("old", 5, subject="S03")
    assert ask_donning(tmp_path, "S03", ask=lambda _: 1 / 0, **quiet).donning == 1
    # Unreadable or odd session.json files are skipped.
    for name, text in (("x", "[1, 2]"), ("y", '{"metadata": null}'), ("z", "\xff")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "session.json").write_text(text)
    assert ask_donning(tmp_path, "S01", redonned=False, **quiet).session_in_donning == 3
    said = []
    ask_donning(tmp_path, "s01", redonned=True, say=said.append)
    assert any("S01" in m for m in said)  # a near-duplicate ID is pointed out


def test_port_names_that_look_like_usb():
    for name in ("/dev/cu.usbmodem1101", "/dev/tty.usbmodem1101", "/dev/ttyACM0", "/dev/ttyUSB0"):
        assert _looks_usb(name), name
    for name in ("/dev/cu.Bluetooth-Incoming-Port", "/dev/ttyS0", "COM3"):
        assert not _looks_usb(name), name
