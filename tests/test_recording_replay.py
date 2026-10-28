"""Recordings replayed as the armband (acquisition.replay): the live path without hardware."""

from pathlib import Path

import numpy as np
import pytest

from deftemg_acquisition.acquisition import AcquisitionSession, SessionConfig
from deftemg_acquisition.acquisition import protocol as P
from deftemg_acquisition.acquisition.device import DeviceError
from deftemg_acquisition.acquisition.replay import RecordedStream, ReplayArmband, emg_packets
from deftemg_acquisition.data.trials import load_trial, scan_trials


def _record(stream, tmp_path, speed=20.0):
    """Replay ``stream`` as one trial through AcquisitionSession; returns (the trial recorded
    again, the device)."""
    device = ReplayArmband(stream, speed=speed)
    config = SessionConfig(data_dir=tmp_path, session_name="replayed", live_stream=False,
                           continuous=False,
                           filter_enabled=stream.filter_status.enabled)  # fmt: skip
    with AcquisitionSession(config, device=device) as session:
        session.start_trial(label="replay")
        assert device.finished.wait(60)
        result = session.end_trial()
    return load_trial(Path(result.npz_path)), device


def _first_npz(session_dir: Path) -> Path:
    return next(i.npz_path for i in scan_trials([session_dir]) if not i.is_calibration)


def test_replayed_trial_records_the_same_emg(prompter_trial, tmp_path):
    """A recorded trial's bytes streamed through the armband interface: recorded again, it is
    the same EMG, sample for sample."""
    npz = _first_npz(prompter_trial)
    stream = RecordedStream.load([npz])
    again, device = _record(stream, tmp_path)
    assert device.n_delivered == len(stream.chunks) and device.finished.is_set()
    np.testing.assert_array_equal(again.emg_uv, load_trial(npz).emg_uv)


def test_breaks_in_the_recording_stay_breaks(synthetic_sessions, tmp_path):
    """Lost packets and a stall: the trial recorded again breaks in the same places."""
    uv = load_trial(_first_npz(synthetic_sessions[0])).emg_uv[:12000]
    packets = emg_packets(uv)
    del packets[5000:5020]  # 20 packets lost
    chunks = [b"".join(packets[a : a + 6]) for a in range(0, len(packets), 6)]
    t = np.arange(len(chunks)) * int(6 / 2000 * 1e9)
    t[len(chunks) * 3 // 4 :] += int(0.3e9)  # a stall of 0.3 s
    stream = RecordedStream(chunks, t, P.FilterStatus(False, 60, 60))
    again, _ = _record(stream, tmp_path, speed=50.0)
    assert len(again.segment_bounds()) == 3
    kept = np.delete(np.round(uv * 60) / 60, np.s_[5000:5020], axis=0)  # 1/60 µV counts
    np.testing.assert_allclose(again.emg_uv, kept, atol=1e-9)


def test_a_session_without_its_bytes_replays_from_its_emg(synthetic_sessions):
    """Synthetic sessions have no .bin: their EMG is sent as filter-off packets, every trial in
    recording order, on one timeline."""
    stream = RecordedStream.load([synthetic_sessions[0]])
    assert not stream.filter_status.enabled
    assert len(stream.trials) == len(scan_trials([synthetic_sessions[0]]))
    assert np.all(np.diff(stream.t_ns) > 0)
    (first, _), (_, end) = stream.trials[0], stream.trials[1]
    decoded = P.decode(b"".join(stream.chunks[:end]), filter_enabled=False)
    expected = np.round(load_trial(Path(first)).emg_uv * 60) / 60
    np.testing.assert_allclose(decoded.emg_uv, expected, atol=1e-9)
    assert decoded.n_dropped == 0


def test_replay_keeps_the_recorded_filter_setting():
    stream = RecordedStream([b""], np.zeros(1, np.int64), P.FilterStatus(False, 60, 60))
    device = ReplayArmband(stream)
    assert device.configure_filters(60, 60, enabled=False) == stream.filter_status
    with pytest.raises(DeviceError, match="filter off"):
        device.configure_filters(60, 60, enabled=True)
    with pytest.raises(ValueError, match="speed"):
        ReplayArmband(stream, speed=0)
