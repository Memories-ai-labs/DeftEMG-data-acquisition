"""The armband on another computer (``acquisition.remote``), over loopback.

Every stream here has a source_id unique to its test (from the simulated armband's port), and
the receiving side asks for it by that id: another session running on this computer or the
network (a recording with the armband, another test run) publishes streams of the same types.
"""

import base64
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
from deftemg_acquisition.acquisition.armband import protocol as P
from deftemg_acquisition.acquisition.armband.device import DeviceError
from deftemg_acquisition.acquisition.armband.recorder import TrialRecorder
from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband
from deftemg_acquisition.clock import host_ns

pylsl = pytest.importorskip("pylsl")
from deftemg_acquisition.acquisition.remote import RemoteArmband  # noqa: E402


def unique_port() -> str:
    return f"remote-test-{uuid.uuid4().hex[:8]}"


def wait_until(cond, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_a_remote_session_records_the_senders_bytes_at_the_senders_times(tmp_path):
    """The receiving session saves exactly the bytes the sender read from its serial port, each
    chunk at the time the sender read it (one computer here, so one clock: equal to within the
    clock synchronisation's error), and with the sender's armband filter setting."""
    port = unique_port()
    sent = TrialRecorder()  # everything the sender reads
    device = Armband(port, ser=SimulatedArmband())
    device.attach_sink("all", sent)
    config = SessionConfig(data_dir=tmp_path, session_name="send", filter_enabled=False)
    with AcquisitionSession(config, device=device) as sender:
        assert sender.raw is not None
        remote = RemoteArmband(sender.source_id, timeout=10)
        status = remote.resolve()
        assert status == sender.filter_status and not status.enabled
        config = SessionConfig(data_dir=tmp_path, session_name="recv", filter_enabled=False)
        with AcquisitionSession(config, device=remote) as receiver:
            assert receiver.raw is None  # a remote session doesn't send the bytes on again
            assert receiver.source_id == f"waveletech:remote:{sender.source_id}"
            assert wait_until(lambda: remote.n_chunks > 50, 10)
            receiver.start_trial(label="grip")
            n0 = remote.n_chunks
            assert wait_until(lambda: remote.n_chunks > n0 + 300, 10)
            result = receiver.end_trial()
    assert remote.n_gaps == 0 and remote.n_bytes_lost == 0
    assert result.n_emg > 500 and result.n_dropped == 0
    info = (tmp_path / "recv" / "session.json").read_text()
    assert sender.source_id in info and "remote_host" in info

    raw, t_sent, end_sent = sent.snapshot()
    got = np.load(result.npz_path)
    got_raw = (tmp_path / "recv" / "trial_001_grip.bin").read_bytes()
    start = raw.find(got_raw[:4096])
    assert start >= 0 and raw[start : start + len(got_raw)] == got_raw  # the bytes, unchanged
    # the same chunks, at the same times
    ends = got["chunk_end_offset"] + start
    idx = np.searchsorted(end_sent, ends)
    assert np.array_equal(end_sent[idx], ends)
    err_ms = (got["chunk_host_time_ns"] - t_sent[idx]) / 1e6
    assert np.all(np.abs(err_ms) < 5), err_ms
    assert np.all(np.diff(got["emg_host_time_ns"]) >= 0)


def test_the_remote_armband_counts_lost_bytes_and_keeps_time_moving_forward():
    remote = RemoteArmband("nowhere")
    got = []
    remote.attach_sink("t", lambda t, d: got.append((t, d)))
    remote.acquiring = True
    lsl_minus_host = pylsl.local_clock() - host_ns() / 1e9
    now = pylsl.local_clock()

    def deliver(offset: int, data: bytes, stamp: float) -> None:
        remote._deliver([str(offset), base64.b64encode(data).decode()], stamp, lsl_minus_host)

    deliver(0, b"abcd", now - 0.010)
    deliver(4, b"ef", now - 0.020)  # stamped before the previous chunk: held at its time
    deliver(10, b"gh", now - 0.005)  # bytes 6-9 never arrived
    deliver(12, b"ij", now + 5.0)  # in the future (a clock offset error): now at the latest
    assert [d for _, d in got] == [b"abcd", b"ef", b"gh", b"ij"]
    t = [t for t, _ in got]
    assert t[1] == t[0] and t[2] > t[1] and t[3] <= host_ns()
    assert (remote.n_gaps, remote.n_bytes_lost, remote.n_chunks) == (1, 4, 4)
    remote.acquiring = False
    deliver(14, b"kl", now)  # not acquiring: read, but not handed on
    assert len(got) == 4 and remote.n_gaps == 1


def test_the_remote_armband_takes_the_senders_filter_setting(tmp_path):
    port = unique_port()
    config = SessionConfig(data_dir=tmp_path, session_name="send", filter_enabled=True)
    with AcquisitionSession(config, device=Armband(port, ser=SimulatedArmband())) as sender:
        remote = RemoteArmband(sender.source_id, timeout=10)
        assert remote.configure_filters(60, 60).enabled
        with pytest.raises(DeviceError, match="filter is on"):
            remote.configure_filters(60, 60, enabled=False)
        remote.close()  # never opened: nothing to do


def test_no_remote_armband_is_an_error_naming_what_was_looked_for():
    with pytest.raises(DeviceError, match="no armband stream from 'waveletech:nothing-here"):
        RemoteArmband(f"waveletech:nothing-here-{uuid.uuid4().hex[:8]}", timeout=0.5).resolve()


def test_serve_in_another_process_feeds_a_session_here(tmp_path):
    """``serve`` in its own process (the computer with the armband) and a session here: the
    EMG arrives complete and promptly."""
    port = unique_port()
    serve = subprocess.Popen(
        [sys.executable, "-m", "deftemg_acquisition.cli.serve", "--simulate", "--port", port,
         "--duration", "60", "--status-every", "1"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        remote = RemoteArmband(f"waveletech:{port}", timeout=20)
        config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False,
                               filter_enabled=remote.resolve().enabled)  # fmt: skip
        with AcquisitionSession(config, device=remote) as session:
            assert wait_until(lambda: remote.n_chunks > 50, 10)
            session.start_trial()
            time.sleep(1.5)
            result = session.end_trial()
    finally:
        serve.terminate()
        out = serve.communicate(timeout=10)[0]
    assert f"--remote waveletech:{port}" in out
    assert result.n_dropped == 0 and remote.n_bytes_lost == 0
    assert result.n_emg / result.duration_s == pytest.approx(P.EMG_RATE_HZ, rel=0.1)
    assert remote.latency_summary()["median_ms"] < 50


class FakeInfo:
    def __init__(self, source_id: str, hostname: str):
        self._s, self._h = source_id, hostname

    def source_id(self) -> str:
        return self._s

    def hostname(self) -> str:
        return self._h


def test_the_viewer_reads_one_sessions_streams(monkeypatch):
    import socket

    from deftemg_acquisition.viz import viewer

    monkeypatch.setattr(socket, "gethostname", lambda: "here")
    mac, local = FakeInfo("waveletech:/dev/cu.usbmodem1", "mac"), FakeInfo("waveletech:x", "here")
    try:
        viewer.select_source(None)  # this computer's first, then the same one for every stream
        assert viewer._pick([mac, local]) is local
        assert viewer._pick([mac]) is None
        viewer.select_source("mac")  # by host name
        assert viewer._pick([local, mac]) is mac
        viewer.select_source("waveletech:x")  # by source_id
        assert viewer._pick([mac, local]) is local
        assert viewer._pick([mac]) is None
    finally:
        viewer.select_source(None)


def test_the_clis_take_a_remote_armband():
    from deftemg_acquisition.cli import prompter
    from deftemg_acquisition.viz import viewer

    args = prompter.parser().parse_args(["--remote", "mac", "x.json"])
    assert args.remote == "mac"
    assert viewer.parser().parse_args(["--source", "mac"]).source == "mac"
