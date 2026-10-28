import logging
import time

import numpy as np
import pytest
from scipy import signal

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
from deftemg_acquisition.acquisition import protocol as P
from deftemg_acquisition.acquisition.sim import SimulatedArmband
from deftemg_acquisition.analysis.filtering import filter_emg

pylsl = pytest.importorskip("pylsl")
from deftemg_acquisition.acquisition.live import (  # noqa: E402 (needs liblsl)
    LiveFilter,
    LiveStreamer,
    describe_filter,
    live_filter_config,
)

FS = P.EMG_RATE_HZ


def amplitude(x: np.ndarray, f0: float) -> float:
    """Median over channels of the least-squares amplitude of an ``f0`` sinusoid in ``x``."""
    t = np.arange(len(x)) / FS
    basis = np.stack([np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t), t**0], axis=1)
    coef = np.linalg.lstsq(basis, x, rcond=None)[0]
    return float(np.median(np.hypot(coef[0], coef[1])))


def test_live_filter_choices():
    assert describe_filter(live_filter_config(20.0, "60")) == "high-pass 20 Hz, notch 60 Hz"
    assert describe_filter(live_filter_config(20.0, "50")) == "high-pass 20 Hz, notch 50 Hz"
    comb60 = live_filter_config(None, "comb60")
    assert describe_filter(comb60) == "notch 60, 180, 300, 420, 540, 660, 780, 900 Hz"
    comb50 = live_filter_config(None, "comb50")
    assert describe_filter(comb50) == "notch 50, 150, 250, 350, 450, 550, 650, 750, 850, 950 Hz"
    comb60all = live_filter_config(None, "comb60all")
    assert (
        describe_filter(comb60all)
        == "notch " + ", ".join(str(60 * k) for k in range(1, 17)) + " Hz"
    )
    comb50all = live_filter_config(None, "comb50all")
    assert (
        describe_filter(comb50all)
        == "notch " + ", ".join(str(50 * k) for k in range(1, 20)) + " Hz"
    )
    assert describe_filter(live_filter_config(40.0, "none")) == "high-pass 40 Hz"
    off = live_filter_config(None, "none")
    assert describe_filter(off) == "none" and LiveFilter(off).sos is None
    assert describe_filter(None) == "none"
    with pytest.raises(ValueError, match="notch"):
        live_filter_config(20.0, "comb")


def test_live_filter_chunks_offset_and_reset():
    rng = np.random.default_rng(0)
    n = int(2 * FS)
    t = np.arange(n) / FS
    emg = 20 * np.sin(2 * np.pi * 100 * t)[:, None] * np.ones(4)
    offset = np.array([800.0, -1500.0, 50.0, 3000.0])  # electrode offsets, as in raw data
    x = emg + offset

    cfg = live_filter_config(20.0, "comb60")
    live = LiveFilter(cfg)
    cuts = np.sort(rng.integers(0, n, 60))
    chunked = np.concatenate([live(c) for c in np.split(x, cuts)])
    # Chunk by chunk gives the offline causal filter of the whole recording.
    np.testing.assert_allclose(chunked, filter_emg(x, FS, cfg), atol=1e-9)

    # Starting from the first sample, the offsets don't ring through the high-pass ...
    assert np.abs(chunked[:200]).max() < 40
    # ... as they do when the filter starts from rest.
    assert np.abs(signal.sosfilt(live.sos, x, axis=0)[:200]).max() > 500
    assert amplitude(chunked[-1000:], 100) == pytest.approx(20, rel=0.05)  # passes

    # After a break, reset() starts over from the next chunk's first sample.
    live.reset()
    assert np.abs(live(x[:200] + 5000)).max() < 40


class SpyFilter:
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __call__(self, x):
        return x


def test_live_streamer_resets_filter_after_a_gap():
    streamer = LiveStreamer("test:gap", filter_enabled=False)
    streamer.emg_filter = spy = SpyFilter()
    pkt = P.SYNC + bytes([P.TYPE_EMG, 0]) + bytes(P.PACKET_LEN_FILTER_OFF - P.HEADER_LEN)
    t0 = 10**9
    streamer._publish(t0, pkt)  # first chunk: starts from its first sample
    streamer._publish(t0 + 10_000_000, pkt)  # 10 ms later: same stream
    assert spy.resets == 1
    streamer._publish(t0 + 2 * 10**9, pkt)  # after a stall
    assert spy.resets == 2


def test_live_streamer_resyncs_where_chunks_were_discarded():
    streamer = LiveStreamer("test:discard", filter_enabled=False, max_queued_chunks=2)
    streamer.emg_filter = spy = SpyFilter()
    resets_before = []  # resets so far as each chunk is published
    publish = streamer._publish
    streamer._publish = lambda t_ns, data: (resets_before.append(spy.resets), publish(t_ns, data))
    pkt = P.SYNC + bytes([P.TYPE_EMG, 0]) + bytes(P.PACKET_LEN_FILTER_OFF - P.HEADER_LEN)
    t0, dt = 10**9, 10_000_000
    for i in range(3):  # the third finds the queue full and is discarded
        streamer(t0 + i * dt, pkt)
    streamer._queue.get()  # the worker takes the first one ...
    streamer(t0 + 3 * dt, pkt)  # ... making room for the fourth, right after the lost one
    streamer.start()  # publishes the second and the fourth
    streamer.stop()
    # Chunk 2 continues chunk 1 (its only reset is the first-chunk one, inside publish); the
    # resync for the discarded chunk comes just before chunk 4.
    assert resets_before == [0, 2]


@pytest.mark.parametrize(("notch", "hum_uv"), [("60", 0.0), ("none", SimulatedArmband.HUM_UV)])
def test_live_stream_filtered_without_device_filter(tmp_path, notch, hum_uv):
    port = f"sim-livefilter-{notch}"
    config = SessionConfig(
        data_dir=tmp_path, session_name="s", filter_enabled=False, live_notch=notch
    )
    with AcquisitionSession(config, device=Armband(port, ser=SimulatedArmband())) as session:
        found = pylsl.resolve_bypred(f"source_id='waveletech:{port}' and type='EMG'", 1, 5)
        assert found
        inlet = pylsl.StreamInlet(found[0])
        desc = inlet.info(timeout=5).desc()
        assert desc.child_value("device_filter") == "off"
        expected = describe_filter(live_filter_config(20.0, notch))
        assert desc.child_value("software_filter") == expected
        inlet.open_stream(timeout=5)
        session.start_trial()
        time.sleep(1.0)  # past the notch's start-up transient (~0.5 s)
        result = session.end_trial()
        samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=100_000)
    x = np.asarray(samples)[-1000:]
    assert x.shape == (1000, 18)
    assert amplitude(x, 60) == pytest.approx(hum_uv, abs=0.2 * SimulatedArmband.HUM_UV)
    recorded = np.load(result.npz_path)["emg_uv"][-1000:]  # the recording stays raw
    assert amplitude(recorded, 60) == pytest.approx(SimulatedArmband.HUM_UV, rel=0.2)


@pytest.mark.parametrize(
    ("powerline_hz", "notch", "expected", "warns"),
    [
        (50, None, "comb50all", False),  # by default, the mains frequency and every harmonic
        (60, None, "comb60all", False),
        (
            50,
            "comb50",
            "high-pass 20 Hz, notch 50, 150, 250, 350, 450, 550, 650, 750, 850, 950 Hz",
            False,
        ),
        (60, "50", "high-pass 20 Hz, notch 50 Hz", True),  # mismatch: warned about
    ],
)
def test_live_notch_mains_frequency(tmp_path, caplog, powerline_hz, notch, expected, warns):
    config = SessionConfig(
        data_dir=tmp_path,
        session_name="s",
        filter_enabled=False,
        powerline_hz=powerline_hz,
        live_notch=notch,
    )
    device = Armband(f"sim-mains-{powerline_hz}-{notch}", ser=SimulatedArmband())
    with caplog.at_level(logging.WARNING), AcquisitionSession(config, device=device) as session:
        assert session.filter_status.powerline_hz == powerline_hz
        if notch is None:
            expected = describe_filter(live_filter_config(20.0, expected))
        assert describe_filter(session.live.emg_filter.cfg) == expected
    assert any("check the mains" in r.getMessage() for r in caplog.records) == warns
