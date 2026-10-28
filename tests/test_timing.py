import numpy as np
import pytest

from deftemg_acquisition.analysis.timing import (
    analyse_trial,
    detect_onsets,
    emg_envelope,
    fit_sample_times,
    usb_delay,
)

T0 = 1_792_000_000_000_000_000  # an arbitrary host time in ns


def chunked_arrivals(n, rate=2000.0, chunk=16, delay_ms=3.0, seed=0):
    """True sample times, and host arrival times as stamped per USB chunk."""
    rng = np.random.default_rng(seed)
    true = T0 + np.round(np.arange(n) * 1e9 / rate).astype(np.int64)
    host = np.empty(n, np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        host[s:e] = true[e - 1] + int((delay_ms + rng.uniform(0, 1)) * 1e6)
    return true, host


def test_fit_sample_times_recovers_rate_and_smooths_chunks():
    true, host = chunked_arrivals(40_000, rate=2000.1)
    fitted, clock = fit_sample_times(host)
    assert clock.rate_hz == pytest.approx(2000.1, abs=0.01)
    assert clock.arrival_jitter_ms_p99 < 6
    # fitted times differ from the truth by a constant (mean delay), not by chunk steps
    err_ms = (fitted - true) / 1e6
    assert np.ptp(err_ms) < 0.5


def test_usb_delay_and_drift():
    dev_ms = 1_792_000_000_000 + np.arange(0, 60_000, 5)  # 60 s of IMU packets
    # constant 4 ms delay, and the host clock runs 50 ppm fast relative to the armband
    host_ns = (dev_ms + 4.0) * 1e6 + (dev_ms - dev_ms[0]) * 1e6 * 50e-6
    old = np.array([dev_ms[0] - 30])  # one packet with the clock from before TIME
    d = usb_delay(
        np.concatenate([[host_ns[0]], host_ns]).astype(np.int64),
        np.concatenate([old, dev_ms]).astype(np.uint64),
        int(dev_ms[0]),
    )
    assert d.median_ms == pytest.approx(4.0, abs=0.1)  # drift trend removed
    assert d.p95_ms - d.p5_ms < 0.1
    assert d.drift_ppm == pytest.approx(50, abs=1)
    # Host clock offset from the wall clock (monotonic recordings) is taken from the anchor.
    offset = 1_000_000_000_000
    d2 = usb_delay(
        np.concatenate([[host_ns[0]], host_ns]).astype(np.int64) - offset,
        np.concatenate([old, dev_ms]).astype(np.uint64),
        int(dev_ms[0]),
        wall_minus_host_ns=offset,
    )
    assert d2.median_ms == pytest.approx(d.median_ms)


def test_detect_onsets_measures_known_lag():
    rng = np.random.default_rng(1)
    n = 2000 * 20
    t = T0 + np.arange(n) * 500_000
    emg = rng.normal(size=(n, 18)) * 3
    onsets = T0 + np.array([3, 6.5, 10, 14]) * 1e9
    offsets = onsets + np.array([0.5, 2.0, 1.0, 0.8]) * 1e9
    lag_ns = 40_000_000
    for on, off in zip(onsets, offsets, strict=True):
        burst = (t >= on + lag_ns) & (t < off + lag_ns)
        emg[burst] *= 15
    found = detect_onsets(t, emg_envelope(emg), onsets.astype(np.int64), offsets.astype(np.int64))
    lags_ms = (found - onsets) / 1e6
    assert np.all(found > 0)
    np.testing.assert_allclose(lags_ms, 40, atol=15)


def test_detect_onsets_skips_prompts_without_a_response():
    rng = np.random.default_rng(2)
    t = T0 + np.arange(2000 * 5) * 500_000
    emg = rng.normal(size=(len(t), 18)) * 3
    on = np.array([T0 + 2_000_000_000])
    assert detect_onsets(t, emg_envelope(emg), on, on + 1_000_000_000)[0] == -1


def test_prompter_trial_with_sync_patch(tmp_path):
    import os
    import time

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    seq = Sequence(
        "sync",
        "single",
        [Prompt(0.3, 0.3, ("index",), ("high",)), Prompt(0.9, 0.3, ("ring",), ("low",))],
    )
    app = pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp_path / "s.json", seq)], tail_s=0.3, sync_patch=True)
        win.show()
        win.start_block()
        deadline = time.monotonic() + 5
        while win.running and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        result = session.trials[0]
        win.close()

    assert result.host_clock == "monotonic" and set(result.anchor_start) == {"wall_ns", "host_ns"}
    with np.load(result.npz_path) as data:
        assert str(data["host_clock"]) == "monotonic"
        assert len(data["sync_flash_paint_ns"]) == 2
        data_ref = data["prompt_onset_paint_ns"]
        assert np.all(data_ref >= data["prompt_onset_host_ns"])
        timing, arrays = analyse_trial(data, "sync")
    # The simulated armband runs in real time: on a busy machine (a CI runner) its packets
    # arrive in bursts and frames are drawn late, so a short trial's fitted rate and delays
    # stray. These bounds still catch a broken fit or a wrong unit.
    assert timing.clock.rate_hz == pytest.approx(2000, rel=0.05)
    assert 0 <= timing.paint_delay_ms_median < 500
    np.testing.assert_array_equal(arrays["prompt_ref_ns"], data_ref)
    # Host times are monotonic and the device clock was set from the wall clock; the anchor
    # converts between them, so the delay comes out at a few ms.
    assert -1 < timing.usb.median_ms < 100
    assert timing.n_prompts == 2 and len(arrays["emg_time_ns"]) == result.n_emg


def test_fits_survive_stalls():
    # 20 s of data, then the host sleeps for 30 s (the device sends nothing, its clock runs on),
    # then 20 s more; the sample clock is 15 ppm slow.
    rate = 2000 * (1 - 15e-6)
    true1, host1 = chunked_arrivals(40_000, rate=rate)
    true2, host2 = chunked_arrivals(40_000, rate=rate, seed=1)
    shift = int(50e9)
    host = np.concatenate([host1, host2 + shift])
    seq = np.arange(len(host))  # no lost packets: the counter just paused
    fitted, clock = fit_sample_times(host, seq)
    assert clock.n_stalls == 1 and clock.n_segments == 2
    assert clock.stalled_s == pytest.approx(30, abs=0.1)
    assert clock.rate_error_ppm == pytest.approx(-15, abs=2)
    assert clock.arrival_jitter_ms_p99 < 6
    err_ms = (fitted - np.concatenate([true1, true2 + shift])) / 1e6
    assert np.ptp(err_ms) < 0.5


def test_session_reports_stall(tmp_path):
    import time

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband

    sim = SimulatedArmband()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=sim)) as session:
        session.start_trial(label="stall")
        time.sleep(0.3)
        sim.stall(0.4)
        time.sleep(0.7)
        result = session.end_trial()
    assert result.n_stalls == 1 and result.stalled_s == pytest.approx(0.4, abs=0.1)
    with np.load(result.npz_path) as data:
        timing, _ = analyse_trial(data, "stall")
    assert timing.clock.n_stalls == 1
    assert timing.clock.rate_hz == pytest.approx(2000, rel=0.05)  # real time (the sync test)
