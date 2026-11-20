import numpy as np
import pytest

from deftemg_acquisition.acquisition import protocol as P
from deftemg_acquisition.analysis.link_alignment import (
    Line,
    align_link_trial,
    lower_fit,
    segment_ids,
)

PERIOD_US = 451.0
BRIDGE_PPM, LINK_PPM = 20e-6, 8e-6
HOST0_NS = 1_000_000_000_000
LATENCY_US = 300.0  # bridge to dongle: a constant, the same on every band
MARKER_EVERY = 111


def raw_packet(kind, seq):
    body = bytearray(59 - P.HEADER_LEN)
    return P.SYNC + bytes([kind, seq & 0xFF]) + bytes(body)


def marker(seq, *, count, lost, bridge_us, dongle_us, no, prev_type):
    b = bytearray(59 - P.HEADER_LEN)
    b[0:2], b[2], b[3], b[4] = b"SY", 3, 1, 59
    b[5:13] = int(bridge_us).to_bytes(8, "little", signed=True)
    b[13:17] = int(count).to_bytes(4, "little")
    b[17] = prev_type
    b[18:26] = int(dongle_us).to_bytes(8, "little", signed=True)
    b[26:30] = int(no).to_bytes(4, "little")
    b[38:40] = int(lost).to_bytes(2, "little")  # packet bytes 43-44: packets lost before the bridge
    return P.SYNC + bytes([P.TYPE_MARK, seq & 0xFF]) + bytes(b)


def make_link(
    n_packets,
    *,
    seed=0,
    band_lost=(),
    wifi_lost=(),
    marker_lost=(),
    no0=0,
    delay_scale=1.0,
    t0_us=5e6,
):
    """A raw-mode stream as the computer would read it, and the true host time of each packet.

    ``band_lost``: packet indices the bridge never saw. ``wifi_lost``: indices lost after it.
    ``marker_lost``: marker numbers lost after it. Returns bytes, marker host times, truth.
    """
    rng = np.random.default_rng(seed)
    band_lost, wifi_lost, marker_lost = set(band_lost), set(wifi_lost), set(marker_lost)

    def bridge_us(i):
        return t0_us + i * PERIOD_US * (1 + BRIDGE_PPM)

    def host_ns(dongle_us):
        return HOST0_NS + dongle_us * 1e3 * (1 + LINK_PPM)

    stream, mk_host, truth = bytearray(), [], {}
    seen = lost_before = no = 0
    for i in range(n_packets):
        if i % MARKER_EVERY == 0 and i > 0:
            b_us = bridge_us(i) + rng.exponential(600 * delay_scale)
            d_us = b_us + LATENCY_US
            if no0 + no not in marker_lost:
                stream += marker(
                    i - 1, count=seen, lost=lost_before, bridge_us=b_us, dongle_us=d_us,
                    no=no0 + no, prev_type=P.TYPE_EMG,
                )  # fmt: skip
                mk_host.append(host_ns(d_us) + rng.exponential(400e3 * delay_scale))
            no += 1
        if i in band_lost:
            lost_before += 1
            continue
        seen += 1
        truth[i] = host_ns(bridge_us(i) + LATENCY_US)
        if i in wifi_lost:
            continue
        stream += raw_packet(P.TYPE_IMU if i % 10 == 9 else P.TYPE_EMG, i)
    return bytes(stream), np.array(mk_host), truth


def trial_arrays(stream, mk_host):
    d = P.decode(stream, filter_enabled=False)
    arrays = {k: v for k, v in d.as_dict().items() if not isinstance(v, int)}
    arrays["marker_host_time_ns"] = np.asarray(mk_host, np.int64)
    return d, arrays


def test_lower_fit_follows_the_lower_edge():
    rng = np.random.default_rng(1)
    x = np.arange(400.0)
    y = 7.0 + 2.5 * x + rng.exponential(3.0, x.size)
    line = lower_fit(x, y)
    assert isinstance(line, Line)
    assert line.slope == pytest.approx(2.5, rel=2e-3)
    assert np.all(y - line(x) >= -1e-6)  # nothing below it
    assert np.median(y - line(x)) < 3.0  # and it is close to the floor, not through the middle


def test_segments_split_where_the_link_restarts():
    seg = segment_ids(
        np.array([5, 6, 7, 0, 1]), np.array([9, 10, 11, 0, 1]), np.arange(5), np.arange(5)
    )
    assert seg.tolist() == [0, 0, 0, 1, 1]
    assert segment_ids(np.array([]), np.array([]), np.array([]), np.array([])).size == 0


def check_alignment(d, arrays, truth, *, max_err_ms=1.0):
    al = align_link_trial(arrays)
    for kind, t_ns, valid, idx in (
        ("emg", al.emg_t_ns, al.emg_valid, al.emg_index),
        ("imu", al.imu_t_ns, al.imu_valid, al.imu_index),
    ):
        assert valid.all(), kind
        want = np.array([truth[int(i)] for i in idx])  # truth is keyed by true packet index
        err = (t_ns - want) / 1e6
        # the lower edge sits a little under the typical delay, the same for every packet
        assert np.ptp(err) < max_err_ms, (kind, np.ptp(err))
        assert abs(np.median(err)) < 2.0
    return al


def test_alignment_recovers_packet_times_through_loss_and_drift():
    n = 6000
    stream, mk_host, truth = make_link(
        n,
        wifi_lost=set(range(1000, 1010)) | {3000, 4001},
    )
    d, arrays = trial_arrays(stream, mk_host)
    # the packets' own true index: the test stream's SN is i % 256, so rebuild from the numbering
    al = align_link_trial(arrays)
    assert al.emg_valid.all() and al.imu_valid.all()
    # packet i of the generator has sn i % 256; check indices are what the generator used
    idx = np.concatenate([al.emg_index, al.imu_index])
    sn = np.concatenate([d.emg_seq, d.imu_seq]) % 256
    assert np.array_equal(idx % 256, sn)
    check_alignment(d, arrays, truth)
    seg = al.segments[0]
    assert seg.packet_period_us == pytest.approx(PERIOD_US * (1 + BRIDGE_PPM), rel=1e-4)
    # 2.7 s of 0.4 ms read jitter cannot pin the link clock better than ~100 ppm
    assert seg.link_drift_ppm == pytest.approx(LINK_PPM * 1e6, abs=300)
    assert al.n_unplaced_intervals == 0


def test_alignment_survives_an_outage_longer_than_the_sequence_wraps():
    # 600 packets and the markers among them lost: sequence numbers alone cannot count them
    lost = set(range(2000, 2600))
    stream, mk_host, truth = make_link(6000, wifi_lost=lost, marker_lost={18, 19, 20, 21, 22})
    d, arrays = trial_arrays(stream, mk_host)
    check_alignment(d, arrays, truth)


def test_alignment_splits_and_places_each_segment_after_a_restart():
    s1, h1, t1 = make_link(3000, seed=1)
    # the link restarts 2 s later: its counters start again, its clocks keep going on
    s2, h2, t2 = make_link(3000, seed=2, t0_us=5e6 + 3000 * PERIOD_US + 2e6)
    stream = s1 + s2
    d, arrays = trial_arrays(stream, np.concatenate([h1, h2]))
    al = align_link_trial(arrays)
    assert len(al.segments) == 2  # the marker numbers start again
    t, ok = al.emg_t_ns, al.emg_valid
    assert ok.mean() > 0.95  # only the packets before the new connection's first marker are lost
    assert np.all(np.diff(t[ok]) > 0)
    # both stretches on one clock: the second starts about 2 s after the first ends
    first_end = t[ok][np.flatnonzero(al.emg_index[ok] > 2900)[0]]
    second_start = t[ok][-1] - 2998 * PERIOD_US * 1e3 * 0.9  # inside the second stretch
    assert second_start > first_end


def test_align_link_cli_decodes_older_trials_and_writes_times(tmp_path, monkeypatch):
    import json

    from deftemg_acquisition.acquisition.recorder import TrialRecorder
    from deftemg_acquisition.cli import align_link

    stream, mk_host, _ = make_link(2000)
    session = tmp_path / "raw" / "s1"
    rec = TrialRecorder()
    rec(1, stream[:30000])
    rec(2, stream[30000:])
    files = rec.save(session / "trial_001", filter_enabled=False)
    # an older trial: no marker arrays in the .npz, the .bin and chunk table are enough
    z = dict(np.load(files.npz))
    np.savez(files.npz, **{k: v for k, v in z.items() if not k.startswith("marker_")})
    (session / "trials.jsonl").write_text(json.dumps({"npz_path": "x/trial_001.npz"}) + "\n")
    (session / "session.json").write_text(json.dumps({"filter_status": {"enabled": False}}))
    arrays = align_link.trial_arrays(files.npz, filter_enabled=False)
    assert len(arrays["marker_no"]) > 10 and arrays["marker_host_time_ns"].tolist()[:1] == [1]
    monkeypatch.setattr(
        "sys.argv", ["align-link", str(session), "--out-dir", str(tmp_path / "processed")]
    )
    align_link.main()
    out = np.load(tmp_path / "processed" / "s1" / "trial_001_aligned.npz")
    assert out["emg_t_ns"].size == len(z["emg_seq"]) and out["emg_valid"].any()
