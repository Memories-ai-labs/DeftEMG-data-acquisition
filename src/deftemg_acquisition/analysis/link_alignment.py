"""Offline alignment of a wireless link: every band packet on the computer's clock.

A wireless link (band -> bridge -> Wi-Fi -> dongle -> computer) adds delay and jitter that grow
with the link's load, so arrival time is a poor time axis. The link therefore inserts timing
packets (``protocol.decode_markers``) carrying its Wi-Fi clock, which the bridge and the dongle of
one pair share. Aligned offline, per recording and per band, from the saved ``marker_*`` arrays:

1. Split into segments wherever the link restarted (a marker number, a band packet count or a
   clock goes backwards). Each segment is fitted on its own.
2. Number every band packet. A marker holds the link's count of band packets before it, which is
   the index of the packet after it. The sequence numbers carry that through the packets until
   the next marker and place the ones lost on the way to the computer; the next marker's count
   says how many were lost, which settles a loss longer than the sequence counter wraps over.
3. Fit packet index to bridge time, on the lower edge of the points: delay only ever adds, so
   the earliest arrivals follow the band's own clock.
4. Bridge time is dongle time (the two clocks are locked).
5. Fit dongle time to computer time (when the marker was read), again on the lower edge.
6. Compose the fits: every packet gets a computer time, the same clock for every band.

The remaining error is the scatter of each fit (see :attr:`SegmentFit.resid_a_us` and
``resid_b_us``) and a delay that is the same for bands on the same link design, which cancels
between them. Bands recorded together share the computer's clock, so their packets can be
resampled onto one grid from the times returned here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

MIN_MARKERS = 4  # per segment, to fit its lines
KEEP_FRACTION = 0.3
FIT_ROUNDS = 4
SEQ_MOD = 256


@dataclass(frozen=True)
class Line:
    """``y = y0 + slope * (x - x0)``; centred so large clock values keep their precision."""

    x0: float
    y0: float
    slope: float

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        return self.y0 + self.slope * (np.asarray(x, dtype=np.float64) - self.x0)


def lower_fit(
    x: np.ndarray, y: np.ndarray, keep: float = KEEP_FRACTION, rounds: int = FIT_ROUNDS
) -> Line:
    """Straight line through the lower edge of the points.

    Fits all points, keeps the ``keep`` fraction with the lowest residuals and refits, ``rounds``
    times, then shifts the line down to the lowest point.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        raise ValueError("need at least two points to fit a line")
    xs, ys = x - x[0], y - y[0]
    k = max(2, int(keep * x.size))
    sel = np.ones(x.size, bool)
    for _ in range(rounds):
        b, a = np.polyfit(xs[sel], ys[sel], 1)
        r = ys - (a + b * xs)
        sel = r <= np.partition(r, k - 1)[k - 1]
    b, a = np.polyfit(xs[sel], ys[sel], 1)
    a += float(np.min(ys - (a + b * xs)))
    return Line(float(x[0]), float(y[0] + a), float(b))


@dataclass
class SegmentFit:
    """One stretch of a recording between link restarts, with its fits."""

    first_marker: int
    n_markers: int
    a: Line | None  # packet index -> bridge time (us)
    b: Line | None  # dongle time (us) -> computer time (ns)
    resid_a_us: tuple[float, float] = (np.nan, np.nan)  # p50, p95 of the points above the line
    resid_b_us: tuple[float, float] = (np.nan, np.nan)

    @property
    def packet_period_us(self) -> float:
        return float(self.a.slope) if self.a else np.nan

    @property
    def link_drift_ppm(self) -> float:
        """Link clock against the computer's, in ppm (dongle us -> computer ns, slope 1000)."""
        return (self.b.slope / 1000.0 - 1.0) * 1e6 if self.b else np.nan


@dataclass
class Alignment:
    """Computer time (ns, the clock of ``marker_host_time_ns``) of every packet, EMG and IMU."""

    emg_t_ns: np.ndarray  # int64; 0 where not valid
    imu_t_ns: np.ndarray
    emg_valid: np.ndarray  # bool: placed by a fitted segment
    imu_valid: np.ndarray
    emg_index: np.ndarray  # int64 band packet index (shared count, AA and BB)
    imu_index: np.ndarray
    segments: list[SegmentFit] = field(default_factory=list)
    n_unplaced_intervals: int = 0  # marker intervals whose numbering disagreed with the next one

    def summary(self) -> str:
        lines = []
        for i, s in enumerate(self.segments):
            if s.a is None or s.b is None:
                lines.append(f"  segment {i}: {s.n_markers} markers, too few to fit")
                continue
            lines.append(
                f"  segment {i}: {s.n_markers} markers, packet period {s.packet_period_us:.2f} us, "
                f"link clock {s.link_drift_ppm:+.0f} ppm vs computer; scatter p50/p95: "
                f"band->bridge {s.resid_a_us[0] / 1e3:.2f}/{s.resid_a_us[1] / 1e3:.2f} ms, "
                f"dongle->computer {s.resid_b_us[0] / 1e3:.2f}/{s.resid_b_us[1] / 1e3:.2f} ms"
            )
        n = self.emg_valid.size + self.imu_valid.size
        ok = int(self.emg_valid.sum() + self.imu_valid.sum())
        lines.append(
            f"  placed {ok} of {n} packets; {self.n_unplaced_intervals} intervals unplaced"
        )
        return "\n".join(lines)


def segment_ids(marker_no, band_count, bridge_tsf_us, dongle_tsf_us) -> np.ndarray:
    """Segment number of each marker: a new one wherever the link restarted."""
    n = len(marker_no)
    if n == 0:
        return np.empty(0, np.int64)
    back = (
        (np.diff(marker_no) < 0)
        | (np.diff(band_count) < 0)
        | (np.diff(bridge_tsf_us) < 0)
        | (np.diff(dongle_tsf_us) < 0)
    )
    return np.concatenate([[0], np.cumsum(back)]).astype(np.int64)


def _fit_gaps(gaps: np.ndarray, lost: int) -> np.ndarray:
    """Make the sequence-number gaps of an interval add up to the ``lost`` its markers imply.

    The next marker's count is the better record. Too many gaps (packets the link never saw)
    come off the biggest first; too few (a loss longer than the counter wraps over) go on the
    biggest.
    """
    gaps = gaps.astype(np.int64).copy()
    excess = int(gaps.sum()) - lost
    if excess > 0:
        for k in np.argsort(-gaps, kind="stable"):
            take = min(excess, int(gaps[k]))
            gaps[k] -= take
            excess -= take
            if excess == 0:
                break
    elif excess < 0:
        gaps[int(np.argmax(gaps))] -= excess
    return gaps


def _percentiles(resid: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(resid, 50)), float(np.percentile(resid, 95))


def align_link_trial(d: Mapping[str, np.ndarray]) -> Alignment:
    """Put the packets of a trial recorded over a wireless link on the computer's clock.

    ``d`` is a trial's arrays (the ``.npz``, or ``decode`` plus the recorder's host times):
    ``emg_seq``, ``emg_offset``, ``imu_seq``, ``imu_offset`` and the ``marker_*`` arrays. The link
    counts packets it saw, so a packet lost before it (band or USB) is not numbered, which
    shifts the later ones by one packet period (0.45 ms) each.
    """
    emg_off, imu_off = np.asarray(d["emg_offset"]), np.asarray(d["imu_offset"])
    n_emg = emg_off.size
    off = np.concatenate([emg_off, imu_off])
    useq = np.concatenate([np.asarray(d["emg_seq"]), np.asarray(d["imu_seq"])]).astype(np.int64)
    order = np.argsort(off, kind="stable")
    off, useq = off[order], useq[order]
    sn = useq % SEQ_MOD

    m = {k: np.asarray(d[f"marker_{k}"]) for k in (
        "offset", "band_count", "prev_seq", "bridge_tsf_us", "dongle_tsf_us", "no", "host_time_ns",
    )}  # fmt: skip
    seg = segment_ids(m["no"], m["band_count"], m["bridge_tsf_us"], m["dongle_tsf_us"])
    midx = m["band_count"].astype(np.int64)  # index of the packet after each marker

    # fits per segment
    fits: list[SegmentFit] = []
    for s in np.unique(seg):
        sel = np.flatnonzero(seg == s)
        sf = SegmentFit(int(sel[0]), int(sel.size), None, None)
        if sel.size >= MIN_MARKERS:
            sf.a = lower_fit(midx[sel], m["bridge_tsf_us"][sel])
            sf.b = lower_fit(m["dongle_tsf_us"][sel], m["host_time_ns"][sel])
            sf.resid_a_us = _percentiles(m["bridge_tsf_us"][sel] - sf.a(midx[sel]))
            sf.resid_b_us = _percentiles(
                (m["host_time_ns"][sel] - sf.b(m["dongle_tsf_us"][sel])) / 1e3
            )
        fits.append(sf)

    # number the packets: each interval after a marker is anchored by that marker
    n_pkt = off.size
    idx = np.zeros(n_pkt, np.int64)
    packet_seg = np.full(n_pkt, -1, np.int64)
    n_unplaced = 0
    if len(m["offset"]) and n_pkt:
        j = np.searchsorted(m["offset"], off) - 1  # preceding marker of each packet, -1 if none
        bounds = np.flatnonzero(np.diff(j, prepend=-2) != 0)
        ends = np.append(bounds[1:], n_pkt)
        starts_of = {int(j[b]): (int(b), int(e)) for b, e in zip(bounds, ends, strict=True)}
        for jj, (b, e) in starts_of.items():
            if jj < 0:
                continue
            # gaps in the sequence numbers: packets lost before each packet, the first against
            # the marker's own record of the packet before it
            gaps = np.diff(sn[b:e], prepend=int(m["prev_seq"][jj])) % SEQ_MOD
            gaps = np.where(gaps == 0, SEQ_MOD, gaps) - 1  # a repeat reads as a full wrap
            same = jj + 1 < len(seg) and seg[jj + 1] == seg[jj]
            if same:
                # the next marker's count says how many packets this interval held in all
                lost = int(midx[jj + 1] - midx[jj]) - (e - b)
                if lost < 0:
                    packet_seg[b:e] = -1
                    n_unplaced += 1
                    continue
                gaps = _fit_gaps(gaps, lost)
            idx[b:e] = midx[jj] + np.arange(e - b) + np.cumsum(gaps)
            packet_seg[b:e] = seg[jj]
        if -1 in starts_of and 0 in starts_of:  # before the first marker: counted back from it
            b, e = starts_of[-1]
            f = starts_of[0][0]
            idx[b:e] = idx[f] - (useq[f] - useq[b:e])
            packet_seg[b:e] = seg[0]
        # packets of a new connection before its first marker cannot be numbered
        for jj, (b, e) in starts_of.items():
            if 0 <= jj < len(seg) - 1 and seg[jj + 1] != seg[jj]:
                packet_seg[b:e] = -1

    t_ns = np.zeros(n_pkt, np.int64)
    valid = np.zeros(n_pkt, bool)
    for s, sf in enumerate(fits):
        if sf.a is None or sf.b is None:
            continue
        sel = np.flatnonzero(packet_seg == s)
        if sel.size:
            t_ns[sel] = np.rint(sf.b(sf.a(idx[sel]))).astype(np.int64)
            valid[sel] = True

    inv = np.empty(n_pkt, np.int64)
    inv[order] = np.arange(n_pkt)  # back to the order of emg then imu
    t_ns, valid, idx = t_ns[inv], valid[inv], idx[inv]
    return Alignment(
        emg_t_ns=t_ns[:n_emg], imu_t_ns=t_ns[n_emg:],
        emg_valid=valid[:n_emg], imu_valid=valid[n_emg:],
        emg_index=idx[:n_emg], imu_index=idx[n_emg:],
        segments=fits, n_unplaced_intervals=n_unplaced,
    )  # fmt: skip
