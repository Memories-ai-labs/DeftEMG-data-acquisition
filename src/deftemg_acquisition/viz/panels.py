"""Plot panels shared by the live viewer and the replay: EMG traces, spectrum, IMU, controls.

Panels draw whatever data they are given against "time relative to now" (seconds, now = 0),
so the live viewer and the replay look the same.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.signal import welch

from deftemg_acquisition.acquisition.protocol import EMG_RATE_HZ, N_EMG_CHANNELS
from deftemg_acquisition.viz.band import AutoScale, BandRmsWidget, band_rms

EMG_CHANNELS = [f"ch{i}" for i in range(1, N_EMG_CHANNELS + 1)]
AXIS_COLORS = ["#e15759", "#59a14f", "#4e79a7"]  # x, y, z
# One hue per channel, evenly around the colour wheel starting at red for ch1 (orange, yellow,
# green, blue, ... back towards red), matching the vendor software. Shared by traces and spectrum.
CHANNEL_COLORS = [pg.intColor(i, hues=N_EMG_CHANNELS) for i in range(N_EMG_CHANNELS)]
MEAN_COLOR = "#f0f0f0"
MAX_WINDOW_S = 30.0
DEFAULT_SPACING_UV = 300.0  # starting distance between channel traces
DEFAULT_RMS_MAX_UV = 100.0  # band RMS at the top of the colour scale (fixed unless "auto")
SPECTRUM_CHOICES = ["Mean of channels", "All channels", *EMG_CHANNELS]


def samples_per_column(window_s: float, columns: int, rate_hz: float = EMG_RATE_HZ) -> int:
    """Samples in each column of a trace ``columns`` pixels wide showing ``window_s``: from
    the window, not from how many samples a frame happens to have, so it holds still."""
    return max(1, int(window_s * rate_hz) // max(columns, 1))


def peak_envelope(
    t: np.ndarray, x: np.ndarray, per_column: int, first_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``x`` (n, channels) against ``t`` reduced to each column of ``per_column`` samples' min
    then max, for all channels at once (pyqtgraph's "peak" downsampling, without its
    per-curve work).

    ``first_index``: the stream position of ``x[0]`` (samples since the stream began).
    Columns start at multiples of ``per_column`` in the stream, so as the window scrolls each
    column keeps its samples and its picture; only the newest (still filling, at the right
    edge) changes. Columns cut at whatever sample a frame starts on would all shift a little
    every frame, and noisy EMG would shimmer. The oldest partial column is left out.
    Unchanged when ``per_column`` < 2."""
    n, k = len(t), per_column
    if k < 2 or n < 2 * k:
        return t, x
    skip = -first_index % k  # the oldest partial column
    m = (n - skip) // k  # whole columns
    rest = n - skip - m * k  # the newest, still filling
    end = skip + m * k
    tb = t[skip:end].reshape(m, k)
    xb = x[skip:end].reshape(m, k, x.shape[1])
    env = np.empty((m + (rest > 0), 2, x.shape[1]), x.dtype)
    xb.min(axis=1, out=env[:m, 0])
    xb.max(axis=1, out=env[:m, 1])
    tt = np.empty((m + (rest > 0), 2), t.dtype)
    tt[:m, 0], tt[:m, 1] = tb[:, 0], tb[:, -1]  # a column's start and end
    if rest:
        env[m, 0], env[m, 1] = x[end:].min(axis=0), x[end:].max(axis=0)
        tt[m] = t[end], t[-1]
    return tt.reshape(-1), env.reshape(-1, x.shape[1])


SWEEP_CURSOR_COLOR = "#f0f0f0"


def _break_at_wrap(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A NaN row where ``xs`` jumps back (the sweep's wrap), so the line isn't drawn across."""
    wrap = np.flatnonzero(np.diff(xs) < 0)
    if len(wrap):
        i = wrap[0] + 1
        xs = np.insert(xs, i, np.nan)
        ys = np.insert(ys.astype(np.float32, copy=False), i, np.nan, axis=0)
    return xs, ys


def _columns(t: np.ndarray, x: np.ndarray, dt: float):
    """The samples by column of ``dt`` s on a fixed time grid (column ``floor(t / dt)``):
    each column's index and its min and max, all channels at once."""
    col = np.floor(t / dt).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, col[1:] != col[:-1]])
    lo = np.minimum.reduceat(x, starts, axis=0)
    hi = np.maximum.reduceat(x, starts, axis=0)
    return col[starts], lo, hi


def _column_points(left: np.ndarray, lo: np.ndarray, hi: np.ndarray, dt: float):
    """Each column drawn as its min then max, at fixed places inside the column: 0.2 and 0.7
    of it, clear of the half-pixel rounding edges at 100% and 200% scale, so a column moved
    by whole pixels lands on the same pixels (at 0.25, 200% puts it on an edge)."""
    xs = np.stack([left + 0.2 * dt, left + 0.7 * dt], axis=1).reshape(-1)
    return xs, np.stack([lo, hi], axis=1).reshape(-1, lo.shape[1])


def scroll_envelope(
    t: np.ndarray, x: np.ndarray, window_s: float, columns: int, now: float
) -> tuple[np.ndarray, np.ndarray]:
    """A scrolling display of ``x`` (n, channels) at times ``t`` (any clock; ``now`` on it),
    one min and max per column of ``window_s / columns``, returned against time relative to
    now. Columns sit on a fixed time grid and the picture moves in whole columns: each frame
    shifts it by a whole number of pixels when ``columns`` is the plot's width, so nothing is
    rounded differently from one frame to the next (a picture moving by fractions of a pixel
    shimmers, its columns landing on different pixels each frame). The newest column, at the
    right edge, is the one still filling; the right edge trails now by under a column."""
    dt = window_s / max(columns, 1)
    now_col = math.floor(now / dt)
    col, lo, hi = _columns(t, x, dt)
    keep = col > now_col - columns
    return _column_points((col[keep] - now_col - 1) * dt, lo[keep], hi[keep], dt)


def sweep_points(t: np.ndarray, x: np.ndarray, window_s: float):
    """A sweep display of ``x`` (n, channels) at times ``t`` (s, any clock): the screen holds
    ``window_s``, each sample at ``t mod window_s``, and a cursor at the newest sample moves
    left to right, replacing the previous sweep as it goes. Returns (x positions, values with
    a NaN row where the sweep wraps, the cursor's position)."""
    newest = t[-1]
    keep = t > newest - window_s
    xs, ys = _break_at_wrap(np.mod(t[keep], window_s), x[keep])
    return xs, ys, float(newest % window_s)


def sweep_envelope(t: np.ndarray, x: np.ndarray, window_s: float, columns: int):
    """``sweep_points`` reduced to each column's min and max, the columns fixed on the screen
    and in time (``window_s / columns`` each, by the samples' times, so jitter or lost packets
    don't move them): a column is drawn once its samples are in, and stays as it is until the
    sweep comes round again. The cursor's column holds the new sweep only."""
    dt = window_s / max(columns, 1)
    col, lo, hi = _columns(t, x, dt)
    keep = col > col[-1] - columns  # the cursor's column of the last sweep is replaced
    xs, env = _column_points(np.mod(col[keep], columns) * dt, lo[keep], hi[keep], dt)
    xs, env = _break_at_wrap(xs, env)
    return xs, env, float(t[-1] % window_s)


def marker_place(t: float, now: float, sweep: tuple[float, float] | None) -> tuple[float, bool]:
    """Where a marker at time ``t`` goes, and whether it shows: scrolling, at ``t - now``
    once reached; sweeping (``sweep``: window, newest sample's time), where the sweep drew
    it, until the sweep replaces it."""
    if sweep is None:
        return t - now, bool(t <= now)
    window_s, newest = sweep
    return float(t % window_s), bool(newest - window_s < t <= newest)


def auto_spacing(x: np.ndarray) -> float:
    """Trace spacing that fits the typical channel's peaks (99.5th pct of |x|) in its lane."""
    if not len(x):
        return DEFAULT_SPACING_UV
    peak = np.median(np.percentile(np.abs(x - x.mean(axis=0)), 99.5, axis=0))
    return float(max(1.0, 2 * peak))


class DisplayControls(QtWidgets.QWidget):
    """Window length, trace spacing, offset removal, pause, and spectrum channel choice."""

    def __init__(
        self, window_s: float, on_auto_spacing: Callable[[], None], pause: bool = True,
        rows: int = 1, spectrum: bool = True, sweep: bool = False,
    ):  # fmt: skip
        """``rows``: 2 puts the spectrum and RMS settings on a second row (narrow windows);
        ``spectrum``: False leaves out the spectrum's channel choice (no spectrum shown);
        ``sweep``: the live viewer's traces sweep, with a box to scroll them instead
        (``scroll_box``); without it (replay) they scroll."""
        super().__init__()
        self.window_spin = QtWidgets.QDoubleSpinBox(
            minimum=0.5, maximum=MAX_WINDOW_S, value=window_s, singleStep=1.0, suffix=" s"
        )
        self.spacing_spin = QtWidgets.QDoubleSpinBox(
            minimum=1, maximum=100_000, value=DEFAULT_SPACING_UV, singleStep=50, suffix=" µV"
        )
        self.spacing_spin.setToolTip("Vertical distance between channel traces")
        auto_btn = QtWidgets.QPushButton("Auto spacing")
        auto_btn.clicked.connect(on_auto_spacing)
        self.offset_box = QtWidgets.QCheckBox("Remove offset", checked=True)
        self.offset_box.setToolTip("Subtract each channel's mean over the visible window")
        self.pause_box = QtWidgets.QCheckBox("Pause")
        self.scroll_box = QtWidgets.QCheckBox("Scroll")
        self.scroll_box.setToolTip(
            "Traces scroll right to left, the newest at the right edge, instead of standing\n"
            "still while a cursor sweeps left to right at the newest sample (the default)"
        )
        self.spec_combo = QtWidgets.QComboBox()
        self.spec_combo.addItems(SPECTRUM_CHOICES)
        self.rms_window_spin = QtWidgets.QSpinBox(
            minimum=20, maximum=2000, value=200, singleStep=50, suffix=" ms"
        )
        self.rms_window_spin.setToolTip("Window over which the band RMS is computed")
        self.rms_auto_box = QtWidgets.QCheckBox("auto", checked=False)
        self.rms_auto_box.setToolTip("Scale the RMS colours to the most active channel")
        self.rms_max_spin = QtWidgets.QDoubleSpinBox(
            minimum=1, maximum=10_000, value=DEFAULT_RMS_MAX_UV, singleStep=10, suffix=" µV"
        )
        self.rms_max_spin.setToolTip("RMS shown at the top of the colour scale (when not auto)")
        self.rms_auto_box.toggled.connect(lambda on: self.rms_max_spin.setEnabled(not on))

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        layout = QtWidgets.QHBoxLayout()
        outer.addLayout(layout)
        for label, w in [("Window", self.window_spin), ("Spacing", self.spacing_spin)]:
            layout.addWidget(QtWidgets.QLabel(label))
            layout.addWidget(w)
        layout.addWidget(auto_btn)
        layout.addWidget(self.offset_box)
        if pause:
            layout.addWidget(self.pause_box)
        if sweep:
            layout.addWidget(self.scroll_box)
        if rows == 2:
            layout.addStretch(1)
            layout = QtWidgets.QHBoxLayout()
            outer.addLayout(layout)
        else:
            layout.addSpacing(20)
        if spectrum:
            layout.addWidget(QtWidgets.QLabel("Spectrum"))
            layout.addWidget(self.spec_combo)
            layout.addSpacing(20)
        layout.addWidget(QtWidgets.QLabel("RMS"))
        layout.addWidget(self.rms_window_spin)
        layout.addWidget(QtWidgets.QLabel("max"))
        layout.addWidget(self.rms_max_spin)
        layout.addWidget(self.rms_auto_box)
        layout.addStretch(1)

    @property
    def window_s(self) -> float:
        return self.window_spin.value()

    @property
    def spacing(self) -> float:
        return self.spacing_spin.value()


class BandRms:
    """The band RMS widget plus its scaling, fed with the latest EMG window."""

    def __init__(self, arm: str, controls: DisplayControls):
        self.widget = BandRmsWidget(arm, CHANNEL_COLORS)
        self.controls = controls
        self.auto = AutoScale()

    @property
    def window_s(self) -> float:
        return self.controls.rms_window_spin.value() / 1000

    def draw(self, x: np.ndarray) -> None:
        """``x``: the most recent ``window_s`` of EMG, (n, 18) µV."""
        rms = band_rms(x)
        scale = self.auto.update(rms)
        if not self.controls.rms_auto_box.isChecked():
            scale = self.controls.rms_max_spin.value()
        self.widget.set_rms(rms, scale, self.controls.rms_window_spin.value())


class TracePlot(pg.PlotWidget):
    """18 stacked EMG channels plus labelled marker lines: scrolling (time relative to now,
    newest at the right) or sweeping (``draw``'s ``sweep_now``: see ``sweep_envelope``)."""

    def __init__(self):
        super().__init__(title="EMG (µV)")
        self.setLabel("bottom", "time (s)")
        self.showGrid(x=True, alpha=0.2)
        self.getAxis("left").setWidth(44)  # room for "ch18"; labels that don't fit are dropped
        self.curves = []
        for color in CHANNEL_COLORS:
            # draw() reduces the data to the plot's width itself (peak_envelope), for all
            # channels at once; pyqtgraph's own downsampling and clipping would redo it per curve
            c = self.plot(pen=pg.mkPen(color, width=1))
            self.curves.append(c)
        self.cursor = pg.InfiniteLine(0, pen=pg.mkPen(SWEEP_CURSOR_COLOR, width=1))
        self.cursor.setVisible(False)
        self.addItem(self.cursor)
        self.markers: list[tuple[float, pg.InfiniteLine]] = []  # (time, line), caller's timeline
        self._view: tuple | None = None  # (spacing, window_s, sweeping) the axes show
        self._sweep: tuple[float, float] | None = None  # (window_s, newest sample's time)
        # The window and spacing controls set the view (draw() kept resetting a mouse zoom)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()

    def draw(
        self, rel_t: np.ndarray, x: np.ndarray, spacing: float, window_s: float,
        remove_offset: bool, first_index: int = 0, sweep_now: float | None = None,
        now: float | None = None,
    ) -> None:  # fmt: skip
        """``now``: the samples' times are ``rel_t + now``; scrolling then moves in whole
        columns (``scroll_envelope``). Without it, columns follow the stream position of
        ``x[0]``, ``first_index`` (``peak_envelope``). ``sweep_now``: sweep instead of scrolling,
        ``rel_t + sweep_now`` being the samples' times (``sweep_envelope``)."""
        sweeping = sweep_now is not None
        self._sweep = None
        if len(rel_t):
            if remove_offset:
                x = x - x.mean(axis=0)
            # one min and max per column of the plot (logical pixels, as pyqtgraph's own
            # downsampling); 1000 before it is laid out
            columns = int(self.getViewBox().width()) or 1000
            if sweeping:
                t, env, cursor = sweep_envelope(rel_t + sweep_now, x, window_s, columns)
                self._sweep = (window_s, float(rel_t[-1] + sweep_now))
                self.cursor.setValue(cursor)
            elif now is not None:
                t, env = scroll_envelope(rel_t + now, x, window_s, columns, now)
            else:
                k = samples_per_column(window_s, columns)
                t, env = peak_envelope(rel_t, x, k, first_index)
            env = env - np.arange(len(self.curves)) * spacing
            for i, c in enumerate(self.curves):
                if sweeping:  # NaN where the sweep wraps: not joined across the cursor
                    c.setData(t, env[:, i], connect="finite")
                else:
                    c.setData(t, env[:, i], connect="all", skipFiniteCheck=True)
        else:
            for c in self.curves:
                c.setData([], [])
        self.cursor.setVisible(self._sweep is not None)
        # Ticks and ranges only when they change: setting them redraws the axes
        if self._view != (spacing, window_s, sweeping):
            self._view = (spacing, window_s, sweeping)
            n = len(self.curves)
            self.getAxis("left").setTicks([[(-i * spacing, EMG_CHANNELS[i]) for i in range(n)]])
            self.setYRange(-(n - 0.5) * spacing, 0.5 * spacing, padding=0)
            self.setXRange(*((0, window_s) if sweeping else (-window_s, 0)), padding=0)
            self.setLabel("bottom", "time in sweep (s)" if sweeping else "time (s)")

    def add_marker(self, t: float, text: str, color: str, width: float = 2) -> None:
        """A labelled vertical line at time ``t``, shown once ``now`` reaches it."""
        line = pg.InfiniteLine(
            0, pen=pg.mkPen(color, width=width), label=text,
            labelOpts={"position": 0.97, "color": color, "anchors": [(0, 0), (0, 0)]},
        )  # fmt: skip
        self.addItem(line)
        self.markers.append((t, line))

    def place_markers(self, now: float, keep_s: float | None = None) -> None:
        """Position markers relative to ``now`` (sweeping: where the sweep drew them, until it
        clears them); drop those older than ``keep_s``."""
        for t, line in self.markers:
            value, visible = marker_place(t, now, self._sweep)
            line.setValue(value)
            line.setVisible(visible)
        if keep_s is not None:
            for item in [m for m in self.markers if m[0] < now - keep_s]:
                self.removeItem(item[1])
                self.markers.remove(item)


class SpectrumPlot(pg.PlotWidget):
    """Welch spectrum of the mean, all, or one EMG channel, in the trace colours."""

    def __init__(self):
        super().__init__(title="EMG spectrum (µV²/Hz)")
        self.setLogMode(y=True)
        self.setLabel("bottom", "frequency (Hz)")
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setXRange(0, EMG_RATE_HZ / 2)
        for f in (60, 120, 180):
            self.addItem(pg.InfiniteLine(f, pen=pg.mkPen("#888", style=QtCore.Qt.DashLine)))
        self.curves = [self.plot(pen=pg.mkPen(color, width=1)) for color in CHANNEL_COLORS]
        self.mean_curve = self.plot(pen=pg.mkPen(MEAN_COLOR, width=2))

    def draw(self, x: np.ndarray, choice: int) -> None:
        """``choice``: index into SPECTRUM_CHOICES."""
        if len(x) < 512:
            return
        f, pxx = welch(x, fs=EMG_RATE_HZ, nperseg=512, axis=0)  # pxx: (n_freq, n_ch)
        if choice == 0:
            self.mean_curve.setData(f[1:], np.maximum(pxx.mean(axis=1)[1:], 1e-12))
            self.mean_curve.show()
            shown = {}
        else:
            self.mean_curve.hide()
            channels = range(pxx.shape[1]) if choice == 1 else [choice - 2]
            shown = {i: pxx[:, i] for i in channels}  # each channel keeps its trace colour
        for i, c in enumerate(self.curves):
            if i in shown:
                c.setData(f[1:], np.maximum(shown[i][1:], 1e-12))
                c.show()
            else:
                c.hide()


class ImuPlots:
    """Gyro and accelerometer plots (x, y, z) on the same time window as the traces."""

    def __init__(self):
        self.gyro = pg.PlotWidget(title="Gyro (rad/s)")
        self.acc = pg.PlotWidget(title="Accelerometer (m/s²)")
        self.curves = []
        self._view: tuple[float, bool] | None = None  # (window_s, sweeping) the axes show
        for plot in (self.gyro, self.acc):
            plot.addLegend(offset=(5, 5))
            plot.showGrid(x=True, y=True, alpha=0.2)
            for axis, color in zip("xyz", AXIS_COLORS, strict=True):
                # 1 px: wider lines leave Qt's fast path (1.5 px cost a fifth of a viewer frame)
                self.curves.append(plot.plot(pen=pg.mkPen(color, width=1), name=axis))

    def draw(
        self, rel_t: np.ndarray, gyro: np.ndarray, acc: np.ndarray, window_s: float,
        sweep_now: float | None = None,
    ) -> None:  # fmt: skip
        """``sweep_now``: sweep as the EMG traces do (``TracePlot.draw``)."""
        sweeping = sweep_now is not None and len(rel_t) > 1
        if sweeping:
            xs, ys, _ = sweep_points(rel_t + sweep_now, np.hstack([gyro, acc]), window_s)
            for k, c in enumerate(self.curves):
                c.setData(xs, ys[:, k], connect="finite")
        else:
            for k, c in enumerate(self.curves):
                c.setData(rel_t, gyro[:, k] if k < 3 else acc[:, k - 3], connect="all")
        if self._view != (window_s, sweeping):  # setting the range redraws the axes
            self._view = (window_s, sweeping)
            for plot in (self.gyro, self.acc):
                plot.setXRange(*((0, window_s) if sweeping else (-window_s, 0)), padding=0)
