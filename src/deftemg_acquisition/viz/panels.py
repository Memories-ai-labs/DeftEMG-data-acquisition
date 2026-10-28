"""Plot panels shared by the live viewer and the replay: EMG traces, spectrum, IMU, controls.

Panels draw whatever data they are given against "time relative to now" (seconds, now = 0),
so the live viewer and the replay look the same.
"""

from __future__ import annotations

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
        rows: int = 1, spectrum: bool = True,
    ):  # fmt: skip
        """``rows``: 2 puts the spectrum and RMS settings on a second row (narrow windows);
        ``spectrum``: False leaves out the spectrum's channel choice (no spectrum shown)."""
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
    """18 stacked EMG channels against time relative to now, plus labelled marker lines."""

    def __init__(self):
        super().__init__(title="EMG (µV)")
        self.setLabel("bottom", "time (s)")
        self.showGrid(x=True, alpha=0.2)
        self.getAxis("left").setWidth(44)  # room for "ch18"; labels that don't fit are dropped
        self.curves = []
        for color in CHANNEL_COLORS:
            c = self.plot(pen=pg.mkPen(color, width=1))
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
            self.curves.append(c)
        self.markers: list[tuple[float, pg.InfiniteLine]] = []  # (time, line), caller's timeline

    def draw(
        self, rel_t: np.ndarray, x: np.ndarray, spacing: float, window_s: float, remove_offset: bool
    ) -> None:
        if len(rel_t):
            if remove_offset:
                x = x - x.mean(axis=0)
            for i, c in enumerate(self.curves):
                c.setData(rel_t, x[:, i] - i * spacing)
        else:
            for c in self.curves:
                c.setData([], [])
        n = len(self.curves)
        self.getAxis("left").setTicks([[(-i * spacing, EMG_CHANNELS[i]) for i in range(n)]])
        self.setYRange(-(n - 0.5) * spacing, 0.5 * spacing, padding=0)
        self.setXRange(-window_s, 0, padding=0)

    def add_marker(self, t: float, text: str, color: str, width: float = 2) -> None:
        """A labelled vertical line at time ``t``, shown once ``now`` reaches it."""
        line = pg.InfiniteLine(
            0, pen=pg.mkPen(color, width=width), label=text,
            labelOpts={"position": 0.97, "color": color, "anchors": [(0, 0), (0, 0)]},
        )  # fmt: skip
        self.addItem(line)
        self.markers.append((t, line))

    def place_markers(self, now: float, keep_s: float | None = None) -> None:
        """Position markers relative to ``now``; drop those older than ``keep_s``."""
        for t, line in self.markers:
            line.setValue(t - now)
            line.setVisible(bool(t <= now))
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
        for plot in (self.gyro, self.acc):
            plot.addLegend(offset=(5, 5))
            plot.showGrid(x=True, y=True, alpha=0.2)
            for axis, color in zip("xyz", AXIS_COLORS, strict=True):
                self.curves.append(plot.plot(pen=pg.mkPen(color, width=1.5), name=axis))

    def draw(self, rel_t: np.ndarray, gyro: np.ndarray, acc: np.ndarray, window_s: float) -> None:
        for k, c in enumerate(self.curves):
            c.setData(rel_t, gyro[:, k] if k < 3 else acc[:, k - 3])
        for plot in (self.gyro, self.acc):
            plot.setXRange(-window_s, 0, padding=0)
