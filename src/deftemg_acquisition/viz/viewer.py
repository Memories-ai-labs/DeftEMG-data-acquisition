"""Live viewer for the armband's LSL streams: raw EMG traces, IMU, and EMG spectrum.

    pixi run viewer                 # in one terminal
    pixi run acquire --simulate     # in another (or with a real --port)

The viewer only reads from LSL, so it can be started, closed, or crash at any time without
affecting acquisition or saving. It reconnects on its own if the acquisition side restarts.

``--compact``: the EMG traces and the band RMS only, with their controls. ``--beside``: laid
out to sit right of the prompter (the band RMS on the left of the EMG, the controls on two
rows); ``--geometry X Y W H``: where its window goes, frame included (``viz.screen``:
``prompter --viewer`` puts it on the right of the screen, the prompter on the left).

Other code can add panels under the band RMS (``panels``: :class:`ViewerPanel`, e.g. another
LSL stream) and its own options (``parser`` and ``run``).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from deftemg_acquisition.acquisition.live import (
    EMG_CHANNELS,
    EMG_STREAM_TYPE,
    IMU_CHANNELS,
    IMU_STREAM_TYPE,
    MARKER_STREAM_TYPE,
    STREAM_PREFIX,
)
from deftemg_acquisition.viz.panels import (
    MAX_WINDOW_S,
    BandRms,
    DisplayControls,
    ImuPlots,
    SpectrumPlot,
    TracePlot,
    auto_spacing,
)

FRAME_MS = 33  # traces and IMU redraw
SLOW_MS = 250  # spectrum and status bar


class Ring:
    """Fixed-capacity buffer of (timestamp, sample) rows; keeps the most recent ones."""

    def __init__(self, capacity: int, n_ch: int):
        self.t = np.zeros(capacity)
        self.x = np.zeros((capacity, n_ch), np.float32)
        self.n = 0  # total rows ever written

    def extend(self, t: np.ndarray, x: np.ndarray) -> None:
        cap = len(self.t)
        t, x = t[-cap:], x[-cap:]
        idx = (self.n + np.arange(len(t))) % cap
        self.t[idx], self.x[idx] = t, x
        self.n += len(t)

    def latest(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        cap = len(self.t)
        k = min(self.n, cap, n if n is not None else cap)
        idx = (self.n - k + np.arange(k)) % cap
        return self.t[idx], self.x[idx]


class Inlet:
    """Finds an LSL stream by type in the background and pulls whatever is available."""

    def __init__(
        self, stype: str, n_ch: int, capacity: int, fmt: str = "numeric",
        flags: int = pylsl.proc_clocksync | pylsl.proc_dejitter,
    ):  # fmt: skip
        """``flags``: LSL processing; leave out ``proc_dejitter`` for an irregular stream."""
        self.flags = flags
        pred = f"type='{stype}' and starts-with(name,'{STREAM_PREFIX}')"
        self._resolver = pylsl.ContinuousResolver(pred=pred)
        self.inlet: pylsl.StreamInlet | None = None
        self.name = ""
        self.fmt = fmt
        self.ring = Ring(capacity, n_ch) if fmt == "numeric" else None
        self._buf = np.zeros((4096, n_ch), np.float32) if fmt == "numeric" else None

    def poll(self) -> list[tuple[float, str]]:
        """Pull new data into the ring; for string streams return the new (time, value)s."""
        if self.inlet is None:
            found = self._resolver.results()
            if not found:
                return []
            self.name = found[0].name()
            self.inlet = pylsl.StreamInlet(found[0], max_buflen=10, processing_flags=self.flags)
        if self.fmt != "numeric":
            samples, ts = self.inlet.pull_chunk(timeout=0.0)
            return [(t, s[0]) for t, s in zip(ts, samples, strict=True)]
        while True:
            _, ts = self.inlet.pull_chunk(
                timeout=0.0, max_samples=len(self._buf), dest_obj=self._buf
            )
            if not ts:
                return []
            self.ring.extend(np.asarray(ts), self._buf[: len(ts)])
            if len(ts) < len(self._buf):
                return []


class ViewerPanel:
    """A panel for the viewer's side column, under the band RMS (``Viewer``'s ``panels``):
    ``widget``, ``height`` its starting height (px), and hooks called each frame."""

    widget: QtWidgets.QWidget
    height: int = 300

    def poll(self) -> None:
        """Every frame, paused or not (e.g. pull new data)."""

    def draw(self, now: float, window_s: float) -> None:
        """Every frame unless paused: ``now`` on the LSL clock, ``window_s`` s shown."""


class Viewer(QtWidgets.QMainWindow):
    def __init__(
        self, window_s: float, spectrum_s: float, arm: str = "right", compact: bool = False,
        beside: bool = False, panels: list[ViewerPanel] | tuple[ViewerPanel, ...] = (),
    ):  # fmt: skip
        """``compact``: the EMG traces and the band RMS only, with their controls (no spectrum,
        IMU or status line). ``beside`` (right of the prompter): the side panels left of the
        EMG traces, the controls on two rows. ``panels``: more side panels, under the band
        RMS."""
        super().__init__()
        self.setWindowTitle("EMG armband – live")
        self.spectrum_s = spectrum_s
        self.compact = compact
        emg_cap = int(MAX_WINDOW_S * 2000 * 1.1)
        self.emg = Inlet(EMG_STREAM_TYPE, len(EMG_CHANNELS), emg_cap)
        self.imu = None
        if not compact:
            self.imu = Inlet(IMU_STREAM_TYPE, len(IMU_CHANNELS), int(MAX_WINDOW_S * 208 * 1.1))
        self.markers = Inlet(MARKER_STREAM_TYPE, 1, 0, fmt="string")
        self.trial_text = "no trial"
        self._last_counts = (0, 0)
        self._rates = (0.0, 0.0)

        pg.setConfigOptions(antialias=False)
        self.controls = DisplayControls(window_s, self.auto_spacing, rows=2 if beside else 1,
                                        spectrum=not compact)  # fmt: skip
        self.status = QtWidgets.QLabel()
        self.traces = TracePlot()
        self.band = BandRms(arm, self.controls)
        self.spectrum = None if compact else SpectrumPlot()
        self.imu_plots = None if compact else ImuPlots()
        self.panels = list(panels)

        side = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        column = [(self.band.widget, 340)] + [(pn.widget, pn.height) for pn in self.panels]
        if not compact:
            column += [(self.spectrum, 190), (self.imu_plots.gyro, 150), (self.imu_plots.acc, 150)]
        for w, _ in column:
            side.addWidget(w)
        side.setSizes([h for _, h in column])
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        if beside:  # next to the prompter: the band RMS on the left
            split.addWidget(side)
            split.addWidget(self.traces)
            split.setSizes([400, 600])
        else:
            split.addWidget(self.traces)
            split.addWidget(side)
            split.setSizes([700, 500])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.controls)
        layout.addWidget(split, 1)
        if not compact:
            layout.addWidget(self.status)
        self.setCentralWidget(central)
        self.resize(1400, 900)

        self._timers = [QtCore.QTimer(self, timeout=self.update_plots, interval=FRAME_MS)]
        if not compact:
            self._timers.append(QtCore.QTimer(self, timeout=self.update_slow, interval=SLOW_MS))
        for timer in self._timers:
            timer.start()

    def closeEvent(self, event) -> None:
        """A closed viewer stops reading its streams (it may outlive its window, e.g. in tests)."""
        for timer in self._timers:
            timer.stop()
        super().closeEvent(event)

    # --- updates -----------------------------------------------------------------------------

    def auto_spacing(self) -> None:
        _, x = self.emg.ring.latest(int(self.controls.window_s * 2000))
        self.controls.spacing_spin.setValue(auto_spacing(x))

    def update_plots(self) -> None:
        self.emg.poll()
        if self.imu is not None:
            self.imu.poll()
        for pn in self.panels:
            pn.poll()
        for t, m in self.markers.poll():
            self.add_marker(t, m)
        if self.controls.pause_box.isChecked():
            return

        now = pylsl.local_clock()
        win = self.controls.window_s
        t, x = self.emg.ring.latest(int(win * 2000))
        self.traces.draw(
            t - now, x, self.controls.spacing, win, self.controls.offset_box.isChecked()
        )
        if self.imu is not None:
            ti, xi = self.imu.ring.latest(int(win * 208))
            self.imu_plots.draw(ti - now, xi[:, 0:3], xi[:, 3:6], win)
        for pn in self.panels:
            pn.draw(now, win)
        self.traces.place_markers(now, keep_s=MAX_WINDOW_S)
        _, xr = self.emg.ring.latest(int(self.band.window_s * 2000))
        self.band.draw(xr)

    def update_slow(self) -> None:
        """Spectrum and status bar, at a few Hz."""
        if not self.controls.pause_box.isChecked():
            _, x = self.emg.ring.latest(int(self.spectrum_s * 2000))
            self.spectrum.draw(x, self.controls.spec_combo.currentIndex())
        n_emg, n_imu = self.emg.ring.n, self.imu.ring.n
        dt = SLOW_MS / 1000
        a = 0.3  # smoothing of the displayed sample rates
        self._rates = tuple(
            (1 - a) * r + a * (n - n0) / dt
            for r, n, n0 in zip(self._rates, (n_emg, n_imu), self._last_counts, strict=True)
        )
        self._last_counts = (n_emg, n_imu)

        parts = []
        if self.emg.inlet is None:
            parts.append("waiting for EMG stream…")
        else:
            parts.append(f"EMG {self._rates[0]:6.0f} Hz")
        if self.imu.inlet is not None and self.imu.ring.n:
            last = self.imu.ring.latest(1)[1][0]
            info = dict(zip(IMU_CHANNELS, last, strict=True))
            parts += [
                f"IMU {self._rates[1]:4.0f} Hz",
                f"battery {info['battery_pct']:.0f}% ({info['battery_mv']:.0f} mV)",
                f"temp {info['temperature_c']:.1f} °C",
                f"dropped packets {info['dropped_total']:.0f}",
            ]
        parts.append(self.trial_text)
        self.status.setText("   |   ".join(parts))

    def add_marker(self, t: float, marker: str) -> None:
        kind, idx, *rest = marker.split("|") + [""]
        if kind not in ("trial_start", "trial_end"):
            return  # e.g. prompt markers from the task UI
        label = rest[0] if kind == "trial_start" else ""
        start = kind == "trial_start"
        color = "#59a14f" if start else "#e15759"
        text = f"trial {idx} {label}".strip() if start else f"end {idx}"
        self.traces.add_marker(t, text, color)
        self.trial_text = f"recording trial {idx} {label}".strip() if start else "no trial"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--window", type=float, default=5.0, help="seconds of data shown")
    ap.add_argument("--spectrum-seconds", type=float, default=2.0, help="data used per spectrum")
    ap.add_argument(
        "--arm", choices=["right", "left"], default="right", help="arm wearing the band"
    )
    ap.add_argument("--compact", action="store_true", help="the EMG traces and band RMS only")
    ap.add_argument(
        "--beside", action="store_true",
        help="right of the prompter: side panels on the left, controls on two rows",
    )  # fmt: skip
    ap.add_argument(
        "--geometry", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
        help="the window's place on the screen, frame included",
    )  # fmt: skip
    return ap


def run(args: argparse.Namespace, panels: list[ViewerPanel] | tuple = ()) -> None:
    """Open the viewer for ``parser()``'s ``args`` and run until it is closed."""
    app = pg.mkQApp("EMG viewer")
    viewer = Viewer(
        args.window,
        args.spectrum_seconds,
        args.arm,
        compact=args.compact,
        beside=args.beside,
        panels=panels,
    )
    viewer.show()
    if args.geometry:
        from deftemg_acquisition.viz.screen import snap

        snap(viewer, QtCore.QRect(*args.geometry))
    sys.exit(app.exec())


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
