"""Live viewer for the armband's LSL streams: raw EMG traces, IMU, and EMG spectrum.

    pixi run viewer                 # in one terminal
    pixi run acquire --simulate     # in another (or with a real --port)

The viewer only reads from LSL, so it can be started, closed, or crash at any time without
affecting acquisition or saving. It reconnects on its own if the acquisition side restarts.

``--compact``: the EMG traces and the band RMS only, with their controls. ``--beside``: laid
out to sit right of the prompter (the band RMS on the left of the EMG, the controls on two
rows); ``--geometry X Y W H``: where its window goes, frame included (``viz.screen``:
``prompter --viewer`` puts it on the right of the screen, the prompter on the left).

``--source``: the session to show when several publish (on this computer and others on the
network), by its ``source_id`` or its computer's host name; without it, one on this computer.

Other code can add panels under the band RMS (``panels``: :class:`ViewerPanel`, e.g. another
LSL stream) and its own options (``parser`` and ``run``).
"""

from __future__ import annotations

import argparse
import socket
import sys
import time

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
from deftemg_acquisition.acquisition.protocol import EMG_RATE_HZ, IMU_RATE_HZ
from deftemg_acquisition.viz.gpu_traces import GpuTracePlot, make_trace_plot
from deftemg_acquisition.viz.panels import (
    MAX_WINDOW_S,
    BandRms,
    DisplayControls,
    ImuPlots,
    SpectrumPlot,
    auto_spacing,
    legend_text_size,
    text_style,
)
from deftemg_acquisition.viz.screen import fit_font, font_points

WATCHDOG_MS = 100  # with GPU traces, frames follow the screen (Viewer._frame_shown)
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

    def last_seconds(self, seconds: float, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
        """The rows of the last ``seconds`` by their timestamps (up to the newest row's).
        By time, not by ``seconds * rate_hz`` rows: a device running faster than its nominal
        rate (the IMU sends ~217 packets/s, not 208) would leave the window short, and a sweep
        would show the shortfall as a gap after its cursor."""
        t, x = self.latest(int(seconds * rate_hz * 1.25) + 16)
        if not len(t):
            return t, x
        first = np.searchsorted(t, t[-1] - seconds, side="right")
        return t[first:], x[first:]

    def latest(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        cap = len(self.t)
        k = min(self.n, cap, n if n is not None else cap)
        idx = (self.n - k + np.arange(k)) % cap
        return self.t[idx], self.x[idx]


# The session whose streams the viewer reads (``--source``, ``select_source``): its source_id or
# the host name of the computer it runs on; None: any (one on this computer first). Then the
# same one (``_chosen``) for every stream, so EMG, IMU, markers and panels all match.
_source: str | None = None
_chosen: str | None = None  # the source_id the first stream found came from


def select_source(source: str | None) -> None:
    """Read the streams of ``source`` only (a session's ``source_id``, or a host name); None:
    any session's. Applies to every :class:`Inlet` not yet connected, made before or after."""
    global _source, _chosen
    _source, _chosen = source, None


def _pick(found: list[pylsl.StreamInfo]) -> pylsl.StreamInfo | None:
    """Of the streams ``found`` (one type), the one to read: from the selected session."""
    global _chosen
    if _source:
        found = [s for s in found if _source in (s.source_id(), s.hostname())]
    if _chosen is not None:
        found = [s for s in found if s.source_id() == _chosen]
    if not found:
        return None
    here = socket.gethostname()
    found = sorted(found, key=lambda s: s.hostname() != here)  # this computer's first
    if _chosen is None:
        sources = sorted({f"{s.source_id()} on {s.hostname()}" for s in found})
        if len(sources) > 1:
            print(f"viewer: streams from {', '.join(sources)}; showing {found[0].source_id()} "
                  "(--source picks one)", file=sys.stderr)  # fmt: skip
        _chosen = found[0].source_id()
    return found[0]


class Inlet:
    """Finds an LSL stream by type in the background and pulls whatever is available.

    Of the session given to :func:`select_source` (``--source``); without one, of the first
    session found (this computer's first), the same one for every inlet."""

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
            info = _pick(found) if found else None
            if info is None:
                return []
            self.name = info.name()
            self.inlet = pylsl.StreamInlet(info, max_buflen=10, processing_flags=self.flags)
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
    # Set by the viewer before each draw: whether the EMG traces sweep (the "Sweep" box) rather
    # than scroll; a panel showing time can follow it (time in the sweep: ``now`` mod
    # ``window_s``, the cursor at ``now``; see ``panels.sweep_points``)
    sweep: bool = False

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
        # room for the longest window from a device running fast (Ring.last_seconds)
        emg_cap = int(MAX_WINDOW_S * EMG_RATE_HZ * 1.3)
        self.emg = Inlet(EMG_STREAM_TYPE, len(EMG_CHANNELS), emg_cap)
        self.imu = None
        if not compact:
            self.imu = Inlet(
                IMU_STREAM_TYPE, len(IMU_CHANNELS), int(MAX_WINDOW_S * IMU_RATE_HZ * 1.3)
            )
        self.markers = Inlet(MARKER_STREAM_TYPE, 1, 0, fmt="string")
        self.trial_text = "no trial"
        self._last_counts = (0, 0)
        self._rates = (0.0, 0.0)

        pg.setConfigOptions(antialias=False)
        self.controls = DisplayControls(window_s, self.auto_spacing, rows=2 if beside else 1,
                                        spectrum=not compact, sweep=True)  # fmt: skip
        self.status = QtWidgets.QLabel()
        self.traces = make_trace_plot()  # drawn by the GPU when there is one
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
        # Open at 90% of the screen's available area, centred (run's --geometry places it
        # instead); the text follows the window's height as it is resized (_refit_text)
        area = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(area.width() * 0.9), int(area.height() * 0.9))
        self.move(area.center() - self.rect().center())
        self._refit = QtCore.QTimer(self, singleShot=True, interval=150, timeout=self._refit_text)

        # GPU traces: a frame as soon as the last one is on the screen (frameSwapped), so frames
        # follow the screen's refresh; a 16 ms timer against a 60 Hz screen showed a frame
        # twice (or skipped one) every ~25 frames, a visible hitch. The timer then only keeps
        # the streams read while no frames are shown (window hidden or minimised).
        self._vsync = isinstance(self.traces, GpuTracePlot)
        self._last_frame = 0.0
        if self._vsync:
            self.traces.canvas.frameSwapped.connect(self._frame_shown)
            frames = QtCore.QTimer(self, timeout=self._watchdog, interval=WATCHDOG_MS)
        else:
            # CPU traces: frames paced to the screen's refresh, each due 1/rate after the last
            # (a whole-ms interval drifts against the screen as 16 ms does against 60 Hz)
            self._period = 1.0 / _refresh_rate(self)
            self._deadline = time.monotonic()
            frames = QtCore.QTimer(self, timeout=self._paced_frame, singleShot=True)
        # Frames on time: a coarse timer may fire up to 5% late, uneven scrolling
        frames.setTimerType(QtCore.Qt.PreciseTimer)
        self._closed = False
        self._timers = [frames]
        if not compact:
            self._timers.append(QtCore.QTimer(self, timeout=self.update_slow, interval=SLOW_MS))
        for timer in self._timers:
            timer.start()

    def closeEvent(self, event) -> None:
        """A closed viewer stops reading its streams (it may outlive its window, e.g. in tests)."""
        self._closed = True
        for timer in self._timers:
            timer.stop()
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refit.start()  # once the resizing pauses

    def _refit_text(self) -> None:
        """The text as large as the window is tall (``screen.fit_font``): at full height as
        when the app started, smaller in a smaller window (never below the platform's
        default). Plots keep the font they were made with, so they are restyled."""
        before = font_points(1)
        if fit_font(height=self.height()) == before:
            return
        for plot in self.findChildren(pg.PlotWidget):
            text_style(plot)
            legend = plot.getPlotItem().legend
            if legend is not None:
                legend.setLabelTextSize(legend_text_size())
        self.band.widget.update()
        self.traces.update()
        if isinstance(self.traces, GpuTracePlot):
            self.traces.canvas.update()

    def _paced_frame(self) -> None:
        """CPU traces: a frame, and the next one due a refresh period after this one's due
        time (from now again after a frame that ran late)."""
        if self._closed:
            return
        self.update_plots()
        now = time.monotonic()
        self._deadline += self._period
        if self._deadline < now:
            self._deadline = now + self._period
        self._timers[0].start(max(0, round((self._deadline - now) * 1000)))

    def _frame_shown(self) -> None:
        """GPU traces: the last frame is on the screen; prepare the next one."""
        self._last_frame = time.monotonic()
        QtCore.QTimer.singleShot(0, self._next_frame)  # once the swap has returned

    def _next_frame(self) -> None:
        if not self._closed:
            self.update_plots()
            self.traces.canvas.update()  # also when paused: keeps the frames coming

    def _watchdog(self) -> None:
        """GPU traces: read the streams and restart the frames when none have been shown for a
        while (the window hidden, or the first frame)."""
        if time.monotonic() - self._last_frame > WATCHDOG_MS / 1000:
            self._next_frame()

    # --- updates -----------------------------------------------------------------------------

    def auto_spacing(self) -> None:
        _, x = self.emg.ring.last_seconds(self.controls.window_s, EMG_RATE_HZ)
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
        t, x = self.emg.ring.last_seconds(win, EMG_RATE_HZ)
        sweep_now = None if self.controls.scroll_box.isChecked() else now
        self.traces.draw(
            t - now, x, self.controls.spacing, win, self.controls.offset_box.isChecked(),
            sweep_now=sweep_now, now=now,
        )  # fmt: skip
        if self.imu is not None:
            ti, xi = self.imu.ring.last_seconds(win, IMU_RATE_HZ)
            self.imu_plots.draw(ti - now, xi[:, 0:3], xi[:, 3:6], win, sweep_now=sweep_now)
        for pn in self.panels:
            pn.sweep = sweep_now is not None
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


def _refresh_rate(widget: QtWidgets.QWidget) -> float:
    """The refresh rate (Hz) of the screen the widget is on; 60 when it can't be told."""
    screen = widget.screen() if hasattr(widget, "screen") else None
    screen = screen or QtWidgets.QApplication.primaryScreen()
    rate = screen.refreshRate() if screen is not None else 0.0
    return rate if 20 <= rate <= 360 else 60.0


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
    ap.add_argument(
        "--source", metavar="SOURCE",
        help="the session to show, by its source_id or the host name of its computer, when "
             "several publish (default: one on this computer, else the first found)",
    )  # fmt: skip
    return ap


def run(args: argparse.Namespace, panels: list[ViewerPanel] | tuple = ()) -> None:
    """Open the viewer for ``parser()``'s ``args`` and run until it is closed."""
    app = pg.mkQApp("EMG viewer")
    fit_font(app)  # before any window
    select_source(getattr(args, "source", None))
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
