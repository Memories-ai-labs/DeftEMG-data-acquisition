"""The EMG traces drawn by the GPU, when there is one: ``make_trace_plot`` picks this or
``panels.TracePlot``.

``TracePlot`` (pyqtgraph) draws 18 traces with the CPU; on a large high-DPI screen that is most
of a viewer frame (~35 ms at 1400 x 900 on a 200% screen), which caps the scrolling at well
under the screen's 60 Hz. ``GpuTracePlot`` draws the same picture (title, channel labels, time
axis and grid, the traces, labelled markers) with QPainter in a ``QOpenGLWindow`` of its own,
embedded with ``createWindowContainer``: the GPU fills the pixels (a frame takes ~3 ms of CPU),
and the rest of the window stays as it is (a ``QOpenGLWidget`` would make Qt compose the whole
window through OpenGL, which is slower here than not using the GPU at all).

It is used only where it helps (``gpu_traces_available``): PyOpenGL installed (linux-64 here; it
reads the renderer's name), a display (not Qt's offscreen or minimal platform, as in tests), an
OpenGL context that can be made, and a hardware renderer (not Mesa's llvmpipe / softpipe, which
would be slower than the CPU path). ``VIEWER_GPU=0`` keeps ``TracePlot``.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from deftemg_acquisition.viz.panels import (
    CHANNEL_COLORS,
    EMG_CHANNELS,
    TracePlot,
    peak_envelope,
    samples_per_column,
)

log = logging.getLogger(__name__)

# QOpenGLWindow is in QtGui in Qt 5, in QtOpenGL in Qt 6
try:
    _QOpenGLWindow = QtGui.QOpenGLWindow
except AttributeError:
    try:
        from pyqtgraph.Qt import QT_LIB

        _QOpenGLWindow = importlib.import_module(f"{QT_LIB}.QtOpenGL").QOpenGLWindow
    except (ImportError, AttributeError):
        _QOpenGLWindow = None

SOFTWARE_RENDERERS = ("llvmpipe", "softpipe", "swrast", "software")
_available: tuple[bool, str] | None = None


def gpu_traces_available() -> tuple[bool, str]:
    """Whether ``GpuTracePlot`` can be used here, and why (or why not). Checked once."""
    global _available
    if _available is None:
        _available = _check()
        log.info("EMG traces: %s (%s)", "GPU" if _available[0] else "CPU", _available[1])
    return _available


def _check() -> tuple[bool, str]:
    if os.environ.get("VIEWER_GPU", "1") == "0":
        return False, "VIEWER_GPU=0"
    if _QOpenGLWindow is None:
        return False, "this Qt binding has no QOpenGLWindow"
    platform = QtGui.QGuiApplication.platformName()
    if platform in ("", "offscreen", "minimal"):
        return False, f"no display (Qt platform {platform or 'none'})"
    try:
        from OpenGL import GL
    except ImportError:
        return False, "PyOpenGL not installed"
    ctx = QtGui.QOpenGLContext()
    if not ctx.create():
        return False, "no OpenGL context"
    surface = QtGui.QOffscreenSurface()
    surface.setFormat(ctx.format())
    surface.create()
    try:
        if not ctx.makeCurrent(surface):
            return False, "OpenGL context can't be made current"
        renderer = GL.glGetString(GL.GL_RENDERER)
        renderer = renderer.decode(errors="replace") if renderer else ""
    except Exception as e:  # noqa: BLE001 (any failure here means: use the CPU path)
        return False, f"OpenGL check failed ({e})"
    finally:
        ctx.doneCurrent()
    if not renderer or any(s in renderer.lower() for s in SOFTWARE_RENDERERS):
        return False, f"software renderer ({renderer or 'unknown'})"
    return True, renderer


def make_trace_plot() -> TracePlot | GpuTracePlot:
    """The EMG traces panel: drawn by the GPU when ``gpu_traces_available``, else by the CPU."""
    return GpuTracePlot() if gpu_traces_available()[0] else TracePlot()


@dataclass
class _Marker:
    """A labelled vertical line at a time on the caller's timeline (``TracePlot`` markers'
    ``setValue`` / ``setVisible`` / ``isVisible``)."""

    text: str
    color: QtGui.QColor
    width: float
    value: float = 0.0  # time relative to now
    visible: bool = False

    def setValue(self, value: float) -> None:  # noqa: N802 (as pyqtgraph's InfiniteLine)
        self.value = value

    def setVisible(self, visible: bool) -> None:  # noqa: N802
        self.visible = visible

    def isVisible(self) -> bool:  # noqa: N802
        return self.visible


class _TraceCanvas(_QOpenGLWindow or QtGui.QWindow):  # QWindow: never shown (no OpenGL)
    """Paints the traces panel with QPainter on OpenGL (``GpuTracePlot`` holds the state)."""

    def __init__(self, plot: GpuTracePlot):
        super().__init__()
        self.plot = plot

    def paintGL(self) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            self.plot.paint(p, self.width(), self.height())
        finally:
            p.end()


class GpuTracePlot(QtWidgets.QWidget):
    """18 stacked EMG channels against time relative to now, plus labelled marker lines, drawn
    by the GPU; the same calls as ``panels.TracePlot`` (``draw``, ``add_marker``,
    ``place_markers``, ``markers``)."""

    TITLE = "EMG (µV)"
    LEFT, RIGHT, TOP, BOTTOM = 52, 10, 26, 40  # margins around the plot area (px)

    def __init__(self):
        super().__init__()
        self.markers: list[tuple[float, _Marker]] = []  # (time, line), caller's timeline
        self._t = np.empty(0)
        self._x = np.empty((0, len(CHANNEL_COLORS)), np.float32)
        self.spacing = 300.0
        self.window_s = 5.0
        self._background = QtGui.QColor(pg.getConfigOption("background"))
        self._foreground = pg.mkColor(pg.getConfigOption("foreground"))
        self._pens = [QtGui.QPen(QtGui.QColor(c), 1) for c in CHANNEL_COLORS]
        for pen in self._pens:
            pen.setCosmetic(True)
        self.canvas = _TraceCanvas(self)
        container = QtWidgets.QWidget.createWindowContainer(self.canvas, self)
        container.setMinimumSize(200, 150)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

    # --- the TracePlot calls ---------------------------------------------------------------------

    def draw(
        self, rel_t: np.ndarray, x: np.ndarray, spacing: float, window_s: float,
        remove_offset: bool, first_index: int = 0,
    ) -> None:  # fmt: skip
        """``first_index``: the stream position of ``x[0]`` (``peak_envelope``)."""
        if len(rel_t) and remove_offset:
            x = x - x.mean(axis=0)
        # one min and max per column of the plot (logical pixels, as TracePlot)
        columns = max(1, self.canvas.width() - self.LEFT - self.RIGHT)
        k = samples_per_column(window_s, columns)
        self._t, self._x = peak_envelope(rel_t, x, k, first_index) if len(rel_t) else (rel_t, x)
        self.spacing, self.window_s = spacing, window_s
        self.canvas.update()

    def add_marker(self, t: float, text: str, color: str, width: float = 2) -> None:
        """A labelled vertical line at time ``t``, shown once ``now`` reaches it."""
        self.markers.append((t, _Marker(text, QtGui.QColor(color), width)))

    def place_markers(self, now: float, keep_s: float | None = None) -> None:
        """Position markers relative to ``now``; drop those older than ``keep_s``."""
        for t, line in self.markers:
            line.setValue(t - now)
            line.setVisible(bool(t <= now))
        if keep_s is not None:
            self.markers = [m for m in self.markers if m[0] >= now - keep_s]
        self.canvas.update()

    # --- painting (on the canvas) ---------------------------------------------------------------

    def paint(self, p: QtGui.QPainter, w: int, h: int) -> None:
        p.fillRect(QtCore.QRect(0, 0, w, h), self._background)
        area = QtCore.QRectF(self.LEFT, self.TOP, w - self.LEFT - self.RIGHT,
                             h - self.TOP - self.BOTTOM)  # fmt: skip
        if area.width() < 10 or area.height() < 10:
            return
        n = len(self._pens)
        y_top, y_bottom = 0.5 * self.spacing, -(n - 0.5) * self.spacing
        sx = area.width() / self.window_s
        sy = area.height() / (y_top - y_bottom)

        def px(t):  # time relative to now -> x pixel
            return area.left() + (t + self.window_s) * sx

        def py(v):  # µV -> y pixel
            return area.top() + (y_top - v) * sy

        dim = QtGui.QColor(self._foreground)
        font = p.font()
        font.setPointSizeF(9)
        p.setFont(font)
        fm = QtGui.QFontMetricsF(font)

        # Title
        p.setPen(self._foreground)
        p.drawText(QtCore.QRectF(0, 2, w, self.TOP - 4), QtCore.Qt.AlignCenter, self.TITLE)

        # Time axis: ticks, labels and grid lines
        step = _tick_step(self.window_s)
        grid = QtGui.QColor(self._foreground)
        grid.setAlpha(51)  # TracePlot's showGrid(alpha=0.2)
        t = -math.floor(self.window_s / step) * step
        while t <= 1e-9:
            x = px(t)
            p.setPen(QtGui.QPen(grid, 1))
            p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
            p.setPen(dim)
            p.drawLine(QtCore.QPointF(x, area.bottom()), QtCore.QPointF(x, area.bottom() + 5))
            label = f"{t:g}" if step >= 1 else f"{t:.{max(0, -math.floor(math.log10(step)))}f}"
            p.drawText(QtCore.QRectF(x - 30, area.bottom() + 5, 60, fm.height()),
                       QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, label)  # fmt: skip
            t += step
        p.drawText(QtCore.QRectF(area.left(), h - fm.height() - 2, area.width(), fm.height()),
                   QtCore.Qt.AlignCenter, "time (s)")  # fmt: skip
        p.setPen(dim)
        p.drawLine(area.bottomLeft(), area.bottomRight())
        p.drawLine(area.topLeft(), area.bottomLeft())

        # Channel labels
        for i in range(n):
            y = py(-i * self.spacing)
            p.drawLine(QtCore.QPointF(area.left() - 5, y), QtCore.QPointF(area.left(), y))
            p.drawText(QtCore.QRectF(0, y - fm.height() / 2, area.left() - 8, fm.height()),
                       QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, EMG_CHANNELS[i])  # fmt: skip

        p.save()
        p.setClipRect(area)
        # Traces: all channels mapped to pixels at once, one polyline each
        if len(self._t) > 1:
            xs = px(np.asarray(self._t, np.float64))
            ys = area.top() + (y_top - (self._x - np.arange(n) * self.spacing)) * sy
            poly = fn.create_qpolygonf(len(xs))
            arr = fn.ndarray_from_qpolygonf(poly)
            arr[:, 0] = xs
            for i, pen in enumerate(self._pens):
                arr[:, 1] = ys[:, i]
                p.setPen(pen)
                p.drawPolyline(poly)
        # Markers: a line and its label near the top, left-aligned at the line
        for _, m in self.markers:
            if not m.visible or not (-self.window_s <= m.value <= 0):
                continue
            x = px(m.value)
            pen = QtGui.QPen(m.color, m.width)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
            if m.text:
                p.drawText(QtCore.QRectF(x + 2, area.top() + 2, 300, fm.height()),
                           QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, m.text)  # fmt: skip
        p.restore()


def _tick_step(window_s: float) -> float:
    """A 1 / 2 / 5 x 10^k step giving about 5 to 10 ticks over the window."""
    raw = window_s / 6
    mag = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 5, 10):
        if raw <= k * mag:
            return k * mag
    return 10 * mag
