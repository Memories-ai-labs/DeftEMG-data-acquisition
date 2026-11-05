"""The live viewer's and replay's EMG traces: drawn by the GPU when there is one
(``GpuTracePlot``), else by the CPU redrawing only what changed (``CpuTracePlot``);
``make_trace_plot`` picks.

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
would be slower than the CPU path). ``VIEWER_GPU=0`` draws them with the CPU.
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
    SWEEP_CURSOR_COLOR,
    _columns,
    marker_place,
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


def make_trace_plot() -> GpuTracePlot | CpuTracePlot:
    """The EMG traces panel: drawn by the GPU when ``gpu_traces_available``, else by the CPU
    (``CpuTracePlot``, redrawing only what changed)."""
    return GpuTracePlot() if gpu_traces_available()[0] else CpuTracePlot()


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


class _TraceView(QtWidgets.QWidget):
    """18 stacked EMG channels plus labelled marker lines, scrolling or sweeping: the state and
    the picture shared by ``GpuTracePlot`` and ``CpuTracePlot``, with the same calls as
    ``panels.TracePlot`` (``draw``, ``add_marker``, ``place_markers``, ``markers``).

    The traces are kept as columns on a fixed time grid (``window_s / columns`` each, one per
    pixel of the plot's width; ``panels._columns``): each column's absolute index and its min
    and max per channel. Scrolling, column ``c`` is drawn ``c - first`` pixels from the left
    (``first``: the column at the left edge, from now), so the picture moves in whole pixels;
    sweeping, ``c mod columns`` pixels, so it stays put. "Remove offset" subtracts each
    channel's mean over the window, updated once a second (``OFFSET_EVERY_S``), not every frame:
    a picture shifting up or down by fractions of a pixel would shimmer too, and a stable one is
    what ``CpuTracePlot`` needs to draw only what changed."""

    TITLE = "EMG (µV)"
    LEFT, RIGHT, TOP, BOTTOM = 52, 10, 26, 40  # margins around the plot area (px)
    OFFSET_EVERY_S = 1.0

    def __init__(self):
        super().__init__()
        n = len(CHANNEL_COLORS)
        self.markers: list[tuple[float, _Marker]] = []  # (time, line), caller's timeline
        self.spacing = 300.0
        self.window_s = 5.0
        self._columns = 1000  # columns across the plot (one per pixel once laid out)
        self._cols = np.empty(0, np.int64)  # each kept column's index on the time grid
        self._lo = np.empty((0, n), np.float32)  # and its min and max per channel
        self._hi = np.empty((0, n), np.float32)
        self._first = 0  # scrolling: the column at the left edge
        self._sweep: tuple[float, float] | None = None  # (window_s, newest sample's time)
        self._cursor = 0.0  # the sweep's cursor (s into the sweep)
        self._offset: np.ndarray | None = None  # per channel, subtracted ("Remove offset")
        self._offset_t = -math.inf  # when it was taken (the samples' clock)
        self._offset_version = 0  # changes whenever the offset does
        self._background = QtGui.QColor(pg.getConfigOption("background"))
        self._foreground = pg.mkColor(pg.getConfigOption("foreground"))
        self._pens = [QtGui.QPen(QtGui.QColor(c), 1) for c in CHANNEL_COLORS]
        for pen in self._pens:
            pen.setCosmetic(True)

    def _surface_width(self) -> int:
        """The width (logical px) the panel is painted at."""
        return self.width()

    def _request_paint(self) -> None:
        self.update()

    # --- the TracePlot calls ---------------------------------------------------------------------

    def draw(
        self, rel_t: np.ndarray, x: np.ndarray, spacing: float, window_s: float,
        remove_offset: bool, first_index: int = 0, sweep_now: float | None = None,
        now: float | None = None,
    ) -> None:  # fmt: skip
        """As ``TracePlot.draw`` (``first_index`` is not needed: columns sit on the time grid).
        Without ``now`` (nor ``sweep_now``), ``rel_t`` itself is taken as the clock."""
        columns = self._surface_width() - self.LEFT - self.RIGHT
        if columns < 10:  # not laid out yet
            columns = 1000
        sweeping = sweep_now is not None
        base = sweep_now if sweeping else (now if now is not None else 0.0)
        if len(rel_t) and remove_offset:
            newest = float(rel_t[-1] + base)
            stale = self._offset is None or self._offset.shape != (x.shape[1],)
            if stale or abs(newest - self._offset_t) >= self.OFFSET_EVERY_S:
                self._offset, self._offset_t = x.mean(axis=0), newest
                self._offset_version += 1
            x = x - self._offset
        elif not remove_offset and self._offset is not None:
            self._offset = None
            self._offset_version += 1
        self._sweep = None
        if len(rel_t):
            dt = window_s / columns
            t_abs = rel_t + base
            col, lo, hi = _columns(t_abs, x, dt)
            if sweeping:
                keep = col > col[-1] - columns  # the cursor's column of the last sweep: replaced
                self._sweep = (window_s, float(t_abs[-1]))
                self._cursor = float(t_abs[-1] % window_s)
            else:
                now_col = math.floor(base / dt)
                keep = col > now_col - columns
                self._first = now_col - columns + 1
            self._cols, self._lo, self._hi = col[keep], lo[keep], hi[keep]
        else:
            self._cols = self._cols[:0]
        self._columns, self.spacing, self.window_s = columns, spacing, window_s
        self._request_paint()

    def add_marker(self, t: float, text: str, color: str, width: float = 2) -> None:
        """A labelled vertical line at time ``t``, shown once ``now`` reaches it."""
        self.markers.append((t, _Marker(text, QtGui.QColor(color), width)))

    def place_markers(self, now: float, keep_s: float | None = None) -> None:
        """As ``TracePlot.place_markers``."""
        for t, line in self.markers:
            value, visible = marker_place(t, now, self._sweep)
            line.setValue(value)
            line.setVisible(visible)
        if keep_s is not None:
            self.markers = [m for m in self.markers if m[0] >= now - keep_s]
        self._request_paint()

    # --- painting ---------------------------------------------------------------------------------

    def _draw_columns(
        self, p: QtGui.QPainter, sel: slice, left: float, top: float, col_w: float, sy: float
    ) -> None:
        """Columns ``sel`` (of the kept ones) as polylines, one per channel, each column its min
        then max at 0.2 and 0.7 of its width (clear of the half-pixel rounding edges at 100%
        and 200% scale), broken where the sweep wraps. ``left``, ``top``: the plot area's
        corner on the painter."""
        cols = self._cols[sel]
        if not len(cols):
            return
        n = len(self._pens)
        pos = np.mod(cols, self._columns) if self._sweep is not None else cols - self._first
        xs = left + np.stack([pos + 0.2, pos + 0.7], axis=1).reshape(-1) * col_w
        v = np.stack([self._lo[sel], self._hi[sel]], axis=1).reshape(-1, n)
        ys = top + (0.5 * self.spacing - (v - np.arange(n) * self.spacing)) * sy
        breaks = np.flatnonzero(np.diff(xs) < 0) + 1  # where the sweep wraps
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(xs)], strict=True):
            if b - a < 2:
                continue
            poly = fn.create_qpolygonf(int(b - a))
            arr = fn.ndarray_from_qpolygonf(poly)
            arr[:, 0] = xs[a:b]
            for i, pen in enumerate(self._pens):
                arr[:, 1] = ys[a:b, i]
                p.setPen(pen)
                p.drawPolyline(poly)

    def _paint_traces(self, p: QtGui.QPainter, area: QtCore.QRectF, sy: float) -> None:
        """All the columns, every frame (the GPU fills the pixels)."""
        col_w = area.width() / self._columns
        self._draw_columns(p, slice(None), area.left(), area.top(), col_w, sy)

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
        sweeping = self._sweep is not None
        x0 = 0.0 if sweeping else -self.window_s  # the window's left edge on the time axis

        def px(t):  # time on the axis (relative to now, or into the sweep) -> x pixel
            return area.left() + (t - x0) * sx

        def py(v):  # µV -> y pixel
            return area.top() + (y_top - v) * sy

        p.save()
        p.setClipRect(area)
        self._paint_traces(p, area, sy)
        p.restore()

        dim = QtGui.QColor(self._foreground)
        font = p.font()
        font.setPointSizeF(9)
        p.setFont(font)
        fm = QtGui.QFontMetricsF(font)

        # Title
        p.setPen(self._foreground)
        p.drawText(QtCore.QRectF(0, 2, w, self.TOP - 4), QtCore.Qt.AlignCenter, self.TITLE)

        # Time axis: ticks, labels and grid lines (over the traces, faint)
        step = _tick_step(self.window_s)
        grid = QtGui.QColor(self._foreground)
        grid.setAlpha(51)  # TracePlot's showGrid(alpha=0.2)
        t = 0.0 if sweeping else -math.floor(self.window_s / step) * step
        while t <= x0 + self.window_s + 1e-9:
            x = px(t)
            p.setPen(QtGui.QPen(grid, 1))
            p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
            p.setPen(dim)
            p.drawLine(QtCore.QPointF(x, area.bottom()), QtCore.QPointF(x, area.bottom() + 5))
            label = f"{t:g}" if step >= 1 else f"{t:.{max(0, -math.floor(math.log10(step)))}f}"
            p.drawText(QtCore.QRectF(x - 30, area.bottom() + 5, 60, fm.height()),
                       QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, label)  # fmt: skip
            t += step
        axis_label = "time in sweep (s)" if sweeping else "time (s)"
        p.drawText(QtCore.QRectF(area.left(), h - fm.height() - 2, area.width(), fm.height()),
                   QtCore.Qt.AlignCenter, axis_label)  # fmt: skip
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
        if sweeping:  # the cursor, at the newest sample
            cursor = QtGui.QPen(QtGui.QColor(SWEEP_CURSOR_COLOR), 1)
            cursor.setCosmetic(True)
            p.setPen(cursor)
            x = px(self._cursor)
            p.drawLine(QtCore.QPointF(x, area.top()), QtCore.QPointF(x, area.bottom()))
        # Markers: a line and its label near the top, left-aligned at the line
        for _, m in self.markers:
            if not m.visible or not (x0 <= m.value <= x0 + self.window_s):
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


class GpuTracePlot(_TraceView):
    """The traces drawn by the GPU, in a ``QOpenGLWindow`` of their own (module docstring):
    every column, every frame."""

    def __init__(self):
        super().__init__()
        self.canvas = _TraceCanvas(self)
        fmt = self.canvas.format()
        fmt.setSwapInterval(1)  # each frame waits for the screen's refresh (vsync)
        self.canvas.setFormat(fmt)
        container = QtWidgets.QWidget.createWindowContainer(self.canvas, self)
        container.setMinimumSize(200, 150)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)

    def _surface_width(self) -> int:
        return self.canvas.width()

    def _request_paint(self) -> None:
        self.canvas.update()


class CpuTracePlot(_TraceView):
    """The traces drawn by the CPU (no usable GPU): kept in an image, of which each frame
    redraws only what changed. Scrolling, the image moves left by the whole columns that
    passed (``QPixmap.scroll``) and only the newest columns are drawn; sweeping, only the
    columns behind the cursor. A frame then costs ~1-2 ms of drawing instead of all 18
    traces (~11 ms at 1400 x 900 on a 200% screen; ``TracePlot`` ~35 ms). A new size, spacing,
    window, mode or offset draws the image whole again."""

    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)  # paint() fills it all
        self.setMinimumSize(200, 150)
        self._image: QtGui.QPixmap | None = None
        self._image_key: tuple | None = None
        self._image_first = 0  # scrolling: the column at the image's left edge
        self._done = None  # the last column drawn final (later ones may still change)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        try:
            self.paint(p, self.width(), self.height())
        finally:
            p.end()

    def _paint_traces(self, p: QtGui.QPainter, area: QtCore.QRectF, sy: float) -> None:
        dpr = p.paintEngine().paintDevice().devicePixelRatioF()
        w, h = int(area.width()), int(area.height())
        col_w = area.width() / self._columns
        sweeping = self._sweep is not None
        key = (w, h, dpr, self.spacing, self.window_s, sweeping, self._offset_version,
               self._columns)  # fmt: skip
        whole = (
            self._image is None or key != self._image_key or col_w != 1.0 or self._done is None
            or not len(self._cols)
        )  # fmt: skip
        shift = 0
        if not whole and not sweeping:
            shift = self._first - self._image_first
            whole = not 0 <= shift < self._columns
        if whole:
            self._image = QtGui.QPixmap(max(1, round(w * dpr)), max(1, round(h * dpr)))
            self._image.setDevicePixelRatio(dpr)
            self._image_key = key
        ip = QtGui.QPainter(self._image)
        try:
            if whole:
                ip.fillRect(QtCore.QRectF(0, 0, w, h), self._background)
                sel = slice(None)
            else:
                if shift:  # the picture moves left by whole columns
                    ip.end()
                    self._image.scroll(-round(shift * dpr), 0, self._image.rect())
                    ip.begin(self._image)
                    ip.fillRect(QtCore.QRectF(w - shift, 0, shift, h), self._background)
                # Redraw from the last final column on (the line joining it to the next one, still
                # filling, crosses into it), the line drawn from the column before it
                start = self._done
                i0 = max(0, int(np.searchsorted(self._cols, start)) - 1)
                for x_from, x_to in self._spans(start, int(self._cols[-1])):
                    ip.fillRect(QtCore.QRectF(x_from, 0, x_to - x_from, h), self._background)
                sel = slice(i0, None)
            self._draw_columns(ip, sel, 0.0, 0.0, col_w, sy)
        finally:
            ip.end()
        self._image_first = self._first
        self._done = int(self._cols[-1]) - 1 if len(self._cols) else None  # newest: filling
        p.drawPixmap(area.topLeft(), self._image)

    def _spans(self, first: int, last: int) -> list[tuple[int, int]]:
        """Where columns ``first`` to ``last`` are in the image (x from, x to), cleared before
        they are drawn again: scrolling, to the right edge; sweeping, in two parts across the
        wrap."""
        if self._sweep is None:
            return [(max(0, first - self._first), self._columns)]
        n = self._columns
        if last - first + 1 >= n:
            return [(0, n)]
        a, b = first % n, last % n
        return [(a, b + 1)] if a <= b else [(a, n), (0, b + 1)]


def _tick_step(window_s: float) -> float:
    """A 1 / 2 / 5 x 10^k step giving about 5 to 10 ticks over the window."""
    raw = window_s / 6
    mag = 10 ** math.floor(math.log10(raw))
    for k in (1, 2, 5, 10):
        if raw <= k * mag:
            return k * mag
    return 10 * mag
