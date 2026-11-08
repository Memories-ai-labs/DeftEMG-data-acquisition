"""Windows side by side on the screen: the prompter on the left and the live viewer on the
right, both full height, without overlap; and text sized from the screen (``fit_font``).

    left, right = split_screen()       # frame rectangles on the primary screen
    snap(window, left)                 # after window.show()
    viewer: python -m deftemg_acquisition.viz.viewer --geometry X Y W H   (geometry_args(right))
"""

from __future__ import annotations

import os

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

PROMPTER_ASPECT = 0.52  # the prompter's width / height beside the viewer (alone: 0.62)
MAX_LEFT_FRACTION = 0.45  # of the screen's width, on narrow screens


def split_screen() -> tuple[QtCore.QRect, QtCore.QRect]:
    """(left, right) window frames over the primary screen's available area (menu bar and dock
    excluded): the left as wide as the prompter's shape gives it at full height."""
    area = QtWidgets.QApplication.primaryScreen().availableGeometry()
    w = min(int(area.height() * PROMPTER_ASPECT), int(area.width() * MAX_LEFT_FRACTION))
    left = QtCore.QRect(area.left(), area.top(), w, area.height())
    right = QtCore.QRect(area.left() + w, area.top(), area.width() - w, area.height())
    return left, right


def snap(window: QtWidgets.QWidget, frame: QtCore.QRect) -> None:
    """Move and size a shown top-level ``window`` so its frame, title bar included, fills
    ``frame``. Its frame is known once it is on screen, so this runs again shortly after."""

    def place():
        outer, inner = window.frameGeometry(), window.geometry()
        dw, dh = outer.width() - inner.width(), outer.height() - inner.height()
        window.resize(max(1, frame.width() - dw), max(1, frame.height() - dh))
        window.move(frame.topLeft())  # a window's position is its frame's

    place()
    QtCore.QTimer.singleShot(200, place)


def geometry_args(frame: QtCore.QRect) -> list[str]:
    """``--geometry X Y W H`` for the viewer's command line."""
    return ["--geometry", *(str(v) for v in (frame.x(), frame.y(), frame.width(), frame.height()))]


# The application's text: this fraction of the height it is fitted to (the screen's available
# height as an app starts, a window's as it is resized: ``fit_font``), so it reads the same on a
# 4K screen at 100% as on a laptop; never smaller than the platform's own.
FONT_FRACTION = 0.011
_font_scale = 1.0
_default: tuple[QtGui.QFont, int] | None = None  # the platform's font, and its height in px


def fit_font(app: QtWidgets.QApplication | None = None, height: float | None = None) -> float:
    """Size the application font to ``FONT_FRACTION`` of ``height`` (logical px; by default
    the primary screen's available height, once per run: later calls without a height keep
    it), never below the platform's default. Returns the factor on the platform's default,
    which ``font_points`` / ``font_px`` apply to the sizes set in code. ``VIEWER_FONT_FRACTION``
    overrides ``FONT_FRACTION``. Call right after the QApplication is made, before any window or
    plot is; a window fitting its text to its own height (``Viewer``) restyles its plots after
    (graphics items keep the font they were made with)."""
    global _default, _font_scale
    if not isinstance(app, QtWidgets.QApplication):  # None, or a stand-in (tests)
        app = QtWidgets.QApplication.instance()
    if app is None:
        return _font_scale
    if _default is None:
        font = QtGui.QFont(app.font())
        _default = (font, QtGui.QFontInfo(font).pixelSize())
    if height is None:
        if app.property("fit_font_done"):
            return _font_scale
        app.setProperty("fit_font_done", True)
        screen = app.primaryScreen()
        if screen is None:
            return _font_scale
        height = screen.availableGeometry().height()
    try:
        fraction = float(os.environ.get("VIEWER_FONT_FRACTION", FONT_FRACTION))
    except ValueError:
        fraction = FONT_FRACTION
    base, base_px = _default
    scale = max(1.0, fraction * height / base_px) if base_px > 0 else 1.0
    if abs(scale - _font_scale) > 0.02 * _font_scale:  # small changes: kept as is
        _font_scale = scale
        font = QtGui.QFont(base)
        font.setPointSizeF(base.pointSizeF() * scale)
        app.setFont(font)
    return _font_scale


def font_points(points: float) -> float:
    """A text size set in code (points, as at the platform's default font), scaled as
    ``fit_font`` scaled the application's."""
    return points * _font_scale


def font_px(px: float) -> int:
    """A length that holds text (margins, label boxes; px at the default font), scaled alike."""
    return round(px * _font_scale)
