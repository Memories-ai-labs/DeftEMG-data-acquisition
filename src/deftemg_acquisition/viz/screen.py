"""Windows side by side on the screen: the prompter on the left and the live viewer on the
right, both full height, without overlap.

    left, right = split_screen()       # frame rectangles on the primary screen
    snap(window, left)                 # after window.show()
    viewer: python -m deftemg_acquisition.viz.viewer --geometry X Y W H   (geometry_args(right))
"""

from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets

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
