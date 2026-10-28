"""The live viewer's layouts, and placing it beside the prompter."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _horizontal_split(viewer):
    from pyqtgraph.Qt import QtCore, QtWidgets

    splits = viewer.findChildren(QtWidgets.QSplitter)
    return next(s for s in splits if s.orientation() == QtCore.Qt.Horizontal)


def test_viewer_layouts():
    import pyqtgraph as pg

    from deftemg_acquisition.viz.viewer import Viewer

    pg.mkQApp()
    full = Viewer(5.0, 2.0)
    assert full.spectrum is not None and full.imu is not None
    assert _horizontal_split(full).widget(0) is full.traces  # the side panels on the right
    assert full.controls.layout().count() == 1 and full.controls.spec_combo.parent() is not None

    beside = Viewer(5.0, 2.0, beside=True)  # as prompter --viewer opens it
    assert _horizontal_split(beside).widget(1) is beside.traces  # side panels on the left
    assert beside.controls.layout().count() == 2 and beside.spectrum is not None
    beside.close()

    compact = Viewer(5.0, 2.0, compact=True, beside=True)
    assert compact.spectrum is None and compact.imu is None and compact.imu_plots is None
    assert _horizontal_split(compact).widget(1) is compact.traces  # band RMS on the left
    assert compact.controls.layout().count() == 2  # controls on two rows
    assert compact.controls.spec_combo.parent() is None  # no spectrum, no spectrum choice
    compact.update_plots()  # no IMU to poll or draw
    for w in (full, compact):
        w.close()


def test_split_screen_and_snap():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    from deftemg_acquisition.viz.screen import geometry_args, snap, split_screen

    app = pg.mkQApp()
    area = app.primaryScreen().availableGeometry()
    left, right = split_screen()
    assert left.left() == area.left() and right.left() == left.right() + 1
    assert left.width() + right.width() == area.width()
    assert left.height() == right.height() == area.height()
    assert geometry_args(right) == ["--geometry", *map(str, right.getRect())]
    w = QtWidgets.QWidget()
    w.show()
    snap(w, left)
    assert w.frameGeometry() == left
    w.close()
