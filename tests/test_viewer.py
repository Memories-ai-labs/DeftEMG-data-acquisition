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


def test_peak_envelope_keeps_each_columns_extremes():
    """The traces are drawn from each pixel column's min and max, for all channels at once;
    the newest samples stay at the right edge, and short data is left as it is."""
    import numpy as np

    from deftemg_acquisition.viz.panels import peak_envelope

    rng = np.random.default_rng(0)
    t = np.arange(10_003) / 2000.0
    x = rng.standard_normal((len(t), 18)).astype(np.float32)
    tt, env = peak_envelope(t, x, 1000)  # 10 samples a column; the 3 oldest left out
    assert tt.shape == (2000,) and env.shape == (2000, 18)
    cols = x[3:].reshape(1000, 10, 18)
    np.testing.assert_array_equal(env[0::2], cols.min(axis=1))
    np.testing.assert_array_equal(env[1::2], cols.max(axis=1))
    assert tt[-1] == t[-1] and tt[0] == t[3] and np.all(np.diff(tt) >= 0)
    same_t, same_x = peak_envelope(t[:1500], x[:1500], 1000)
    assert same_t is not None and len(same_t) == 1500 and same_x.shape == (1500, 18)


def test_band_rms_background_is_drawn_once_per_size():
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.viz.band import BandRmsWidget

    pg.mkQApp()
    w = BandRmsWidget("right")
    w.resize(420, 340)
    img = QtGui.QImage(w.size(), QtGui.QImage.Format_ARGB32)
    w.set_rms(np.linspace(5, 90, 18), 100.0, 200)
    w.render(img)
    cached = w._background_cache
    w.set_rms(np.linspace(90, 5, 18), 60.0, 200)  # new RMS and scale: same background
    w.render(img)
    assert w._background_cache is cached
    w.set_rms(np.linspace(90, 5, 18), 60.0, 300)  # the title changes
    w.render(img)
    assert w._background_cache is not cached


def test_trace_plot_falls_back_to_the_cpu_without_a_gpu(monkeypatch):
    """Headless (tests, CI) or with VIEWER_GPU=0, the traces are TracePlot, drawn by the CPU."""
    import pyqtgraph as pg

    from deftemg_acquisition.viz import gpu_traces
    from deftemg_acquisition.viz.panels import TracePlot

    pg.mkQApp()
    ok, why = gpu_traces._check()
    assert not ok and "offscreen" in why
    assert type(gpu_traces.make_trace_plot()) is TracePlot
    monkeypatch.setenv("VIEWER_GPU", "0")
    assert gpu_traces._check() == (False, "VIEWER_GPU=0")


def test_gpu_trace_plot_draws_traces_axes_and_markers():
    """GpuTracePlot's picture (its paint(), drawn here into an image by the CPU): every
    channel's trace in its colour, and a marker only once it is reached."""
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.viz.gpu_traces import GpuTracePlot, _tick_step
    from deftemg_acquisition.viz.panels import CHANNEL_COLORS

    assert [_tick_step(w) for w in (0.5, 2.0, 5.0, 10.0, 30.0)] == [0.1, 0.5, 1, 2, 5]
    pg.mkQApp()
    plot = GpuTracePlot()
    t = np.arange(-10_000, 0) / 2000.0
    x = np.random.default_rng(0).standard_normal((len(t), 18)).astype(np.float32) * 20
    plot.draw(t, x, spacing=300.0, window_s=5.0, remove_offset=True)
    plot.add_marker(0.0, "trial 1", "#59a14f")
    plot.add_marker(1.0, "later", "#e15759")
    plot.place_markers(now=0.5)
    assert [m.isVisible() for _, m in plot.markers] == [True, False]

    w, h = 800, 900
    img = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32)
    p = QtGui.QPainter(img)
    plot.paint(p, w, h)
    p.end()
    area_h = h - plot.TOP - plot.BOTTOM
    for i, color in enumerate(CHANNEL_COLORS):  # each trace's lane holds its colour
        y = int(plot.TOP + (0.5 + i) * area_h / 18)
        want = QtGui.QColor(color).getRgb()[:3]
        row = {img.pixelColor(x, yy).getRgb()[:3] for x in range(plot.LEFT + 5, w - 20, 7)
               for yy in range(y - 3, y + 4)}  # fmt: skip
        assert want in row, f"ch{i + 1}"
    x_marker = int(plot.LEFT + (0.0 - 0.5 + 5.0) / 5.0 * (w - plot.LEFT - plot.RIGHT))
    col = {img.pixelColor(xx, h // 2).getRgb()[:3] for xx in range(x_marker - 2, x_marker + 3)}
    assert (0x59, 0xA1, 0x4F) in col  # the reached marker's line
