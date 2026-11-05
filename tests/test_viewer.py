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


def test_peak_envelope_columns_hold_still_as_the_window_scrolls():
    """The traces are drawn from each column's min and max, for all channels at once. Columns
    start at fixed stream positions, so as the window scrolls by uneven amounts (as frames
    do) every whole column it still shows is unchanged: no shimmer. Short data is left as is."""
    import numpy as np

    from deftemg_acquisition.viz.panels import peak_envelope, samples_per_column

    rng = np.random.default_rng(0)
    stream = rng.standard_normal((30_000, 18)).astype(np.float32)
    times = np.arange(len(stream)) / 2000.0
    k = samples_per_column(5.0, 1000)
    assert k == 10
    seen = {}  # column start time -> its (min, max)
    end = 10_000
    for step in (31, 33, 32, 29, 35, 7, 64):  # samples between frames
        end += step
        first = end - 10_000
        t, env = peak_envelope(times[first:end], stream[first:end], k, first_index=first)
        assert np.all(np.diff(t) >= 0) and t[-1] == times[end - 1]  # newest at the right edge
        starts, lo, hi = t[0::2], env[0::2], env[1::2]
        for i, t0 in enumerate(starts[:-1]):  # whole columns (the last may still be filling)
            j = round(t0 * 2000)
            assert j % k == 0  # on the stream's grid
            np.testing.assert_array_equal(lo[i], stream[j : j + k].min(axis=0))
            np.testing.assert_array_equal(hi[i], stream[j : j + k].max(axis=0))
            if t0 in seen:
                np.testing.assert_array_equal(seen[t0], (lo[i], hi[i]))
            seen[t0] = (lo[i], hi[i])
    same_t, same_x = peak_envelope(times[:15], stream[:15], k)
    assert len(same_t) == 15 and same_x.shape == (15, 18)


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

    pg.mkQApp()
    ok, why = gpu_traces._check()
    assert not ok and "offscreen" in why
    assert type(gpu_traces.make_trace_plot()) is gpu_traces.CpuTracePlot
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


def test_sweep_columns_hold_still_and_break_at_the_cursor():
    """Sweeping, each column holds a fixed slice of time on a fixed place on the screen:
    drawn once, unchanged as the cursor moves on, until the next sweep replaces it. The line
    breaks where the sweep wraps (NaN): the cursor's column holds the new sweep only."""
    import numpy as np

    from deftemg_acquisition.viz.panels import marker_place, sweep_envelope

    rng = np.random.default_rng(1)
    t_all = 100.0 + np.arange(40_000) / 2000.0  # any clock
    x_all = rng.standard_normal((len(t_all), 18)).astype(np.float32)
    window, columns = 5.0, 1000
    dt = window / columns
    seen = {}
    end = 20_000
    for step in (33, 31, 35, 7, 64, 3000):
        end += step
        t, x = t_all[end - 10_000 : end], x_all[end - 10_000 : end]
        xs, env, cursor = sweep_envelope(t, x, window, columns)
        assert cursor == t[-1] % window
        # In time order: the last sweep's rest (from the column after the cursor's to the right
        # edge), a NaN where it wraps, then this sweep (from the left edge to the cursor)
        brk = np.flatnonzero(np.isnan(xs))
        assert len(brk) <= 1
        old, new = (xs[: brk[0]], xs[brk[0] + 1 :]) if len(brk) else (xs[:0], xs)
        assert new.min() >= 0 and new.max() <= cursor + dt
        if len(old):
            assert old.min() > cursor and old.max() <= window
        else:  # no wrap: everything kept is this sweep, behind the cursor
            assert new.min() >= cursor - window
        # Each column (its min and max): its absolute index (sweep, place on the screen)
        sweep = int(t[-1] // window)
        cols = []
        for part, part_sweep in ((old, sweep - 1), (new, sweep)):
            cols += [part_sweep * columns + round((left - 0.2 * dt) / dt) for left in part[0::2]]
        keep = ~np.isnan(xs)
        lo, hi = env[keep][0::2], env[keep][1::2]
        assert len(cols) == len(lo)
        # Whole columns (not the oldest, being replaced, nor the newest, still filling) are the
        # same in every frame that shows them
        for i, c in enumerate(cols[1:-1], start=1):
            if c in seen:
                np.testing.assert_array_equal(seen[c], (lo[i], hi[i]))
            seen[c] = (lo[i], hi[i])
    # Markers: drawn where the sweep put them, until it clears them
    assert marker_place(101.0, 102.0, None) == (-1.0, True)
    assert marker_place(103.0, 102.0, None)[1] is False
    newest = 107.3
    assert marker_place(106.0, newest, (window, newest)) == (106.0 % window, True)
    assert marker_place(102.2, newest, (window, newest))[1] is False  # replaced
    assert marker_place(108.0, newest, (window, newest))[1] is False  # not yet


def test_traces_draw_sweeping():
    """Both traces panels sweep: the CPU one (its curves and cursor) and the GPU one's
    picture (drawn here into an image), and the live viewer offers the sweep, replay not."""
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.viz.gpu_traces import GpuTracePlot
    from deftemg_acquisition.viz.panels import CHANNEL_COLORS, TracePlot
    from deftemg_acquisition.viz.viewer import Viewer

    pg.mkQApp()
    now = 1000.0
    t = now - np.arange(10_000)[::-1] / 2000.0
    x = np.random.default_rng(2).standard_normal((len(t), 18)).astype(np.float32) * 20
    cpu = TracePlot()
    cpu.resize(800, 900)
    cpu.draw(t - now, x, 300.0, 5.0, True, sweep_now=now)
    assert cpu.cursor.isVisible() and cpu.cursor.value() == t[-1] % 5.0
    xs, _ = cpu.curves[0].getData()
    assert np.isnan(xs).sum() == 1 and np.nanmin(xs) >= 0 and np.nanmax(xs) <= 5.0
    cpu.draw(t - now, x, 300.0, 5.0, True)  # back to scrolling
    assert not cpu.cursor.isVisible()

    gpu = GpuTracePlot()
    gpu.draw(t - now, x, 300.0, 5.0, True, sweep_now=now)
    img = QtGui.QImage(800, 900, QtGui.QImage.Format_ARGB32)
    p = QtGui.QPainter(img)
    gpu.paint(p, 800, 900)
    p.end()
    lane = int(gpu.TOP + 0.5 * (900 - gpu.TOP - gpu.BOTTOM) / 18)
    row = {img.pixelColor(xx, yy).getRgb()[:3] for xx in range(gpu.LEFT + 5, 780, 5)
           for yy in range(lane - 3, lane + 4)}  # fmt: skip
    assert QtGui.QColor(CHANNEL_COLORS[0]).getRgb()[:3] in row

    viewer = Viewer(5.0, 2.0)
    assert viewer.controls.scroll_box.parent() is viewer.controls  # offered live
    assert not viewer.controls.scroll_box.isChecked()  # sweeping by default
    viewer.close()


def test_scrolling_moves_in_whole_columns():
    """Scrolling, the picture moves in whole columns (pixels): every point sits at the same
    place inside its column in every frame, and each column keeps its min and max as it
    moves, so nothing lands on different pixels from one frame to the next (shimmer)."""
    import numpy as np

    from deftemg_acquisition.viz.panels import scroll_envelope

    rng = np.random.default_rng(3)
    t_all = 500.0 + np.arange(40_000) / 2000.0
    x_all = rng.standard_normal((len(t_all), 18)).astype(np.float32)
    window, columns = 5.0, 1096
    dt = window / columns
    seen = {}
    for end in (20_000, 20_033, 20_064, 20_101, 20_102, 20_170):
        t, x = t_all[end - 10_000 : end], x_all[end - 10_000 : end]
        now = t[-1] + 0.004  # the clock runs a little ahead of the newest sample
        xs, env = scroll_envelope(t, x, window, columns, now)
        assert xs.max() < 0 and xs.min() >= -window
        pos = (xs + window) / dt  # in columns from the left edge
        frac = np.round(pos - np.floor(pos), 6)
        assert set(frac) <= {0.2, 0.7}
        absolute = np.floor(pos[0::2]).astype(int) + int(np.floor(now / dt)) - columns + 1
        lo, hi = env[0::2], env[1::2]
        for i, c in enumerate(absolute[1:-1], start=1):  # whole columns
            if c in seen:
                np.testing.assert_array_equal(seen[c], (lo[i], hi[i]))
            seen[c] = (lo[i], hi[i])


def test_ring_last_seconds_covers_the_window_from_a_fast_device():
    """The viewer takes each stream's last window by time: the IMU sends ~217 packets/s, not
    its nominal 208, and window * 208 rows would leave the sweep 4% short (a gap)."""
    import numpy as np

    from deftemg_acquisition.viz.viewer import Ring

    ring = Ring(20_000, 3)
    t = 50.0 + np.arange(5000) / 217.0
    ring.extend(t, np.zeros((len(t), 3), np.float32))
    tw, xw = ring.last_seconds(5.0, 208.0)
    span = tw[-1] - tw[0]
    assert tw[-1] == t[-1] and 5.0 - 1 / 217 - 1e-9 < span <= 5.0 and len(xw) == len(tw)
    assert ring.last_seconds(5.0, 208.0)[0][0] > t[-1] - 5.0
    empty = Ring(10, 3)
    assert len(empty.last_seconds(5.0, 208.0)[0]) == 0


def test_cpu_traces_redrawing_only_what_changed_match_drawing_it_all():
    """CpuTracePlot moves its image by whole columns and draws only the newest columns each
    frame (sweeping: those behind the cursor); after many frames its picture is the one drawn
    whole from the same data, at 200% scale."""
    import numpy as np
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.viz.gpu_traces import CpuTracePlot

    pg.mkQApp()
    rng = np.random.default_rng(4)
    t_all = 300.0 + np.arange(40_000) / 2000.0
    x_all = (rng.standard_normal((len(t_all), 18)) * 40).astype(np.float32)
    w, h, dpr = 800, 600, 2.0

    def picture(plot):
        img = QtGui.QImage(int(w * dpr), int(h * dpr), QtGui.QImage.Format_ARGB32)
        img.setDevicePixelRatio(dpr)
        p = QtGui.QPainter(img)
        plot.paint(p, w, h)
        p.end()
        return img

    def frame(plot, end, sweep):
        now = t_all[end - 1] + 0.003
        t, x = t_all[end - 10_000 : end], x_all[end - 10_000 : end]
        plot.draw(t - now, x, 300.0, 5.0, True, sweep_now=now if sweep else None, now=now)

    for sweep in (False, True):
        live = CpuTracePlot()
        live.resize(w, h)
        live.OFFSET_EVERY_S = 1e9  # one offset throughout, as within a second live
        end = 12_000
        for step in [33, 34, 33, 3, 67, 33] * 6 + [2, 1, 40, 33]:
            end += step
            frame(live, end, sweep)
            img = picture(live)
        whole = CpuTracePlot()
        whole.resize(w, h)
        whole._offset, whole._offset_t = live._offset, live._offset_t  # the same offset
        whole.OFFSET_EVERY_S = 1e9
        frame(whole, end, sweep)
        ref = picture(whole)
        left, top = int(whole.LEFT * dpr), int(whole.TOP * dpr)
        right, bottom = int((w - whole.RIGHT) * dpr), int((h - whole.BOTTOM) * dpr)
        cursor = int((whole.LEFT + whole._cursor / 5.0 * (w - whole.LEFT - whole.RIGHT)) * dpr)
        bad = [
            (x, y) for x in range(left, right, 1) for y in range(top, bottom, 3)
            if img.pixel(x, y) != ref.pixel(x, y) and not (sweep and abs(x - cursor) <= 6)
        ]  # fmt: skip
        assert not bad, (sweep, len(bad), bad[:5])


def test_imu_plots_show_the_sweep_cursor():
    """Sweeping, the gyro and accelerometer plots show the cursor at now, as the EMG traces
    do; scrolling, they don't."""
    import numpy as np
    import pyqtgraph as pg

    from deftemg_acquisition.viz.panels import ImuPlots

    pg.mkQApp()
    imu = ImuPlots()
    now = 1000.3
    t = now - np.arange(1000)[::-1] / 217.0
    data = np.random.default_rng(5).standard_normal((len(t), 3))
    imu.draw(t - now, data, data, 5.0, sweep_now=now)
    assert all(c.isVisible() and c.value() == now % 5.0 for c in imu.cursors)
    assert len(imu.cursors) == 2
    imu.draw(t - now, data, data, 5.0)
    assert not any(c.isVisible() for c in imu.cursors)
