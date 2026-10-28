"""Hooks for other code: prompter overlays, task window plugins, viewer and replay panels, and
prompter extensions."""

import json
import logging
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_prompter_overlays_and_on_paint(caplog):
    import pyqtgraph as pg

    from deftemg_acquisition.tasks.prompter import PrompterOverlay, PrompterWidget

    calls = []

    class Recorder(PrompterOverlay):
        def paint_under(self, p, lay, widget):
            calls.append(("under", widget.t))
            p.setPen(pg.mkPen("r"))  # restored after the call

        def paint_over(self, p, lay, widget):
            calls.append(("over", widget.t))

    class Broken(PrompterOverlay):
        def paint_under(self, p, lay, widget):
            raise RuntimeError("overlay bug")

    pg.mkQApp()
    w = PrompterWidget()
    w.resize(400, 700)
    broken, recorder = Broken(), Recorder()
    w.overlays = [broken, recorder]
    painted = []
    w.on_paint = lambda: painted.append(True)
    with caplog.at_level(logging.ERROR):
        assert not w.grab().isNull()
    assert calls == [("under", None), ("over", None)]  # in order, between blocks too
    assert w.overlays == [recorder] and "overlay bug" in caplog.text  # removed, logged
    assert painted == [True]

    def fails():
        raise RuntimeError("on_paint bug")

    w.on_paint = fails
    w.grab()
    assert w.on_paint is None and len(calls) == 4  # turned off; painting goes on


def test_task_window_plugins(tmp_path, caplog):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskPlugin, TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    calls = []

    class Recorder(TaskPlugin):
        def attach(self, window):
            self.window = window
            calls.append("attach")

        def instructions(self, sequence):
            return f"\nplugin note for {sequence.name}\n"

        def block_started(self, sequence, metadata):
            calls.append("started")
            metadata["plugin"] = {"block": sequence.name}

        def block_ending(self, sequence, arrays):
            calls.append("ending")
            assert self.window.t0_ns == int(arrays["scroll_start_host_ns"])
            arrays["plugin_values"] = np.arange(3)

        def block_ended(self, sequence, completed):
            calls.append(("ended", sequence.name, completed))

        def tick(self, t):
            if not calls or calls[-1] != "tick":
                calls.append("tick")

        def close(self):
            calls.append("close")

    class Broken(TaskPlugin):  # every hook after attach raises: the block is recorded
        def __getattribute__(self, name):
            if name in ("instructions", "block_started", "block_ending", "block_ended", "tick",
                        "close"):  # fmt: skip
                raise RuntimeError(f"{name} bug")
            return super().__getattribute__(name)

    class BadAttach(TaskPlugin):
        def attach(self, window):
            raise RuntimeError("attach bug")

    seq = Sequence("quick", "single", [Prompt(0.2, 0.2, ("middle",), ("high",))])
    app = pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with caplog.at_level(logging.ERROR):
        with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
            with pytest.raises(RuntimeError, match="attach bug"):  # before anything is recorded
                TaskWindow(session, [(tmp_path / "q.json", seq)], plugins=[BadAttach()])
            win = TaskWindow(session, [(tmp_path / "q.json", seq)], tail_s=0.2,
                             plugins=[Broken(), Recorder()])  # fmt: skip
            assert calls == ["attach"] and "plugin note for quick" in win.prompter.message
            win.show()
            win.start_block()
            deadline = time.monotonic() + 5
            while win.running and time.monotonic() < deadline:
                app.processEvents(QtCore.QEventLoop.AllEvents, 20)
            result = session.trials[0]
            win.close()
    assert calls == ["attach", "started", "tick", "ending", ("ended", "quick", True), "close"]
    assert caplog.text.count("tick bug") == 1  # logged once, though raised every tick
    assert np.array_equal(np.load(result.npz_path)["plugin_values"], np.arange(3))
    rec = json.loads((tmp_path / "s" / "trials.jsonl").read_text().splitlines()[0])
    assert rec["metadata"]["plugin"] == {"block": "quick"} and rec["metadata"]["completed"]


def test_viewer_panels():
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    from deftemg_acquisition.viz.viewer import Viewer, ViewerPanel

    class Panel(ViewerPanel):
        height = 222

        def __init__(self):
            self.widget = QtWidgets.QLabel("panel")
            self.polled, self.drawn = 0, []

        def poll(self):
            self.polled += 1

        def draw(self, now, window_s):
            self.drawn.append(window_s)

    pg.mkQApp()
    panel = Panel()
    viewer = Viewer(5.0, 2.0, panels=[panel])
    side = next(s for s in viewer.findChildren(QtWidgets.QSplitter)
                if s.orientation() == QtCore.Qt.Vertical)  # fmt: skip
    assert side.widget(0) is viewer.band.widget and side.widget(1) is panel.widget
    viewer.update_plots()
    assert panel.polled == 1 and panel.drawn == [5.0]
    viewer.controls.pause_box.setChecked(True)
    viewer.update_plots()
    assert panel.polled == 2 and panel.drawn == [5.0]  # paused: polled, not drawn
    viewer.close()


def test_replay_panels(prompter_trial):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtWidgets

    from deftemg_acquisition.viz.replay import ReplayPanel, ReplayWindow, load_trial, resolve_trial

    class Panel(ReplayPanel):
        def attach(self, window):
            assert window.prompter is not None
            self.widget = QtWidgets.QLabel("under the traces")
            self.caption = "a key for the prompter"
            self.drawn = []

        def draw(self, now, window_s):
            self.drawn.append(now)

        def status(self, now):
            return f"panel at {now:.1f}"

    pg.mkQApp()
    data = load_trial(resolve_trial(prompter_trial, 1, prompter_trial.parent))
    assert data.trial_id.startswith("s/trial_001")
    panel = Panel()
    win = ReplayWindow(data, window_s=1.0, autoplay=False, panels=[panel])
    middle = panel.widget.parent()
    assert isinstance(middle, QtWidgets.QSplitter) and middle.orientation() == QtCore.Qt.Vertical
    assert middle.widget(0) is win.traces and middle.widget(1) is panel.widget
    assert any(lb.text() == "a key for the prompter" for lb in win.findChildren(QtWidgets.QLabel))
    win.seek(0.5)
    assert panel.drawn[-1] == 0.5 and win.panel_labels[0].text() == "panel at 0.5"
    win.close()


class NoLoop:
    """Stands in for the QApplication of a command's ``run`` / ``main``: ``exec`` closes the
    windows it opened and returns, so no event loop runs inside pytest."""

    def __init__(self, kind):
        self.kind, self.windows = kind, []

    def exec(self):
        from pyqtgraph.Qt import QtWidgets

        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, self.kind) and w.isVisible():
                self.windows.append(w)
                w.close()
        return 0


def test_prompter_extension(tmp_path, monkeypatch):
    import argparse

    import pyqtgraph as pg

    from deftemg_acquisition.cli import prompter
    from deftemg_acquisition.tasks.app import TaskPlugin, TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    seen = {}

    class Recorder(TaskPlugin):
        def attach(self, window):
            seen["plugins"] = window.plugins

        def close(self):
            seen["closed"] = True

    class Extension(prompter.PrompterExtension):
        def add_arguments(self, ap):
            ap.add_argument("--extra", action="store_true")

        def plugins(self, args, config):
            seen["extra"], seen["config"] = args.extra, config
            return [Recorder()]

        def viewer_args(self, args):
            return ["--compact"]  # would open the viewer, but --no-stream

    path = tmp_path / "q.json"
    Sequence("quick", "single", [Prompt(0.2, 0.2, ("middle",), ("high",))]).save(path)
    pg.mkQApp()
    app = NoLoop(TaskWindow)
    monkeypatch.setattr(pg, "mkQApp", lambda *a, **k: app)
    prompter.main(["--simulate", "--no-calibration", "--no-stream", "--extra", "--data-dir",
                   str(tmp_path), "--session", "x", str(path)], extension=Extension())  # fmt: skip
    assert seen["extra"] is True and seen["closed"] is True
    assert seen["config"].session_name == "x" and isinstance(seen["plugins"][0], Recorder)
    assert len(app.windows) == 1 and (tmp_path / "x").is_dir()
    assert isinstance(prompter.parser(), argparse.ArgumentParser)


def test_viewer_and_replay_run_with_panels(prompter_trial, monkeypatch):
    import pylsl
    import pyqtgraph as pg
    import pytest
    from pyqtgraph.Qt import QtWidgets

    from deftemg_acquisition.viz import replay, viewer

    assert viewer.Inlet("EMG", 18, 10, flags=pylsl.proc_clocksync).flags == pylsl.proc_clocksync
    pg.mkQApp()
    app = NoLoop(QtWidgets.QMainWindow)
    monkeypatch.setattr(pg, "mkQApp", lambda *a, **k: app)

    class Panel(viewer.ViewerPanel):
        def __init__(self):
            self.widget = QtWidgets.QLabel("panel")

    panel = Panel()
    args = viewer.parser().parse_args(["--compact", "--beside", "--geometry", "0", "0", "800",
                                       "600"])  # fmt: skip
    with pytest.raises(SystemExit):
        viewer.run(args, panels=[panel])
    [win] = app.windows
    assert isinstance(win, viewer.Viewer) and win.panels == [panel] and win.compact

    made = []

    def panels(data):
        made.append(data)
        return [replay.ReplayPanel()]

    args = replay.parser().parse_args(["--paused", "--data-dir", str(prompter_trial.parent)])
    with pytest.raises(SystemExit):
        replay.run(args, prompter_trial, panels)  # path: instead of args.path
    assert made[0].trial_id.startswith("s/trial_001") and len(app.windows[1].panels) == 1
