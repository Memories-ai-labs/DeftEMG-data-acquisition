import os
import shutil
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_load_trial(prompter_trial):
    from deftemg_acquisition.viz.replay import load_trial, resolve_trial

    npz = resolve_trial(prompter_trial, 1, prompter_trial.parent)
    d = load_trial(npz)
    assert d.hand == "left" and d.sequence is not None and len(d.prompts) == 2
    assert [p.label for p in d.prompts] == ["index H", "thumb+pinky L/H"]
    # Prompt marks are on the EMG timeline at the drawn onset: the first frame showing the press,
    # at or after the scheduled one (how much after depends on the machine's load)
    with np.load(npz) as arrays:
        drawn = (arrays["prompt_onset_paint_ns"] - d.t0_ns) / 1e9
    for mark, prompt, at in zip(d.prompts, d.sequence.prompts, drawn, strict=True):
        assert mark.onset_s == pytest.approx(at, abs=1e-9)
        assert mark.onset_s >= d.scroll_start_s + prompt.onset_s
    assert d.emg_uv.shape == (len(d.emg_t), 18) and np.all(np.diff(d.emg_t) > 0)
    assert d.duration_s == pytest.approx(d.emg_t[-1])


def test_replay_window_navigation(prompter_trial):
    import pyqtgraph as pg

    from deftemg_acquisition.viz.replay import ReplayWindow, load_trial, resolve_trial

    app = pg.mkQApp()
    d = load_trial(resolve_trial(prompter_trial, 1, prompter_trial.parent))
    win = ReplayWindow(d, window_s=1.0, autoplay=False)
    win.show()
    assert win.now == 0.0  # opens at the start with an empty screen, like the live viewer
    from deftemg_acquisition.viz.panels import DEFAULT_SPACING_UV

    assert win.controls.spacing == DEFAULT_SPACING_UV  # as in the live viewer
    win.auto_spacing()
    assert win.controls.spacing > 5  # sized from real data, not the single sample at t = 0
    win.seek(0)
    win.jump_prompt(+1)
    assert win.now == pytest.approx(d.prompts[0].onset_s)
    assert win.prompter.t == pytest.approx(win.now - d.scroll_start_s)
    assert "index H" in win.prompt_label.text()
    lines = [line for _, line in win.traces.markers]
    assert lines[0].isVisible() and not lines[1].isVisible()  # later presses not yet shown
    win.seek(d.prompts[0].offset_s + 0.01)
    assert lines[0].isVisible()  # stays after the press ends, scrolling with the EMG
    win.seek(d.prompts[0].onset_s)
    win.jump_prompt(+1)
    assert win.now == pytest.approx(d.prompts[1].onset_s)
    win.jump_prompt(-1)
    assert win.now == pytest.approx(d.prompts[0].onset_s)
    before = win.now
    win.toggle_play()
    t_end = time.monotonic() + 0.3
    while time.monotonic() < t_end:
        app.processEvents()
    assert win.playing and win.now > before
    win.seek(1e9)
    assert win.now == pytest.approx(d.duration_s)
    assert win.camera.sources == []
    win.close()


def test_calibration_block_records_and_replays(tmp_path):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband
    from deftemg_acquisition.data.trials import scan_trials, select_trials
    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import calibration_sequence
    from deftemg_acquisition.viz.replay import load_trial

    seq = calibration_sequence(rest_s=0.2, hold_s=0.15, gap_s=0.1, reps=1, lead_s=0.1)
    app = pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="c", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp_path / "calibration.json", seq)], tail_s=0.2)
        win.show()
        win.start_block()
        deadline = time.monotonic() + 5
        while win.running and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        win.close()
    info = scan_trials([tmp_path / "c"])[0]
    assert info.is_calibration and info.level_names == ("max",)
    assert select_trials([info], exclude_simulated=False)[1][0][1] == "calibration block"
    replay = load_trial(info.npz_path)
    assert len(replay.prompts) == 5 and replay.prompts[0].label.endswith("MAX")


def test_replay_shows_the_recorded_effort_bar(prompter_trial, tmp_path):
    import pyqtgraph as pg

    from deftemg_acquisition.viz.replay import ReplayWindow, load_trial, resolve_trial

    src = resolve_trial(prompter_trial, 1, prompter_trial.parent)
    session = tmp_path / prompter_trial.name
    shutil.copytree(prompter_trial, session)
    npz = session / src.name
    with np.load(npz) as d:
        arrays = {k: d[k] for k in d.files}
    t0 = int(arrays["emg_host_time_ns"][0])
    arrays["effort_host_ns"] = t0 + np.array([0.5e9, 1.0e9, 1.5e9], np.int64)
    arrays["effort"] = np.array([0.1, np.nan, 0.7], np.float32)
    arrays["effort_max_uv"] = np.ones(3, np.float32)
    arrays["effort_level"] = np.array([-1, 0, 1], np.int8)
    np.savez(npz, **arrays)
    data = load_trial(npz)
    assert data.effort is not None and data.effort.level == [None, "low", "high"]
    pg.mkQApp()
    win = ReplayWindow(data, autoplay=False)
    assert win.prompter.show_effort and win.prompter.effort_zones
    rel = (arrays["effort_host_ns"] - data.t0_ns) / 1e9
    win.seek(float(rel[1]) + 0.01)
    assert win.prompter.effort == (None, "low")  # no reading yet, a low press held
    win.seek(float(rel[2]) + 0.01)
    assert win.prompter.effort[0] == pytest.approx(0.7) and win.prompter.effort[1] == "high"
    win.close()


def _spin(s: float) -> None:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    app, deadline = pg.mkQApp(), time.monotonic() + s
    while time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 5)


def _go(win) -> None:
    """Space starts the countdown; Go 0.1 s later (the windows below)."""
    from test_text_window import Qt, _key

    from deftemg_acquisition.tasks.selfpaced import GO

    _key(win, Qt.Key_Space)
    _spin(0.25)
    assert win.prompter.text_view.phase == GO


TEXT_KW = dict(countdown_s=0.1, go_debounce_s=0.0, tail_s=0.2)  # a text block in about 1 s


@pytest.fixture(scope="module")
def text_trial(tmp_path_factory):
    """A text block recorded by the task window on the host clock: prompt 0 done, prompt 1
    redone during its Go, then done; the trial's npz."""
    from test_text_window import Qt, _grasp_block, _key, _records, _window

    tmp = tmp_path_factory.mktemp("text_replay")
    block = _grasp_block(n=2)
    block.params["conditions"] = ["bottle"]
    with _window(tmp, [block], conditions={"bottle": "full"}, **TEXT_KW) as win:
        win.show()
        win.start_block()
        _spin(0.2)
        for redo in (False, True, False):
            _go(win)
            _spin(0.2)
            _key(win, Qt.Key_R if redo else Qt.Key_Space)
            _spin(0.1)
        _key(win, Qt.Key_Space)  # finish the block
        _spin(0.5)
        assert not win.running
    return _records(tmp)[0]["npz_path"]


def test_text_block_replays_in_text_mode(text_trial, prompter_trial):
    import pyqtgraph as pg

    from deftemg_acquisition.tasks.selfpaced import COUNTDOWN, GO, IDLE, PREVIEW
    from deftemg_acquisition.viz.replay import ReplayWindow, load_trial, resolve_trial

    d = load_trial(text_trial)
    assert d.sequence is None and d.scroll_start_s is None and d.effort is None
    assert d.text_view.block.name == "tuna_can" and d.text_view.conditions == "bottle = full"
    assert [p.label for p in d.prompts] == ["tuna_can:TI", "tuna_can:TIM (redo)", "tuna_can:TIM"]
    with np.load(text_trial) as arrays:
        go = np.where(arrays["text_go_paint_ns"] >= 0, arrays["text_go_paint_ns"],
                      arrays["text_go_host_ns"])  # fmt: skip
        end = arrays["text_end_host_ns"]
        assert (go > arrays["text_go_host_ns"]).any()  # Go drawn after it was logged
    for mark, g, e in zip(d.prompts, go, end, strict=True):
        assert mark.onset_s == pytest.approx((g - d.t0_ns) / 1e9, abs=1e-9)
        assert mark.offset_s == pytest.approx((e - d.t0_ns) / 1e9, abs=1e-9)
        assert 0 < mark.onset_s < mark.offset_s < d.duration_s
    pg.mkQApp()
    win = ReplayWindow(d, window_s=1.0, autoplay=False)
    assert win.prompter.text_mode and win.prompter.text_view.phase == IDLE  # before the block
    win.jump_prompt(+1)
    win.seek(win.now + 0.05)
    view = win.prompter.text_view
    assert view.phase == GO and view.index == 0 and view.conditions == "bottle = full"
    assert "tuna_can:TI" in win.prompt_label.text()
    assert not win.prompter.grab().isNull()  # a frame renders in the Go state
    win.seek(d.prompts[0].offset_s + 0.01)  # rest, the next prompt previewed
    assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 1
    assert win.prompt_label.text() == "prompt: rest"
    win.seek(d.prompts[1].onset_s + 0.01)
    assert "(redo)" in win.prompt_label.text()
    # The view turns to Go when its mark starts: the first frame drawn showing Go
    for mark in d.prompts:
        win.seek(mark.onset_s - 1e-6)
        assert win.prompter.text_view.phase == COUNTDOWN and "rest" in win.prompt_label.text()
        win.seek(mark.onset_s + 1e-6)
        assert win.prompter.text_view.phase == GO and mark.label in win.prompt_label.text()
    win.close()
    # A lane trial replays as before: lanes at the scroll time, no text view
    lane = load_trial(resolve_trial(prompter_trial, 1, prompter_trial.parent))
    assert lane.text_view is None
    win = ReplayWindow(lane, window_s=1.0, autoplay=False)
    win.seek(lane.prompts[0].onset_s)
    assert not win.prompter.text_mode and win.prompter.text_view is None
    assert win.prompter.t == pytest.approx(win.now - lane.scroll_start_s)
    win.close()


def test_attempts_that_never_reached_go_give_no_marks(tmp_path):
    """A countdown cancelled by r, and an early end during a preview, are no marks; an early
    end during Go is marked "(incomplete)". The replayed countdown is the recorded one."""
    from test_text_window import Qt, _grasp_block, _key, _records, _window

    from deftemg_acquisition.tasks.selfpaced import COUNTDOWN, IDLE, PREVIEW
    from deftemg_acquisition.viz.replay import ReplayWindow, load_trial

    with _window(tmp_path, [_grasp_block(n=2)], **TEXT_KW) as win:
        win.show()
        win.start_block()  # run 1: cancel, prompt 0 done, Enter in the preview of prompt 1
        _spin(0.2)
        _key(win, Qt.Key_Space)
        _spin(0.05)
        _key(win, Qt.Key_R)
        _spin(0.1)
        _go(win)
        _spin(0.2)
        _key(win, Qt.Key_Space)
        _spin(0.1)
        _key(win, Qt.Key_Enter)
        _spin(0.5)
        win.start_block()  # run 2: resumes at prompt 1, Enter in its Go
        _spin(0.2)
        _go(win)
        _spin(0.2)
        _key(win, Qt.Key_Enter)
        _spin(0.5)
        assert not win.running
    first, second = (load_trial(r["npz_path"]) for r in _records(tmp_path))
    with np.load(first.npz_path) as arrays:
        assert arrays["text_status"].tolist() == ["cancelled", "ok", "incomplete"]
        assert arrays["text_go_host_ns"][[0, 2]].tolist() == [-1, -1]
        countdown_s = (arrays["text_countdown_host_ns"][0] - first.t0_ns) / 1e9
        end_s = (arrays["text_end_host_ns"][-1] - first.t0_ns) / 1e9
    assert [p.label for p in first.prompts] == ["tuna_can:TI"]
    assert [p.label for p in second.prompts] == ["tuna_can:TIM (incomplete)"]
    for d in (first, second):
        assert all(0 < p.onset_s < min(p.offset_s, d.duration_s) for p in d.prompts)
    win = ReplayWindow(first, autoplay=False)
    win.seek(countdown_s + 0.01)  # in the cancelled countdown: Go was 0.1 s off
    view = win.prompter.text_view
    assert view.phase == COUNTDOWN and 0 < view.countdown_left < 0.1
    assert first.text_view.at(end_s - 0.01).phase == PREVIEW  # the next prompt, then ended
    assert first.text_view.at(end_s).phase == IDLE
    win.close()


def test_a_grasp_calibration_replays_from_its_block_start(tmp_path, monkeypatch):
    """A timed block follows its schedule from block_start_host_ns, not the first EMG sample:
    here the window's clock runs 0.3 s ahead of the EMG's."""
    from test_text_window import _records, _window

    from deftemg_acquisition.clock import host_ns
    from deftemg_acquisition.tasks import prompter
    from deftemg_acquisition.tasks.selfpaced import GO, PREVIEW
    from deftemg_acquisition.tasks.textblock import grasp_calibration_block
    from deftemg_acquisition.viz.replay import ReplayWindow, load_trial

    def ahead():
        return host_ns() + 300_000_000

    monkeypatch.setattr(prompter, "host_ns", ahead)  # Go paint times on the same clock
    cal = grasp_calibration_block({"rest_s": 0.2, "hold_s": 0.2, "gap_s": 0.2}, lead_s=0.1)
    with _window(tmp_path, [cal], tail_s=0.5) as win:
        win._host_ns = ahead
        win.show()
        win.start_block()
        deadline = time.monotonic() + cal.duration_s + 3
        while win.running and time.monotonic() < deadline:
            _spin(0.05)
        assert not win.running
    d = load_trial(_records(tmp_path)[0]["npz_path"])
    with np.load(d.npz_path) as arrays:
        start_s = (int(arrays["block_start_host_ns"]) - d.t0_ns) / 1e9
    assert start_s > 0.2 and len(d.prompts) == len(cal.prompts)
    win = ReplayWindow(d, autoplay=False)
    for i in (0, 1):
        onset = start_s + cal.prompts[i].onset_s
        win.seek(onset - 0.01)
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == i
        win.seek(onset + 0.01)
        assert win.prompter.text_view.phase == GO and win.prompter.text_view.index == i
    win.close()
