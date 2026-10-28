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
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
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
