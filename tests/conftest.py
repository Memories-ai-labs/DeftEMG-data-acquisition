"""Fixtures shared across test modules (session-scoped: each is built once per test run)."""

import os
import sys
import time
import traceback

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def qt_callback_errors(monkeypatch):
    """A Python exception inside a Qt callback (a timer, a paint) fails the running test, with
    its traceback. Without an exception hook PyQt aborts the whole run instead, and the output
    pytest captured, that traceback included, is lost."""
    caught = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc: caught.append(exc))
    yield
    if caught:
        text = "".join(traceback.format_exception(*caught[0]))
        pytest.fail(f"exception in a Qt callback:\n{text}", pytrace=False)


@pytest.fixture(scope="session")
def prompter_trial(tmp_path_factory):
    """Record a short prompter block with the simulated armband; returns the session folder."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    tmp = tmp_path_factory.mktemp("replay")
    seq = Sequence(
        "rep",
        "multi",
        [
            Prompt(0.3, 0.4, ("index",), ("high",)),
            Prompt(1.0, 0.3, ("thumb", "pinky"), ("low", "high")),
        ],
    )
    app = pg.mkQApp()
    config = SessionConfig(data_dir=tmp, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp / "rep.json", seq)], hand="left", tail_s=0.3)
        win.show()
        win.start_block()
        deadline = time.monotonic() + 5
        while win.running and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 5)
        win.close()
    return tmp / "s"


@pytest.fixture(scope="session")
def keyboard_trial(tmp_path_factory):
    """A keyboard-started trial (no prompts) recorded with the simulated armband; npz path."""
    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband

    tmp = tmp_path_factory.mktemp("keyboard")
    config = SessionConfig(data_dir=tmp, session_name="k", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        time.sleep(0.1)
        session.start_trial(label="grip")
        time.sleep(0.3)
        result = session.end_trial()
    return tmp / "k" / os.path.basename(result.npz_path)


@pytest.fixture(scope="session")
def synthetic_sessions(tmp_path_factory):
    """Two small synthetic sessions (strong signal, one with a stall); returns their folders."""
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    root = tmp_path_factory.mktemp("synthetic")
    common = dict(n_blocks=3, per_block=6, lead_in_s=3.0, strength=3.0, compress=False)
    a = write_session(root / "A", SyntheticConfig(seed=1, **common))
    b = write_session(root / "B", SyntheticConfig(seed=2, stall_s=0.3, **common))
    return [a, b]


@pytest.fixture(scope="session")
def legacy_trial(synthetic_sessions, tmp_path_factory):
    """A copy of a synthetic trial saved the old way: three levels and no level names, digit
    names or drawn prompt times (as the earliest recordings were)."""
    import shutil

    src = synthetic_sessions[0]
    dst = tmp_path_factory.mktemp("legacy") / "L"
    shutil.copytree(src, dst)
    npz = sorted(dst.glob("trial_*.npz"))[0]
    with np.load(npz) as d:
        old_keys = ("level_names", "digit_names", "prompt_onset_paint_ns", "prompt_offset_paint_ns")
        arrays = {k: d[k] for k in d.files if k not in old_keys}
    arrays["prompt_level"] = np.where(arrays["prompt_level"] == 1, 2, arrays["prompt_level"])
    np.savez_compressed(npz, **arrays)
    return npz
