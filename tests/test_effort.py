"""The live effort meter (tasks/effort.py) and its bar in the prompter."""

import json
import time

import numpy as np
import pytest

from deftemg_acquisition.analysis.filtering import EmgFilter
from deftemg_acquisition.data.calibration import Calibration
from deftemg_acquisition.tasks.effort import (
    ZONES,
    EffortMeter,
    EffortReference,
    chord_key,
    prompt_level,
    reference_from_trial,
)
from deftemg_acquisition.tasks.sequence import DIGITS, Prompt, Sequence

E = 18
REST_SD = np.full(E, 5.0)


def _vector(channels, amp):
    v = np.zeros(E)
    v[list(channels)] = amp
    return v


def _amp(v):
    return float(np.sqrt(np.mean(v**2)))


def _reference(pinch: bool = False) -> EffortReference:
    maxima = {d: _vector(range(3 * i, 3 * i + 4), 40.0 + 10 * i) for i, d in enumerate(DIGITS)}
    cal = Calibration("s", "s/cal", REST_SD, max_rms_uv=maxima)
    cal.chord_max_rms_uv = (
        {f"thumb+{d}": maxima["thumb"] + maxima[d] for d in DIGITS[1:]} if pinch else dict(maxima)
    )
    return EffortReference.from_calibration(cal)


def test_reference_scales_to_what_is_prompted():
    """100 %: a digit's maximum press, a pinch's own maximum, or a chord's digits together
    (EMG power added); None when a digit has no maximum."""
    ref = _reference()
    assert ref.maximum_uv(("index",)) == pytest.approx(_amp(_vector(range(3, 7), 50.0)))
    a, b = ref.digits_uv["index"], ref.digits_uv["middle"]
    assert ref.maximum_uv(("middle", "index")) == pytest.approx(np.hypot(a, b))
    assert ref.maximum_uv(()) is None
    ref.digits_uv.pop("pinky")
    assert ref.maximum_uv(("ring", "pinky")) is None
    pinch = _reference(pinch=True)
    assert pinch.maximum_uv(("index", "thumb")) == pytest.approx(pinch.efforts_uv["thumb+index"])
    assert chord_key(("pinky", "thumb")) == "thumb+pinky"
    assert prompt_level(("low", "high")) == "high" and prompt_level(("max",)) is None


def test_meter_reads_effort_relative_to_the_maximum():
    """Rest reads ~0; EMG at 30 % of the index maximum (per channel, above the rest noise)
    reads ~0.3; nothing until the window is full or without a reference."""
    rng = np.random.default_rng(0)
    meter = EffortMeter(emg_filter=None)
    meter.push(rng.normal(0, 5.0, (100, E)))
    assert meter.rms_uv() is None and meter.effort(("index",)) is None
    meter.push(rng.normal(0, 5.0, (400, E)))
    assert meter.effort(("index",)) is None  # no reference yet
    meter.reference = _reference()
    assert meter.effort(("index",)) < 0.1
    signal = 0.3 * _vector(range(3, 7), 50.0)
    meter.push(rng.normal(0, 1, (400, E)) * np.sqrt(REST_SD**2 + signal**2))
    assert meter.effort(("index",)) == pytest.approx(0.3, abs=0.05)
    # the same EMG against a two-digit chord's (larger) maximum reads lower
    assert meter.effort(("index", "middle")) < meter.effort(("index",))


def test_meter_decodes_live_armband_chunks(tmp_path):
    """As a sink on a (simulated) armband: chunks decoded and filtered on its worker thread."""
    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband

    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False,
                           filter_enabled=False)  # fmt: skip
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        live = session.live_emg_filter()
        assert live is not None and live.notch == "comb60all" and not live.zero_phase
        meter = EffortMeter(False, live)
        meter.start()
        session.device.attach_sink("effort", meter)
        deadline = time.monotonic() + 3.0
        while meter.rms_uv() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        session.device.detach_sink("effort")
        meter.stop()
    rms = meter.rms_uv()
    assert rms is not None and np.all(np.isfinite(rms)) and rms.shape == (E,)


@pytest.mark.parametrize("task", ["press", "pinch"])
def test_reference_from_a_recorded_calibration_block(tmp_path, task):
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session
    from deftemg_acquisition.data.trials import scan_trials

    cfg = SyntheticConfig(n_blocks=1, per_block=6, calibration=True, calibration_task=task,
                          seed=3, compress=False)  # fmt: skip
    session = write_session(tmp_path / "S", cfg)
    cal = next(i for i in scan_trials([session]) if i.is_calibration)
    ref = reference_from_trial(cal.npz_path, EmgFilter(20.0, "comb60all", zero_phase=False))
    keys = set(DIGITS) if task == "press" else {f"thumb+{d}" for d in DIGITS[1:]}
    assert ref is not None and set(ref.efforts_uv) == keys
    assert all(v > 0 for v in ref.efforts_uv.values()) and ref.calibration_id == cal.id
    assert reference_from_trial(tmp_path / "missing.npz", None) is None


@pytest.mark.parametrize(("value", "level", "state"), [
    (0.30, "low", "inside"), (0.10, "low", "below"), (0.95, "high", "above"),
    (0.50, None, None),
])  # fmt: skip
def test_effort_bar_colour_and_room(value, level, state):
    """One bar under the target row, which moves up to make room; the fill is green inside
    the prompted zone, grey below, vermillion above."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.tasks.prompter import METER_COLOURS, METER_H, PrompterWidget

    pg.mkQApp()
    w = PrompterWidget()
    w.resize(500, 760)
    plain = w.layout_for_size().target_y
    w.show_effort, w.effort_zones = True, dict(ZONES)
    assert w.layout_for_size().target_y == pytest.approx(plain - METER_H)
    w.sequence = Sequence("s", "single", [Prompt(4.0, 1.5, ("index",), ("low",))])
    w.t, w.effort = 4.3, (value, level)
    assert w.effort_state() == state
    img = QtGui.QImage(w.size(), QtGui.QImage.Format_ARGB32)
    w.render(img)
    bar = w.effort_bar(w.layout_for_size())
    x = int(bar.left() + 0.03 * bar.width())  # inside every fill, left of the zones
    got = img.pixelColor(x, int(bar.center().y())).getRgb()[:3]
    want = QtGui.QColor(METER_COLOURS[state or "below"]).getRgb()[:3]
    assert sum(abs(a - b) for a, b in zip(got, want, strict=True)) < 60


def test_task_window_saves_the_effort_trace(tmp_path, caplog):
    """With the meter on, a task block saves its trace and settings; a calibration block that
    can't be used leaves the meter without a reference (and says so)."""
    import pyqtgraph as pg

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import CALIBRATION

    cal = Sequence(CALIBRATION, CALIBRATION, [Prompt(0.1, 0.1, ("index",), ("max",))])
    block = Sequence("b", "single", [Prompt(0.0, 5.0, ("index",), ("low",))])
    pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False,
                           filter_enabled=False)  # fmt: skip
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp_path / "cal.json", cal), (tmp_path / "b.json", block)],
                         tail_s=0.1, effort_meter=True)  # fmt: skip
        assert win.prompter.show_effort and win.effort_meter is not None
        win.start_block()
        win.end_block(completed=True)  # a calibration with no rest: unusable
        assert win.effort_meter.reference is None
        assert any("effort meter" in r.message for r in caplog.records)
        assert "effort bar comes on after a calibration block" in win.prompter.message
        win.effort_meter.reference = _reference()
        win._show_idle()
        assert "Keep the effort bar in the outlined zone" in win.prompter.message
        deadline = time.monotonic() + 3.0
        while win.effort_meter.rms_uv() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        win.start_block()
        for _ in range(3):
            win._tick()
        assert win.prompter.effort is not None and win.prompter.effort[1] == "low"
        win.end_block(completed=True)
        win.close()
    rec = json.loads((tmp_path / "s" / "trials.jsonl").read_text().splitlines()[-1])
    settings = rec["metadata"]["effort_meter"]
    assert settings["reference"] == "s/cal" and settings["zones"]["low"] == list(ZONES["low"])
    z = np.load(rec["npz_path"])
    n = len(z["effort"])
    assert n >= 3 and len(z["effort_host_ns"]) == len(z["effort_max_uv"]) == n
    assert set(z["effort_level"]) <= {-1, 0} and np.isfinite(z["effort"]).all()


def test_a_failing_meter_switches_off_without_stopping_the_block(tmp_path, caplog):
    """An error in the meter during a block logs once and hides the bar; the block runs on."""
    import pyqtgraph as pg

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskWindow

    block = Sequence("b", "single", [Prompt(0.0, 5.0, ("index",), ("low",))])
    pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp_path / "b.json", block)], effort_meter=True)
        win.effort_meter.reference = _reference()

        def broken(_digits):
            raise RuntimeError("boom")

        win.effort_meter.effort = broken
        win.start_block()
        win._tick()
        win._tick()
        assert win.running and not win.prompter.show_effort and win.prompter.effort is None
        assert sum("effort meter failed" in r.message for r in caplog.records) == 1
        win.end_block(completed=True)
        win.close()
    rec = json.loads((tmp_path / "s" / "trials.jsonl").read_text().splitlines()[-1])
    assert rec["metadata"]["completed"]
