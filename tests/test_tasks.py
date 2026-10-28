import json
import os
import time
from collections import Counter

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from deftemg_acquisition.tasks.prompter import Layout  # noqa: E402
from deftemg_acquisition.tasks.sequence import (  # noqa: E402
    DIGITS,
    LEVELS,
    GeneratorConfig,
    Prompt,
    Sequence,
    active_label,
    generate,
)


def test_single_set_is_balanced_and_spaced():
    blocks = generate(GeneratorConfig(task_set="single", reps=2, per_block=7, seed=3), "t")
    prompts = [p for b in blocks for p in b.prompts]
    assert len(prompts) == 5 * 2 * 2 and [len(b.prompts) for b in blocks] == [7, 7, 6]
    counts = Counter((p.digits, p.levels) for p in prompts)
    assert set(counts.values()) == {2} and len(counts) == 10
    for b in blocks:
        assert b.prompts[0].onset_s == pytest.approx(6.5)
        gaps = [q.onset_s - p.offset_s for p, q in zip(b.prompts, b.prompts[1:], strict=False)]
        assert min(gaps) >= 0.8 - 1e-3 and max(gaps) <= 1.5 + 1e-3
        holds = [p.duration_s for p in b.prompts]
        assert min(holds) >= 0.5 - 1e-3 and max(holds) <= 2.0 + 1e-3


@pytest.mark.parametrize("chord_force", ["same", "independent"])
def test_multi_set(chord_force):
    cfg = GeneratorConfig(task_set="multi", chord_sizes=(2, 3), chord_force=chord_force, reps=1)
    prompts = [p for b in generate(cfg, "t") for p in b.prompts]
    assert len(prompts) == (10 + 10) * len(LEVELS)
    assert {len(p.digits) for p in prompts} == {2, 3}
    if chord_force == "same":
        assert all(len(set(p.levels)) == 1 for p in prompts)
    else:
        # each chord's digits see every level exactly once per rep
        for chord in {p.digits for p in prompts}:
            rows = [p.levels for p in prompts if p.digits == chord]
            for i in range(len(chord)):
                assert sorted(r[i] for r in rows) == sorted(LEVELS)


@pytest.mark.parametrize("task_set", ["pinch", "squeeze"])
def test_thumb_sets_join_the_thumb_with_other_digits(task_set):
    """By default the thumb with one other digit, or the whole hand: all 5 chords equally often
    at each level, every digit at the same level; the instructions say which task it is."""
    from deftemg_acquisition.tasks.sequence import THUMB_SETS

    blocks = generate(GeneratorConfig(task_set=task_set, reps=3), "t")
    prompts = [p for b in blocks for p in b.prompts]
    assert len(prompts) == 5 * len(LEVELS) * 3 and [len(b.prompts) for b in blocks] == [15, 15]
    chords = {("thumb", d) for d in DIGITS[1:]} | {DIGITS}
    assert {p.digits for p in prompts} == chords
    assert all(len(set(p.levels)) == 1 for p in prompts)
    counts = Counter((p.digits, p.levels[0]) for p in prompts)
    assert len(counts) == 5 * len(LEVELS) and set(counts.values()) == {3}
    assert {b.task_set for b in blocks} == {task_set} and blocks[0].name == f"t_{task_set}_b01"
    assert blocks[0].params["instructions"] == THUMB_SETS[task_set]
    assert blocks[0].params["chord_sizes"] == [2, 5]
    every = generate(GeneratorConfig(task_set=task_set, chord_sizes=(2, 3, 4, 5), reps=1), "t")
    assert len({p.digits for b in every for p in b.prompts}) == 15  # every set with the thumb
    with pytest.raises(ValueError, match="2-5 digits"):
        generate(GeneratorConfig(task_set=task_set, chord_sizes=(1,)), "t")
    assert "instructions" not in generate(GeneratorConfig(task_set="multi"), "t")[0].params


def test_sequence_roundtrip_and_arrays(tmp_path):
    seq = Sequence("x", "multi", [Prompt(1.0, 2.0, ("index", "ring"), ("low", "high"))])
    seq.save(tmp_path / "x.json")
    back = Sequence.load(tmp_path / "x.json")
    assert back.prompts == seq.prompts
    a = back.prompt_arrays()
    np.testing.assert_array_equal(a["prompt_digits"][0], [0, 1, 0, 1, 0])
    np.testing.assert_array_equal(a["prompt_level"][0], [-1, 0, -1, 1, -1])
    assert list(a["level_names"]) == ["low", "high"]
    assert a["prompt_offset_s"][0] == 3.0
    assert active_label(back, 2.0) == "index:low+ring:high" and active_label(back, 3.0) == ""


def test_layout_egocentric_lanes_and_scroll():
    right = Layout(800, 1000, "right", lookahead_s=3.0)
    left = Layout(800, 1000, "left", lookahead_s=3.0)
    xs = [right.lane_x(d) for d in DIGITS]
    assert xs == sorted(xs)  # thumb leftmost ... pinky rightmost
    assert [left.lane_x(d) for d in DIGITS] == xs[::-1]
    # Arrives at the target exactly at onset, shrinks into it while held, gone at offset.
    assert right.capsule(0.0, 2.0) == (pytest.approx(right.target_y - 2 * right.speed),
                                       right.target_y)  # fmt: skip
    assert right.capsule(-1.0, 1.0)[1] == right.target_y
    assert right.capsule(-2.0, 0.0) is None
    assert right.capsule(3.5, 5.0) is None  # beyond the look-ahead
    assert right.capsule(2.9, 4.0) is not None


def test_task_window_runs_block_and_saves_prompts(tmp_path):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskWindow

    seq = Sequence("quick", "single", [Prompt(0.2, 0.3, ("middle",), ("high",))])
    seq2 = Sequence("next", "single", [Prompt(0.2, 0.3, ("thumb",), ("low",))])
    app = pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp_path / "q.json", seq), (tmp_path / "n.json", seq2)],
                         tail_s=0.2)  # fmt: skip
        win.show()
        win.start_block()
        deadline = time.monotonic() + 5
        while win.running and time.monotonic() < deadline:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
        assert not win.running and win.current == 1 and win.completed == {0}
        result = session.trials[0]
        win.close()

    data = np.load(result.npz_path)
    t0 = int(data["scroll_start_host_ns"])
    assert int(data["prompt_onset_host_ns"][0]) - t0 == 200_000_000
    assert t0 >= result.host_start_ns
    assert data["prompt_level"][0, 2] == 1 and data["prompt_digits"][0].sum() == 1
    drawn_on = int(data["prompt_onset_paint_ns"][0]) - int(data["prompt_onset_host_ns"][0])
    drawn_off = int(data["prompt_offset_paint_ns"][0]) - int(data["prompt_offset_host_ns"][0])
    assert 0 <= drawn_on < 100_000_000 and 0 <= drawn_off < 100_000_000
    rec = json.loads((tmp_path / "s" / "trials.jsonl").read_text().splitlines()[0])
    assert rec["metadata"]["completed"] is True
    assert rec["metadata"]["sequence"]["name"] == "quick" and rec["metadata"]["hand"] == "right"


def test_default_seed_is_random_and_recorded():
    from deftemg_acquisition.tasks.sequence import GeneratorConfig, generate

    a = generate(GeneratorConfig(task_set="single", reps=1, per_block=10), "a")
    b = generate(GeneratorConfig(task_set="single", reps=1, per_block=10), "b")
    assert a[0].params["seed"] != b[0].params["seed"]
    again = generate(GeneratorConfig(task_set="single", reps=1, per_block=10,
                                     seed=a[0].params["seed"]), "a")  # fmt: skip
    assert [p.label for p in again[0].prompts] == [p.label for p in a[0].prompts]


def test_calibration_block_is_short_with_brief_presses():
    """Brief maximum presses, so the shared flexors don't tire: 3 s of rest (after 1 s to
    settle), then one 1.5 s press per digit (two were too tiring), 3 s apart. The ramp skipped
    is a third of the hold, at most 1 s."""
    from deftemg_acquisition.data.calibration import ramp_skip_s
    from deftemg_acquisition.tasks.sequence import calibration_sequence

    seq = calibration_sequence()
    assert [p.digits for p in seq.prompts] == [(d,) for d in DIGITS]  # once each, thumb first
    assert seq.duration_s == pytest.approx(23.5) and seq.params["reps"] == 1
    assert calibration_sequence(reps=2).duration_s == pytest.approx(46.0)
    seq = calibration_sequence(reps=2)
    assert len(seq.prompts) == 2 * len(DIGITS)
    assert {p.duration_s for p in seq.prompts} == {1.5}
    assert seq.prompts[0].onset_s == pytest.approx(4.0)
    gaps = np.diff([p.onset_s for p in seq.prompts]) - 1.5
    assert np.allclose(gaps, 3.0)
    assert ramp_skip_s(1.5) == 0.5 and ramp_skip_s(3.0) == 1.0 and ramp_skip_s(6.0) == 1.0


@pytest.mark.parametrize("task_set", ["single", "multi", "pinch", "squeeze"])
def test_calibration_block_matches_the_task_blocks(task_set):
    """Pinch and squeeze postures differ from pressing, so their sessions calibrate with
    maximum pinches / squeezes of the thumb with each other finger (thumb implied, as in their
    blocks); single and multi sessions with a press of each digit."""
    from deftemg_acquisition.tasks.sequence import (
        CALIBRATION,
        calibration_for,
        calibration_sequence,
    )

    seq = calibration_for(task_set)
    assert seq.task_set == CALIBRATION and seq.level_names == ("max",)
    if task_set in ("pinch", "squeeze"):
        assert seq.calibration_task == task_set and seq.name == f"calibration_{task_set}"
        assert [p.digits for p in seq.prompts] == [("thumb", d) for d in DIGITS[1:]]
        assert seq.implied_digits == ("thumb",) and seq.duration_s == pytest.approx(19.0)
        assert {"pinch": "pinch each", "squeeze": "close the thumb"}[task_set] in (
            seq.params["instructions"])  # fmt: skip
    else:
        assert seq.calibration_task == "press" and seq.name == CALIBRATION
        assert [p.digits for p in seq.prompts] == [(d,) for d in DIGITS]
        assert seq.implied_digits == ()
    assert len(calibration_for(task_set, reps=2).prompts) == 2 * len(seq.prompts)
    old = Sequence(CALIBRATION, CALIBRATION, seq.prompts, {})  # recorded before the task field
    assert old.calibration_task == "press"
    with pytest.raises(ValueError, match="calibration task"):
        calibration_sequence(task="tap")


@pytest.mark.parametrize("task_set", ["pinch", "squeeze", "multi", "calibration_pinch",
                                      "calibration_press"])  # fmt: skip
def test_prompter_leaves_the_implied_thumb_lane_empty(task_set):
    """In pinch and squeeze the thumb always joins in: its lane gets no capsules and no target
    rings, while the prompts (and so the labels) keep it."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.tasks.prompter import DIGIT_COLORS, PrompterWidget

    pg.mkQApp()
    from deftemg_acquisition.tasks.sequence import calibration_sequence

    prompts = [Prompt(4.0, 1.5, ("thumb", "index"), ("high", "high")),
               Prompt(6.0, 1.5, DIGITS, ("low",) * 5)]  # fmt: skip
    if task_set.startswith("calibration_"):  # first effort at 4.0 s: thumb + index, or thumb
        seq = calibration_sequence(task=task_set.removeprefix("calibration_"))
    else:
        seq = Sequence("s", task_set, prompts)
    drawn = task_set in ("multi", "calibration_press")
    assert seq.implied_digits == (() if drawn else ("thumb",))
    w = PrompterWidget()
    w.resize(500, 700)
    w.sequence, w.t = seq, 4.2  # the pinch on the target, the whole hand coming
    img = QtGui.QImage(w.size(), QtGui.QImage.Format_ARGB32)
    w.render(img)
    lay = w.layout_for_size()
    thumb = QtGui.QColor(DIGIT_COLORS["thumb"]).getRgb()[:3]
    centre, half = lay.lane_x("thumb"), lay.lane_w / 2 - 2
    xs = range(round(centre - half), round(centre + half))
    ys = range(int(lay.header_h), int(lay.target_y + lay.radius("high")))
    pixels = (img.pixelColor(x, y).getRgb()[:3] for x in xs for y in ys)
    lit = sum(sum(abs(a - b) for a, b in zip(c, thumb, strict=True)) < 60 for c in pixels)
    assert (lit > 0) == drawn


@pytest.mark.parametrize("kind", ["legacy", "calibration"])
def test_prompter_draws_older_and_calibration_sequences(kind):
    """Recordings from before the two-level task (low / medium / high) and calibration blocks
    ("max") replay without errors in paint (a raise there takes the whole window down)."""
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui

    from deftemg_acquisition.tasks.prompter import PrompterWidget
    from deftemg_acquisition.tasks.sequence import calibration_sequence

    pg.mkQApp()
    if kind == "legacy":
        seq = Sequence("old", "single", [Prompt(1.0, 1.0, ("index",), ("medium",)),
                                         Prompt(3.0, 1.0, ("ring",), ("low",))])  # fmt: skip
        assert seq.level_names == ("low", "medium", "high")
    else:
        seq = calibration_sequence()
    w = PrompterWidget()
    w.resize(500, 700)
    w.sequence, w.t = seq, seq.prompts[0].onset_s + 0.1
    w.message = seq.params.get("instructions", "a long message " * 12)
    img = QtGui.QImage(w.size(), QtGui.QImage.Format_ARGB32)
    w.render(img)
    assert w._ring_levels() == (["low", "medium", "high"] if kind == "legacy" else ["max"])
    # A recorded old-style trial that happened to use only low and high still shows its 3 rings.
    two = Sequence("old", "single", [Prompt(1.0, 1.0, ("index",), ("high",))],
                   {"level_names": ["low", "medium", "high"]})  # fmt: skip
    assert two.level_names == ("low", "medium", "high")


def test_calibration_blocks_come_back_on_schedule(tmp_path):
    import pyqtgraph as pg

    from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig
    from deftemg_acquisition.acquisition.sim import SimulatedArmband
    from deftemg_acquisition.tasks.app import TaskWindow
    from deftemg_acquisition.tasks.sequence import CALIBRATION

    cal = Sequence(CALIBRATION, CALIBRATION, [Prompt(0.1, 0.1, ("index",), ("max",))])
    blocks = [(tmp_path / f"{n}.json", Sequence(n, "squeeze" if n == "d" else "single",
                                                [Prompt(0.1, 0.1, ("thumb",), ("low",))]))
              for n in "abcde"]  # fmt: skip
    pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        win = TaskWindow(session, [(tmp_path / "cal.json", cal), *blocks], tail_s=0.1,
                         recalibrate_s=[600, 1200])  # fmt: skip
        clock = [0.0]
        win._now = lambda: clock[0]

        def run(start: float, end: float | None = None) -> str:
            clock[0] = start
            name = win.sequence.name
            win.start_block()
            assert win.running
            clock[0] = start if end is None else end
            win.end_block(completed=True)
            return name

        assert [run(t) for t in (0, 100, 300)] == ["calibration", "a", "b"]
        # Block c starts before the 10 min are up and ends after: a calibration comes next.
        assert run(590, 650) == "c" and win.sequence.task_set == CALIBRATION
        assert "Calibration due (11 min" in win.prompter.message
        # Block d is a squeeze block: its calibration squeezes (thumb + each finger).
        assert win.sequence.calibration_task == "squeeze"
        assert run(700) == "calibration_squeeze" and win.sequence.name == "d"
        assert run(1800) == "d" and win.sequence.name == "e"  # 1100 s < 1200 s since 700
        clock[0] = 1950  # after a pause, Shift shows the due calibration first
        win.start_block()
        assert not win.running and win.sequence.task_set == CALIBRATION
        assert run(1950) == "calibration" and run(2000) == "e"
        assert len(win.sequences) == 8 and win.completed == set(range(8))
        win.close()
