"""The task window running text blocks: keys, saved arrays and metadata, markers, the sync
patch, the grasp calibration, and the idle screen."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui  # noqa: E402

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig  # noqa: E402
from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband  # noqa: E402
from deftemg_acquisition.tasks import app, prompter  # noqa: E402
from deftemg_acquisition.tasks.app import TaskPlugin, TaskWindow  # noqa: E402
from deftemg_acquisition.tasks.effort import EffortReference  # noqa: E402
from deftemg_acquisition.tasks.prompter import PrompterOverlay  # noqa: E402
from deftemg_acquisition.tasks.selfpaced import (  # noqa: E402
    DONE,
    GO,
    IDLE,
    PREVIEW,
    TextView,
    attempts_from_arrays,
    text_view_at,
)
from deftemg_acquisition.tasks.sequence import CALIBRATION, Prompt, Sequence  # noqa: E402
from deftemg_acquisition.tasks.textblock import (  # noqa: E402
    GRASP_CALIBRATION,
    TEXT_INT_ARRAYS,
    TEXT_STR_ARRAYS,
    TextBlock,
    TextPrompt,
    grasp_calibration_block,
)

Qt = QtCore.Qt
ROOT = Path(__file__).resolve().parents[1]
S = 1_000_000_000  # ns per s
TAIL_S = 0.5


def _grasp_block(name="tuna_can", n=2, **prompt_kw) -> TextBlock:
    prompts = [
        TextPrompt(
            f"Tuna can, from the top\nGrasp {i}",
            "tuna_can",
            grasp=g,
            approach="top",
            highlight=("thumb", "index"),
            **prompt_kw,
        )  # fmt: skip
        for i, g in zip(range(n), ("TI", "TIM", "AD"), strict=False)
    ]
    params = {"object_name": name.replace("_", " ").capitalize(),
              "instructions": "Put the tuna can in front of the subject."}  # fmt: skip
    return TextBlock(name, "grasp", "self_paced", prompts, params)


class Clock:
    """A fake host clock (int ns) for the window's text runner."""

    def __init__(self, s: float = 100.0):
        self.ns = round(s * S)

    def __call__(self) -> int:
        return self.ns

    def advance(self, s: float) -> None:
        self.ns += round(s * S)


@contextmanager
def _window(tmp_path, blocks, clock: Clock | None = None, **kw):
    pg.mkQApp()
    config = SessionConfig(data_dir=tmp_path, session_name="s", live_stream=False)
    with AcquisitionSession(config, device=Armband("sim", ser=SimulatedArmband())) as session:
        markers: list[str] = []
        session.push_marker = markers.append
        kw = {"tail_s": TAIL_S, **kw}
        win = TaskWindow(session, [(tmp_path / f"{b.name}.json", b) for b in blocks], **kw)
        if clock is not None:
            win._host_ns = clock
        win.markers = markers
        try:
            yield win
        finally:
            win.close()


def _key(win: TaskWindow, key) -> None:
    win.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, Qt.NoModifier))


def _wait(win: TaskWindow, clock: Clock, s: float) -> None:
    clock.advance(s)
    win._tick()


def _do_prompt(win: TaskWindow, clock: Clock, task_s: float = 2.0) -> None:
    """Space after the end's debounce, the countdown to Go, the task, Space to end it."""
    clock.advance(1.5)
    _key(win, Qt.Key_Space)
    _wait(win, clock, 3.0)
    assert win.prompter.text_view.phase == GO
    clock.advance(task_s)
    _key(win, Qt.Key_Space)


def _do_maximum(win: TaskWindow, clock: Clock) -> None:
    """A grasp calibration maximum: Space after the end's debounce, the countdown to Go, and
    the end by itself after the hold."""
    clock.advance(1.5)
    _key(win, Qt.Key_Space)
    _wait(win, clock, 3.0)
    view = win.prompter.text_view
    assert view.phase == GO
    _wait(win, clock, view.prompt.duration_s)


def _run_calibration(win: TaskWindow, clock: Clock) -> None:
    """The running grasp calibration to its end: each maximum, Space at DONE, the tail."""
    for _ in win.sequence.prompts:
        _do_maximum(win, clock)
    assert win.running and win.prompter.text_view.phase == DONE
    clock.advance(1.5)
    _key(win, Qt.Key_Space)
    _wait(win, clock, TAIL_S)


def _records(tmp_path) -> list[dict]:
    lines = (tmp_path / "s" / "trials.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_grasp_block_runs_to_the_end_and_saves_its_arrays(tmp_path):
    clock = Clock()
    block, nxt = _grasp_block(), _grasp_block("ball", n=1)
    with _window(tmp_path, [block, nxt], clock, effort_meter=True) as win:
        assert win.prompter.text_mode and win.prompter.text_view is None
        msg = win.prompter.message
        assert "Place: Tuna can" in msg and "2 prompts, about 1 min" in msg
        assert "Put the tuna can" in msg and "effort bar" not in msg
        _key(win, Qt.Key_Shift)
        start = clock.ns
        assert win.running and win.prompter.text_view.phase == PREVIEW
        assert not win.prompter.show_effort or win.prompter.effort is None
        _do_prompt(win, clock)
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 1
        _do_prompt(win, clock)
        # After the last end the block waits; Space during the debounce does nothing
        assert win.running and win.prompter.text_view.phase == DONE
        _key(win, Qt.Key_Space)
        _wait(win, clock, 0.1)
        assert win.running and win.prompter.text_view.phase == DONE
        clock.advance(1.0)
        _key(win, Qt.Key_Space)  # finish: saved after the tail
        finish = clock.ns
        assert win.running and win.prompter.text_view.phase == IDLE
        _wait(win, clock, TAIL_S - 0.01)
        assert win.running
        _wait(win, clock, 0.01)
        assert not win.running and win.completed == {0} and win.current == 1
        assert "Place: Ball" in win.prompter.message and win.prompter.text_view is None
        markers = list(win.markers)

    rec = _records(tmp_path)[0]
    meta = rec["metadata"]
    assert meta["task"] == "text_prompter" and meta["task_set"] == "grasp"
    assert meta["mode"] == "self_paced" and meta["completed"] is True
    assert meta["sequence"] == json.loads(json.dumps(block.to_dict()))
    assert meta["hand"] == "right" and meta["countdown_s"] == 3.0 and meta["go_debounce_s"] == 1.0
    for absent in ("effort_meter", "lookahead_s", "scroll_start_host_ns"):
        assert absent not in meta
    data = np.load(rec["npz_path"])
    keys = set(data.files)
    scalars = {"block_start_host_ns", "text_finish_host_ns"}
    assert set(TEXT_INT_ARRAYS) | set(TEXT_STR_ARRAYS) | scalars <= keys
    assert not [k for k in keys if k.startswith(("prompt_", "effort", "scroll_start"))]
    assert int(data["block_start_host_ns"]) == start
    assert data["text_finish_host_ns"].shape == () and int(data["text_finish_host_ns"]) == finish
    assert data["text_prompt_index"].tolist() == [0, 1]
    assert data["text_status"].tolist() == ["ok", "ok"]
    assert data["text_end_reason"].tolist() == ["operator", "operator"]
    assert (data["text_go_host_ns"] - data["text_countdown_host_ns"] == 3 * S).all()
    assert (data["text_end_host_ns"] - data["text_go_host_ns"] == 2 * S).all()
    assert data["text_grasp"].tolist() == ["TI", "TIM"]
    assert data["text_object"].tolist() == ["tuna_can"] * 2
    assert data["text_go_paint_ns"].tolist() == [-1, -1]  # never shown: no paint
    # Markers in order, text ones only
    assert markers == [
        "text|preview|0", "text|countdown|0", "text|go|0|tuna_can:TI", "text|end|0",
        "text|preview|1", "text|countdown|1", "text|go|1|tuna_can:TIM", "text|end|1",
    ]  # fmt: skip


def test_the_tail_after_the_finish_replays_as_shown(tmp_path):
    """Live, a finished block shows idle during its tail; the saved npz (attempts and
    ``text_finish_host_ns``) gives the same view at every sampled time, DONE before the
    finish (with r offered) and idle after it."""
    clock = Clock()
    block = _grasp_block()
    seen = []

    def sample():
        seen.append((clock.ns, win.prompter.text_view))

    with _window(tmp_path, [block], clock) as win:
        win.start_block()
        sample()
        for _ in block.prompts:
            _do_prompt(win, clock)
            sample()
        _wait(win, clock, 1.5)
        sample()
        clock.advance(0.01)  # a sample at the finish's ns would be before the key
        _key(win, Qt.Key_Space)  # finish
        finish = clock.ns
        sample()
        for _ in range(4):  # through the tail, then saved
            _wait(win, clock, TAIL_S / 5)
            sample()
        assert win.running
        _wait(win, clock, TAIL_S / 5)
        assert not win.running
    phases = [v.phase for _, v in seen]
    assert phases[:4] == [PREVIEW, PREVIEW, DONE, DONE] and phases[4:] == [IDLE] * 5
    assert seen[3][1].can_redo and not any(v.can_redo for _, v in seen[4:])
    rec = _records(tmp_path)[0]
    with np.load(rec["npz_path"]) as data:
        attempts = attempts_from_arrays(data)
        start, saved = int(data["block_start_host_ns"]), int(data["text_finish_host_ns"])
    assert saved == finish
    sequence = TextBlock.from_dict(rec["metadata"]["sequence"])
    for t_ns, view in seen:
        replayed = text_view_at(sequence, attempts, t_ns, start_ns=start, finish_ns=saved)
        assert replayed == view, (t_ns - finish) / S


def test_fixed_duration_ends_by_itself_or_early_by_space(tmp_path):
    clock = Clock()
    block = _grasp_block(n=2, duration_s=2.0)
    with _window(tmp_path, [block], clock) as win:
        win.start_block()
        _key(win, Qt.Key_Space)
        _wait(win, clock, 3.0)
        assert win.prompter.text_view.time_left == pytest.approx(2.0)
        _wait(win, clock, 2.0)  # ends by itself
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 1
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, 3.0)
        clock.advance(1.2)
        _key(win, Qt.Key_Space)  # early
        assert win.prompter.text_view.phase == DONE
        clock.advance(1.5)
        _key(win, Qt.Key_Return)  # Enter after the last end: complete, after the tail
        assert win.running
        _wait(win, clock, TAIL_S)
        assert not win.running and win.completed == {0}
    rec = _records(tmp_path)[0]
    assert rec["metadata"]["completed"] is True
    data = np.load(rec["npz_path"])
    assert data["text_end_reason"].tolist() == ["timer", "operator"]
    ends = data["text_end_host_ns"] - data["text_go_host_ns"]
    assert ends.tolist() == [2 * S, round(1.2 * S)]


def test_redo_voids_the_attempt_and_repeats_the_prompt(tmp_path):
    clock = Clock()
    with _window(tmp_path, [_grasp_block()], clock) as win:
        win.start_block()
        _do_prompt(win, clock)
        clock.advance(0.5)
        _key(win, Qt.Key_R)  # redo prompt 0 from the next one's preview
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 0
        _do_prompt(win, clock)
        _do_prompt(win, clock)
        assert win.prompter.text_view.phase == DONE
        _key(win, Qt.Key_R)  # the last prompt again, from the finish state
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 1
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert not win.running
    data = np.load(_records(tmp_path)[0]["npz_path"])
    assert data["text_prompt_index"].tolist() == [0, 0, 1, 1]
    assert data["text_status"].tolist() == ["redo", "ok", "redo", "ok"]


def test_enter_ends_early_and_the_block_stays_current(tmp_path):
    clock = Clock()
    lane = Sequence("lane", "single", [Prompt(0.2, 0.3, ("middle",), ("high",))])
    with _window(tmp_path, [_grasp_block(), lane], clock) as win:
        win.start_block()
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, 3.0)
        _key(win, Qt.Key_Enter)  # saved at once, incomplete
        assert not win.running and win.current == 0 and win.completed == set()
        assert win.markers[-2:] == ["text|go|1|tuna_can:TIM", "text|end|1"]
        assert win.prompter.text_view is None
        # Shift resumes the block at the prompt after the last "ok" one (step 6c)
        _key(win, Qt.Key_Shift)
        assert win.running and win.prompter.text_view.index == 1
        _key(win, Qt.Key_E)
        assert not win.running and win.current == 0
    recs = _records(tmp_path)
    assert [r["metadata"]["completed"] for r in recs] == [False, False]
    data = np.load(recs[0]["npz_path"])
    assert data["text_status"].tolist() == ["ok", "incomplete"]
    assert data["text_end_reason"].tolist() == ["operator", "block_end"]
    assert np.load(recs[1]["npz_path"])["text_prompt_index"].tolist() == [1]


def test_esc_after_the_last_end_saves_complete_then_quits(tmp_path):
    clock = Clock()
    with _window(tmp_path, [_grasp_block(n=1)], clock) as win:
        win.show()
        win.start_block()
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Escape)
        assert win.isVisible()
        assert win.running  # the tail is still recorded
        _wait(win, clock, TAIL_S)
        assert not win.running and not win.isVisible()
    recs = _records(tmp_path)
    assert len(recs) == 1 and recs[0]["metadata"]["completed"] is True


def test_grasp_calibration_is_operator_started(tmp_path):
    clock = Clock()
    cal = grasp_calibration_block()
    with _window(tmp_path, [cal, _grasp_block()], clock) as win:
        n = len(cal.prompts)
        assert f"{n} prompts, about 1 min" in win.prompter.message
        win.start_block()
        start = clock.ns
        _wait(win, clock, 5.0)  # waits for the operator, however long
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 0
        _key(win, Qt.Key_Space)
        _wait(win, clock, 3.0)
        assert win.prompter.text_view.phase == GO
        _wait(win, clock, 1.5)  # ends by itself after the hold
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 1
        clock.advance(1.5)
        _key(win, Qt.Key_R)  # redo the first maximum
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 0
        for _ in range(n):
            _do_maximum(win, clock)
        # after the last maximum the block waits at DONE, like every text block
        _wait(win, clock, 10.0)
        assert win.running and win.prompter.text_view.phase == DONE
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert not win.running and win.completed == {0} and win.current == 1
    rec = _records(tmp_path)[0]
    meta = rec["metadata"]
    assert meta["task_set"] == "grasp_calibration" and meta["mode"] == "self_paced"
    assert meta["completed"] is True
    data = np.load(rec["npz_path"])
    assert data["text_prompt_index"].tolist() == [0, 0, 1, 2, 3]
    assert data["text_status"].tolist() == ["redo"] + ["ok"] * n
    assert data["text_end_reason"].tolist() == ["timer"] * (n + 1)
    assert data["text_preview_host_ns"][0] == start
    holds = data["text_end_host_ns"] - data["text_go_host_ns"]
    assert (holds == round(1.5 * S)).all()


def test_a_timed_block_is_rejected_at_window_construction(tmp_path, timed_calibration):
    with pytest.raises(ValueError, match="timed text block can only be replayed"):
        with _window(tmp_path, [_grasp_block(), timed_calibration]):
            pass


def test_closing_mid_block_saves_one_incomplete_trial_with_its_markers(tmp_path):
    clock = Clock()
    with _window(tmp_path, [_grasp_block()], clock) as win:
        win.start_block()
        _key(win, Qt.Key_Space)
        clock.advance(3.5)  # Go is due, but no tick has seen it
        win.close()
        assert not win.running
        assert win.markers == ["text|preview|0", "text|countdown|0", "text|go|0|tuna_can:TI",
                               "text|end|0"]  # fmt: skip
    recs = _records(tmp_path)  # the second close (leaving the context) saves nothing more
    assert len(recs) == 1 and recs[0]["metadata"]["completed"] is False
    data = np.load(recs[0]["npz_path"])
    assert data["text_status"].tolist() == ["incomplete"]
    assert data["text_end_reason"].tolist() == ["block_end"]
    assert int(data["text_go_host_ns"][0]) == int(data["text_countdown_host_ns"][0]) + 3 * S


def test_closing_during_the_tail_saves_one_complete_trial(tmp_path):
    clock = Clock()
    with _window(tmp_path, [_grasp_block(n=1)], clock) as win:
        win.start_block()
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S / 2)
        assert win.running
        win.close()  # at once, without the rest of the tail
        assert not win.running
    recs = _records(tmp_path)
    assert len(recs) == 1 and recs[0]["metadata"]["completed"] is True
    assert recs[0]["metadata"]["tail_s"] == TAIL_S
    data = np.load(recs[0]["npz_path"])
    assert data["text_status"].tolist() == ["ok"]


def test_a_stale_text_view_never_puts_a_lane_block_in_text_mode(tmp_path):
    clock = Clock()
    lane = Sequence("lane", "single", [Prompt(0.2, 0.3, ("middle",), ("high",))])
    with _window(tmp_path, [_grasp_block(n=1), lane], clock) as win:
        win.prompter.text_view = TextView(GO, _grasp_block().prompts[0], 0, 1, 0)
        _key(win, Qt.Key_Right)  # shows the lane block
        assert win.prompter.text_view is None and not win.prompter.text_mode
        _key(win, Qt.Key_Left)
        win.start_block()
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        # the lane block is shown next: no text view left over from the text block
        assert win.sequence is lane and win.prompter.text_view is None
        assert not win.prompter.text_mode
        # and it runs as a lane block, with lane markers and arrays
        win.start_block()
        win.end_block(completed=True)
        assert win.markers[-1].startswith("prompt|")
    data = np.load(_records(tmp_path)[1]["npz_path"])
    assert "prompt_onset_host_ns" in data.files and "text_go_host_ns" not in data.files


def test_go_is_painted_and_the_sync_patch_flashes_only_at_go(tmp_path, monkeypatch):
    """The prompter's paint clock is the fake clock too, and each grab() paints at once: the
    first frame showing Go is stamped with that frame's time, and the sync patch flashes once,
    from that frame, for SYNC_FLASH_S, and not in the preview, countdown or finish state."""
    clock = Clock()
    monkeypatch.setattr(prompter, "host_ns", clock)
    with _window(tmp_path, [_grasp_block(n=1)], clock, sync_patch=True) as win:
        w = win.prompter

        def paint() -> bool:  # whether the patch is white in this frame
            w.grab()
            return w._flash_on

        win.start_block()
        assert not paint()  # preview
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        assert not paint()  # countdown
        _wait(win, clock, 2.99)
        assert not paint()
        _wait(win, clock, 0.02)  # 10 ms after the scheduled Go
        go_frame = clock.ns
        assert w.text_view.phase == GO and paint()
        _wait(win, clock, 0.05)
        assert paint()  # still within SYNC_FLASH_S of the Go frame
        _wait(win, clock, 0.06)
        assert not paint()  # over
        clock.advance(1.0)
        _key(win, Qt.Key_Space)
        assert w.text_view.phase == DONE and not paint()  # finish state
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        assert not paint()  # tail
        _wait(win, clock, TAIL_S)
        assert not win.running
    rec = _records(tmp_path)[0]
    assert rec["metadata"]["sync_patch"] is True
    data = np.load(rec["npz_path"])
    go, painted = int(data["text_go_host_ns"][0]), int(data["text_go_paint_ns"][0])
    assert painted == go_frame and painted - go == 10_000_000
    assert data["sync_flash_paint_ns"].tolist() == [go_frame]


# ---- text and lane blocks in one session: calibrations, effort meter, conditions, plugins ----


def _lane(name: str, task_set: str = "single") -> Sequence:
    return Sequence(name, task_set, [Prompt(0.2, 0.3, ("middle",), ("high",))])


def _session_clock(win: TaskWindow) -> list[float]:
    """The window's calibration clock (s), set by the test."""
    now = [0.0]
    win._now = lambda: now[0]
    return now


def _run(win: TaskWindow, clock: Clock, now: list[float], at: float) -> str:
    """Start the current block at session time ``at`` and run it to its end; its name."""
    now[0] = at
    block = win.sequence
    win.start_block()
    assert win.running, f"{block.name} did not start"
    if isinstance(block, TextBlock) and block.is_calibration:
        _run_calibration(win, clock)
    elif isinstance(block, TextBlock):
        for _ in block.prompts:
            _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
    else:
        win.end_block(completed=True)
    assert not win.running
    return block.name


def test_each_kind_of_block_gets_its_own_kind_of_calibration(tmp_path):
    """Lane and grasp calibrations keep separate clocks: neither counts for the other."""
    clock = Clock()
    lane_cal = Sequence(CALIBRATION, CALIBRATION, [Prompt(0.1, 0.1, ("index",), ("max",))])
    t1, t2, t3 = _grasp_block("t1", n=1), _grasp_block("t2", n=1), _grasp_block("t3", n=1)
    blocks = [lane_cal, _lane("a"), t1, _lane("b", "pinch"), t2, t3]
    with _window(tmp_path, blocks, clock, recalibrate_s=[600, 1200]) as win:
        now = _session_clock(win)
        assert win.sequence is lane_cal  # a lane block first: nothing added
        assert _run(win, clock, now, 0) == CALIBRATION and _run(win, clock, now, 100) == "a"
        # The lane calibration does not count for a text block: a grasp calibration first
        assert win.sequence.task_set == GRASP_CALIBRATION and isinstance(win.sequence, TextBlock)
        assert "Calibration due (no calibration of this kind yet)" in win.prompter.message
        assert _run(win, clock, now, 200) == GRASP_CALIBRATION and win.sequence is t1
        assert _run(win, clock, now, 300) == "t1" and win.sequence.name == "b"  # lane: 300 s
        # 650 s after the lane calibration: one is due before b, though the grasp one was 450 s
        # ago, and it pinches like b
        now[0] = 650
        win.start_block()
        assert not win.running and win.sequence.calibration_task == "pinch"
        assert "Calibration due (11 min since the last one)" in win.prompter.message
        assert _run(win, clock, now, 650) == "calibration_pinch"
        assert _run(win, clock, now, 700) == "b" and win.sequence is t2  # grasp: 500 s
        # 600 s after the grasp calibration's first Go (204.5 s: debounce and countdown after
        # its Shift at 200 s): the next one is due
        now[0] = 804
        assert win._calibration_due(app.GRASP) is None
        now[0] = 805
        win.start_block()
        assert not win.running and win.sequence.task_set == GRASP_CALIBRATION
        assert "Calibration due (10 min since the last one)" in win.prompter.message
        assert _run(win, clock, now, 800) == GRASP_CALIBRATION
        assert _run(win, clock, now, 850) == "t2"
        # after the second grasp calibration, the next is due recalibrate_s[-1] later
        assert _run(win, clock, now, 1990) == "t3"
        names = [b.name for _, b in win.sequences]
        assert names == [CALIBRATION, "a", GRASP_CALIBRATION, "t1", "calibration_pinch", "b",
                         GRASP_CALIBRATION, "t2", "t3"]  # fmt: skip
        assert win.completed == set(range(9))
    tasks = [r["metadata"]["task"] for r in _records(tmp_path)]
    assert tasks == ["scroll_prompter", "scroll_prompter", "text_prompter", "text_prompter",
                     "scroll_prompter", "scroll_prompter", "text_prompter", "text_prompter",
                     "text_prompter"]  # fmt: skip


def test_a_first_text_block_gets_its_grasp_calibration_first(tmp_path):
    clock = Clock()
    block = _grasp_block(n=1)
    block.params["calibration"] = {"hold_s": 2.0}
    with _window(tmp_path, [block, _lane("a")], clock, recalibrate_s=[600, 1200]) as win:
        now = _session_clock(win)
        cal = win.sequence
        assert cal.task_set == GRASP_CALIBRATION and len(win.sequences) == 3
        assert cal.params["calibration"]["hold_s"] == 2.0  # the block's overrides
        assert "Calibration due (no calibration of this kind yet)" in win.prompter.message
        _key(win, Qt.Key_Right)  # skipped: the text block still needs one
        assert win.sequence is block
        _key(win, Qt.Key_Shift)  # the skipped one comes back: replaced, not a second one
        assert not win.running and win.sequence.task_set == GRASP_CALIBRATION
        assert len(win.sequences) == 3 and win.current == 0
        assert _run(win, clock, now, 10) == GRASP_CALIBRATION and win.sequence is block
        assert _run(win, clock, now, 20) == "tuna_can"
        # a lane block needs no first calibration of its kind here (the list's builder adds it)
        assert win.sequence.name == "a" and _run(win, clock, now, 3000) == "a"
        assert len(win.sequences) == 3 and win.completed == {0, 1, 2}
        assert "All blocks done" in win.prompter.message
    # Without recalibrate_s (no calibrations), nothing is added
    with _window(tmp_path / "none", [_grasp_block(n=1)], clock) as win:
        assert len(win.sequences) == 1 and win.sequence.name == "tuna_can"


def test_the_grasp_calibration_clock_starts_at_its_first_go(tmp_path):
    """A grasp calibration can wait at its first preview (objects to set out): the next is due
    ``recalibrate_s[0]`` after its first Go, not after its Shift."""
    clock = Clock()
    with _window(tmp_path, [_grasp_block(n=1), _lane("a")], clock, recalibrate_s=[600]) as win:
        now = _session_clock(win)
        now[0] = 100
        win.start_block()
        assert win.sequence.task_set == GRASP_CALIBRATION
        clock.advance(300)  # five minutes at the first preview
        _run_calibration(win, clock)  # Space 1.5 s later, Go after the countdown: 404.5 s
        assert win._last_calibration[app.GRASP] == pytest.approx(404.5)
        now[0] = 1004
        assert win._calibration_due(app.GRASP) is None
        now[0] = 1005
        assert win._calibration_due(app.GRASP) == "10 min since the last one"


def _grasp_calibrations(win: TaskWindow) -> list[int]:
    return [i for i, (_, b) in enumerate(win.sequences) if b.task_set == GRASP_CALIBRATION]


def test_an_ended_early_grasp_calibration_is_never_doubled(tmp_path):
    clock = Clock()
    block = _grasp_block(n=1)
    with _window(tmp_path, [block, _lane("a")], clock, recalibrate_s=[600, 1200]) as win:
        now = _session_clock(win)
        win.start_block()  # the grasp calibration put in first, ended early
        _wait(win, clock, 2.0)
        _key(win, Qt.Key_Return)
        assert not win.running and win.current == 0 and not win.completed
        for _ in range(2):  # skipped, then Shift on the text block: it comes back, once
            _key(win, Qt.Key_Right)
            assert win.sequence is block
            _key(win, Qt.Key_Shift)
            assert not win.running and win.sequence.task_set == GRASP_CALIBRATION
            assert _grasp_calibrations(win) == [win.current]
        assert _run(win, clock, now, 10) == GRASP_CALIBRATION
        assert _run(win, clock, now, 20) == "tuna_can" and _run(win, clock, now, 30) == "a"
        assert _grasp_calibrations(win) == [0] and len(win.sequences) == 3
        assert win.completed == {0, 1, 2} and "All blocks done" in win.prompter.message
    recs = _records(tmp_path)
    assert [r["metadata"]["completed"] for r in recs] == [False, True, True, True]


def test_a_due_note_goes_with_its_kind_of_block(tmp_path):
    clock = Clock()
    with _window(tmp_path, [_grasp_block(n=1), _lane("a")], clock,
                 recalibrate_s=[600, 1200]) as win:  # fmt: skip
        note = "Calibration due (no calibration of this kind yet)"
        assert note in win.prompter.message
        _key(win, Qt.Key_Right)  # the text block: its calibration is still due
        assert note in win.prompter.message
        _key(win, Qt.Key_Right)  # a lane block: not its calibration
        assert win.sequence.name == "a" and "Calibration due" not in win.prompter.message


def test_effort_meter_stays_on_lane_blocks(tmp_path, monkeypatch):
    """A grasp calibration never sets the effort reference, text blocks neither show nor save
    the meter, and a lane block after a text block still does both."""
    calls = []
    monkeypatch.setattr(app, "reference_from_trial", lambda *a: calls.append(a))
    clock = Clock()
    blocks = [grasp_calibration_block(), _grasp_block(n=1), _lane("a")]
    with _window(tmp_path, blocks, clock, effort_meter=True) as win:
        now = _session_clock(win)
        assert _run(win, clock, now, 0) == GRASP_CALIBRATION
        assert calls == [] and win.effort_meter.reference is None
        win.effort_meter.reference = EffortReference("lane_cal", np.zeros(8), {"middle": 50.0},
                                                     {"middle": 50.0})  # fmt: skip
        assert "effort bar" not in win.prompter.message
        win.start_block()
        _wait(win, clock, 0.5)
        assert win.prompter.effort is None
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert win.sequence.name == "a" and "effort bar" in win.prompter.message
        win.start_block()
        win._tick()
        assert win.prompter.show_effort and win.prompter.effort is not None
        win.end_block(completed=True)
    recs = _records(tmp_path)
    for rec in recs[:2]:
        assert "effort_meter" not in rec["metadata"]
        assert not [k for k in np.load(rec["npz_path"]).files if k.startswith("effort")]
    assert recs[2]["metadata"]["effort_meter"]["reference"] == "lane_cal"
    assert len(np.load(recs[2]["npz_path"])["effort_host_ns"]) >= 1


def test_conditions_go_only_to_the_blocks_that_use_them(tmp_path):
    clock = Clock()
    bottle, ball = _grasp_block("bottle", n=1), _grasp_block("ball", n=1)
    bottle.params["conditions"] = ["bottle_fill"]
    conditions = {"bottle_fill": "filled", "lid": "on"}
    with _window(tmp_path, [bottle, ball, _lane("a")], clock, conditions=conditions) as win:
        now = _session_clock(win)
        assert "Conditions: bottle_fill = filled\n" in win.prompter.message
        assert "lid" not in win.prompter.message
        win.start_block()
        assert win.prompter.text_view.conditions == "bottle_fill = filled"
        win.end_block(completed=False)
        assert _run(win, clock, now, 0) == "bottle"
        assert "Conditions" not in win.prompter.message  # the ball uses none
        assert _run(win, clock, now, 0) == "ball" and _run(win, clock, now, 0) == "a"
    recs = _records(tmp_path)
    assert [r["label"] for r in recs] == ["bottle", "bottle", "ball", "a"]
    meta = [r["metadata"] for r in recs]
    assert meta[0]["conditions"] == meta[1]["conditions"] == {"bottle_fill": "filled"}
    assert "conditions" not in meta[2] and "conditions" not in meta[3]


def test_a_block_needing_a_condition_not_given_fails_when_the_window_is_made(tmp_path):
    bottle = _grasp_block("bottle", n=1)
    bottle.params["conditions"] = ["bottle_fill", "lid"]
    match = r"bottle: needs the session condition\(s\) lid \(given: bottle_fill\)"
    with pytest.raises(ValueError, match=match):
        with _window(tmp_path, [_grasp_block(n=1), bottle], conditions={"bottle_fill": "x"}):
            pass
    bottle.params["conditions"] = "bottle_fill"
    with pytest.raises(ValueError, match="params.conditions must be a list of names"):
        with _window(tmp_path, [bottle], conditions={"bottle_fill": "x"}):
            pass
    assert not (tmp_path / "s" / "trials.jsonl").exists()  # nothing recorded


class _Recorder(TaskPlugin):
    def __init__(self):
        self.calls = []

    def instructions(self, sequence):
        self.calls.append(("instructions", sequence))
        return "[plugin note]\n"

    def block_started(self, sequence, metadata):
        self.calls.append(("started", sequence))
        metadata["plugin"] = sequence.task_set

    def block_ending(self, sequence, arrays):
        self.calls.append(("ending", sequence))
        arrays["plugin_x"] = np.arange(3)

    def block_ended(self, sequence, completed):
        self.calls.append(("ended", sequence, completed))


class _LaneOnly(TaskPlugin):
    """Assumes lane blocks: every hook raises on a text block."""

    def instructions(self, sequence):
        return f"{sequence.prompts[0].digits}\n"

    def block_started(self, sequence, metadata):
        metadata["digits"] = list(sequence.prompts[0].digits)

    def block_ending(self, sequence, arrays):
        arrays["digits"] = np.array(sequence.prompts[0].digits)


def test_plugin_hooks_get_the_text_block(tmp_path):
    clock = Clock()
    block = _grasp_block(n=1)
    rec, lane_only = _Recorder(), _LaneOnly()
    with _window(tmp_path, [block, _lane("a")], clock, plugins=[lane_only, rec]) as win:
        assert "[plugin note]" in win.prompter.message  # the lane-only one was skipped
        assert _run(win, clock, _session_clock(win), 0) == "tuna_can"
        assert "('middle',)" in win.prompter.message  # it works on the lane block
    kinds = [c[0] for c in rec.calls if c[1] is block]
    assert kinds[:4] == ["instructions", "started", "ending", "ended"]
    assert ("ended", block, True) in rec.calls
    meta = _records(tmp_path)[0]["metadata"]
    assert meta["plugin"] == "grasp" and "digits" not in meta and meta["completed"] is True
    data = np.load(_records(tmp_path)[0]["npz_path"])
    assert data["plugin_x"].tolist() == [0, 1, 2] and "digits" not in data.files


class _Overlay(PrompterOverlay):
    def __init__(self, text_mode: bool):
        self.text_mode = text_mode
        self.calls = []

    def paint_under(self, p, lay, widget):
        self.calls.append(("under", widget.text_mode, widget.text_view))

    def paint_over(self, p, lay, widget):
        self.calls.append(("over", widget.text_mode, widget.text_view))


def test_only_text_mode_overlays_draw_on_text_blocks(tmp_path):
    clock = Clock()
    lane_ov, text_ov = _Overlay(False), _Overlay(True)
    with _window(tmp_path, [_grasp_block(n=1), _lane("a")], clock) as win:
        w = win.prompter
        w.overlays += [lane_ov, text_ov]
        w.grab()  # between blocks, a text block shown
        assert lane_ov.calls == []
        assert text_ov.calls == [("under", True, None), ("over", True, None)]
        win.start_block()
        text_ov.calls.clear()
        w.grab()
        assert [c[:2] for c in text_ov.calls] == [("under", True), ("over", True)]
        assert all(c[2].phase == PREVIEW for c in text_ov.calls)
        assert lane_ov.calls == [] and w.overlays == [lane_ov, text_ov]  # skipped, not removed
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert win.sequence.name == "a"
        text_ov.calls.clear()
        w.grab()  # a lane block: both drawn
        assert lane_ov.calls == text_ov.calls == [("under", False, None), ("over", False, None)]


# ---- resume: an early-ended block starts again where it stopped ----


def _stem(rec: dict) -> str:
    return os.path.splitext(os.path.basename(rec["npz_path"]))[0]


def _resume_meta(rec: dict) -> tuple:
    meta = rec["metadata"]
    return meta["completed"], meta["text_resume_from"], meta.get("text_resumed_trial")


def _go_and_end_early(win: TaskWindow, clock: Clock) -> None:
    """Start the previewed prompt's countdown, reach Go, then Enter: the block ends early."""
    clock.advance(1.5)
    _key(win, Qt.Key_Space)
    _wait(win, clock, 3.0)
    assert win.prompter.text_view.phase == GO
    _key(win, Qt.Key_Enter)
    assert not win.running


def test_a_resumed_block_chains_its_trials_until_it_completes(tmp_path):
    clock = Clock()
    block, nxt = _grasp_block(n=3), _grasp_block("ball", n=1)
    with _window(tmp_path, [block, nxt], clock) as win:
        assert "3 prompts, about" in win.prompter.message and "Resumes" not in win.prompter.message
        win.start_block()  # run 1: prompt 0 ok, ended early during prompt 1
        _do_prompt(win, clock)
        _go_and_end_early(win, clock)
        assert win.current == 0 and win.resume_point(block).resume_index == 1
        msg = win.prompter.message
        assert "Resumes at prompt 2 of 3\n2 prompts left, about 1 min" in msg
        assert "already done" not in msg
        _key(win, Qt.Key_Shift)  # run 2: starts at prompt 1, ended early with no new "ok"
        assert win.running and win.prompter.text_view.index == 1
        assert win.prompter.text_view.total == 3
        _go_and_end_early(win, clock)
        assert win.resume_point(block).resume_index == 1  # unchanged
        assert "Resumes at prompt 2 of 3" in win.prompter.message
        _key(win, Qt.Key_Shift)  # run 3: from prompt 1 again, to the end
        assert win.prompter.text_view.index == 1
        _do_prompt(win, clock)
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert not win.running and win.completed == {0} and win.current == 1
        assert win.resume_point(block) is None and not win._resume  # dropped once complete
    recs = _records(tmp_path)
    first, second, third = (_stem(r) for r in recs)
    assert [_resume_meta(r) for r in recs] == [
        (False, 0, None), (False, 1, first), (True, 1, second)
    ]  # fmt: skip
    assert "text_resumed_trial" not in recs[0]["metadata"]
    # text_prompt_index is an index into the whole block; each trial holds its own attempts
    indices = [np.load(r["npz_path"])["text_prompt_index"].tolist() for r in recs]
    assert indices == [[0, 1], [1], [1, 2]]
    assert np.load(recs[2]["npz_path"])["text_grasp"].tolist() == ["TIM", "AD"]
    assert third.startswith("trial_")


def test_a_done_block_redone_starts_at_its_first_prompt_and_can_resume(tmp_path):
    clock = Clock()
    block, nxt = _grasp_block(n=3), _grasp_block("ball", n=1)
    with _window(tmp_path, [block, nxt], clock) as win:
        win.start_block()  # ended early, then resumed to the end
        _do_prompt(win, clock)
        _go_and_end_early(win, clock)
        win.start_block()
        for _ in range(2):
            _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert win.completed == {0} and win.current == 1
        _key(win, Qt.Key_Left)  # redo the done block: from its first prompt
        msg = win.prompter.message
        assert "(already done)" in msg and "3 prompts, about" in msg and "resumes" not in msg
        _key(win, Qt.Key_Shift)
        assert win.running and win.prompter.text_view.index == 0
        _do_prompt(win, clock)
        _do_prompt(win, clock)
        _go_and_end_early(win, clock)  # the redo ended early at prompt 2
        msg = win.prompter.message
        assert "Done before; redo resumes at prompt 3 of 3\n1 prompt left, about" in msg
        assert "(already done)" not in msg and win.completed == {0}
        _key(win, Qt.Key_Shift)
        assert win.running and win.prompter.text_view.index == 2
        _key(win, Qt.Key_Enter)
    recs = _records(tmp_path)
    assert [_resume_meta(r)[:2] for r in recs] == [(False, 0), (True, 1), (False, 0), (False, 2)]
    assert "text_resumed_trial" not in recs[2]["metadata"]  # the redo is a fresh run
    assert recs[3]["metadata"]["text_resumed_trial"] == _stem(recs[2])


def test_a_calibration_due_before_a_resume_keeps_the_resume_point(tmp_path):
    clock = Clock()
    block = _grasp_block(n=3)
    with _window(tmp_path, [block, _lane("a")], clock, recalibrate_s=[600, 1200]) as win:
        now = _session_clock(win)
        assert _run(win, clock, now, 0) == GRASP_CALIBRATION and win.sequence is block
        now[0] = 10
        win.start_block()
        _do_prompt(win, clock)
        _go_and_end_early(win, clock)
        assert win.current == 1 and win.resume_point(block).resume_index == 1
        # 700 s after the calibration: Shift puts one in before the block (index 1 -> 2)
        now[0] = 700
        _key(win, Qt.Key_Shift)
        assert not win.running and win.sequence.task_set == GRASP_CALIBRATION
        assert win.sequences[2][1] is block and win.current == 1
        win.start_block()  # that calibration ended early: no resume point
        _wait(win, clock, 2.0)
        _key(win, Qt.Key_Return)
        assert win.resume_point(win.sequence) is None and list(win._resume) == [id(block)]
        _key(win, Qt.Key_Right)  # Shift on the block: the calibration is taken out, put back
        _key(win, Qt.Key_Shift)
        assert not win.running and win.sequence.task_set == GRASP_CALIBRATION
        assert _grasp_calibrations(win) == [0, 1] and win.sequences[2][1] is block
        win.start_block()
        assert win._runner.start_index == 0  # a calibration never resumes
        win.end_block(completed=False)
        win.start_block()
        _run_calibration(win, clock)
        assert not win.running and win.sequence is block and list(win._resume) == [id(block)]
        assert "Resumes at prompt 2 of 3" in win.prompter.message
        win.start_block()  # resumes at prompt 2, not prompt 1
        assert win.running and win.prompter.text_view.index == 1
    recs = _records(tmp_path)
    assert [r["metadata"]["task_set"] for r in recs] == ["grasp_calibration", "grasp"] + [
        "grasp_calibration"] * 3 + ["grasp"]  # fmt: skip
    assert [_resume_meta(r)[1:] for r in recs] == [
        (0, None), (0, None), (0, None), (0, None), (0, None), (1, _stem(recs[1]))
    ]  # fmt: skip


def test_an_ended_early_grasp_calibration_restarts_at_its_first_prompt(tmp_path):
    clock = Clock()
    cal = grasp_calibration_block()
    with _window(tmp_path, [cal, _grasp_block()], clock) as win:
        win.start_block()
        _do_maximum(win, clock)
        _do_maximum(win, clock)  # two grasps done
        _key(win, Qt.Key_Return)
        assert not win.running and win.current == 0 and not win._resume
        assert "Resumes" not in win.prompter.message
        win.start_block()
        assert win.running and win._runner.start_index == 0
        assert win.prompter.text_view.phase == PREVIEW and win.prompter.text_view.index == 0
        _key(win, Qt.Key_Return)
    recs = _records(tmp_path)
    assert [_resume_meta(r) for r in recs] == [(False, 0, None), (False, 0, None)]


def test_a_pair_block_resumes_at_tighten_after_loosen(tmp_path):
    clock = Clock()
    prompts = [
        TextPrompt(f"Jar lid\n{s.capitalize()}", "jar", "Power", "top", None, s, None, "hand")
        for s in ("loosen", "tighten")
    ]
    block = TextBlock("jar", "natural", "self_paced", prompts, {"object_name": "Jar"})
    with _window(tmp_path, [block], clock) as win:
        win.start_block()
        _do_prompt(win, clock)  # loosen ok, then Enter in tighten's preview
        assert win.prompter.text_view.phase == PREVIEW
        _key(win, Qt.Key_Enter)
        assert not win.running and "Resumes at prompt 2 of 2" in win.prompter.message
        win.start_block()
        view = win.prompter.text_view
        assert view.index == 1 and view.prompt.step == "tighten"
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert not win.running and win.completed == {0} and not win._resume
    recs = _records(tmp_path)
    assert [_resume_meta(r) for r in recs] == [(False, 0, None), (True, 1, _stem(recs[0]))]
    first = np.load(recs[0]["npz_path"])
    assert first["text_step"].tolist() == ["loosen", "tighten"]  # tighten only previewed
    assert first["text_status"].tolist() == ["ok", "incomplete"]
    second = np.load(recs[1]["npz_path"])
    assert second["text_step"].tolist() == ["tighten"]
    assert second["text_prompt_index"].tolist() == [1]


def test_all_blocks_done_names_a_redo_that_was_ended_early(tmp_path):
    clock = Clock()
    block = _grasp_block(n=3)
    with _window(tmp_path, [block], clock) as win:
        _run(win, clock, [0.0], 0)
        assert win.completed == {0} and "All blocks done.\nEsc to quit" in win.prompter.message
        win.start_block()  # the redo, ended early after one "ok"
        _do_prompt(win, clock)
        _go_and_end_early(win, clock)
        msg = win.prompter.message
        assert msg.startswith("All blocks done.\ntuna_can: Done before; redo resumes at "
                              "prompt 2 of 3, Shift to go on.\nEsc to quit")  # fmt: skip
        win.start_block()
        assert win.running and win.prompter.text_view.index == 1


def test_a_runner_that_cannot_start_leaves_no_trial_open(tmp_path, monkeypatch):
    clock = Clock()
    with _window(tmp_path, [_grasp_block()], clock) as win:

        def bad_runner(*args, **kw):
            raise ValueError("bad start index")

        monkeypatch.setattr(app, "SelfPacedRunner", bad_runner)
        with pytest.raises(ValueError, match="bad start index"):
            win.start_block()
        assert not win.running and win._runner is None
        assert not win.session.in_trial  # no trial was started
        monkeypatch.undo()
        win.start_block()  # still startable
        assert win.running and win.prompter.text_view.index == 0
        _key(win, Qt.Key_Return)
    assert len(_records(tmp_path)) == 1


def test_saving_shows_until_each_save_returns(tmp_path, monkeypatch):
    """Before a block (text or lane) is saved, the window shows "Saving...", painted and with
    the event queue run (not key presses), since the save blocks the event loop until done."""
    from pyqtgraph.Qt import QtWidgets

    events: list[str] = []
    process = QtWidgets.QApplication.processEvents

    def processed(*flags):
        keys_wait = flags == (QtCore.QEventLoop.ExcludeUserInputEvents,)
        events.append("processed, keys wait" if keys_wait else "processed all")
        process(*flags)

    monkeypatch.setattr(QtWidgets.QApplication, "processEvents", processed)
    with _window(tmp_path, [_grasp_block(n=2), _lane("a")], Clock()) as win:
        repaint, end_trial = win.prompter.repaint, win.session.end_trial

        def painted():
            events.append(f"painted {win.prompter.message}")
            repaint()

        def saved(**kw):
            events.append(f"saving, showing {win.prompter.message}")
            assert win.prompter.t0_perf is None  # the run has ended: nothing scrolls on
            if win.prompter.text_mode:
                assert win.prompter.text_view == TextView(IDLE)
            return end_trial(**kw)

        win.prompter.repaint, win.session.end_trial = painted, saved
        for key in (Qt.Key_Shift, Qt.Key_Return, Qt.Key_Right, Qt.Key_Shift, Qt.Key_Return):
            _key(win, key)
        assert "Saving" not in win.prompter.message and not win.running
    assert (
        events
        == [
            "painted Saving...",
            "processed, keys wait",
            "saving, showing Saving...",
            "processed all",  # the keys pressed during the save, delivered and dropped
        ]
        * 2
    )
    assert [r["label"] for r in _records(tmp_path)] == ["tuna_can", "a"]


@pytest.mark.parametrize("block", ["text", "lane"])
def test_keys_pressed_while_saving_are_dropped(tmp_path, block):
    """Shift or Esc pressed at the "Saving..." screen neither starts the next block nor
    quits once the save returns: they are delivered after it, and dropped."""
    from pyqtgraph.Qt import QtWidgets

    blocks = (
        [_grasp_block(n=2), _grasp_block("mug", n=2)]
        if block == "text"
        else [_lane("a"), _lane("b")]
    )
    with _window(tmp_path, blocks, Clock()) as win:
        win.show()
        end_trial = win.session.end_trial

        def saved(**kw):
            for key in (Qt.Key_Shift, Qt.Key_Escape):
                event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key, Qt.NoModifier)
                QtWidgets.QApplication.postEvent(win, event)
            return end_trial(**kw)

        win.session.end_trial = saved
        _key(win, Qt.Key_Shift)
        _key(win, Qt.Key_Return)
        assert not win.running and win.isVisible() and not win._saving
        _key(win, Qt.Key_Shift)  # keys work again once saved
        assert win.running


# --- Start screens and the terms slide ---------------------------------------------------------

TERMS = [
    {"term": "Light grasp", "text": "touch the object lightly"},
    {"term": "Pick up", "text": "lift the object off the table"},
]
TERMS_SLIDE = (
    "Terms\n\nLight grasp: touch the object lightly\n\nPick up: lift the object off the table"
    "\n\nShift to go on"
)
TERMS_BOLD = {2: len("Light grasp"), 4: len("Pick up")}  # line index: leading characters


def _start_block(name: str, task_set: str, n: int, est_s: float, set_out: list[str],
                 task: str, terms: list[dict] | None = None) -> TextBlock:  # fmt: skip
    """A block as the generator writes it: ``params["start"]`` (and the terms), no
    ``instructions`` or ``object_name``."""
    prompts = [TextPrompt(f"{name}\nPrompt {i}", name, est_s=est_s) for i in range(n)]
    params = {"start": {"set_out": set_out, "task": task}}
    if terms is not None:
        params["terms"] = terms
    return TextBlock(name, task_set, "self_paced", prompts, params)


def test_start_screens_are_one_line_per_item(tmp_path):
    """The grasp, natural and calibration start screens: one line per item with a blank line
    between, no block name (the header has it) and no key instructions but Shift; a condition
    without a label shows its name; an older block file keeps its older screen."""
    from deftemg_acquisition.tasks.textgen import GRASP_TASK, NATURAL_TASK

    grasp = _start_block("S01_b02_tuna_can", "grasp", 32, 9.0, ["Tuna can"], GRASP_TASK)
    grasp.params["conditions"] = ["bottle_fill_rubber"]
    items = ["Aquaphor tube", "Medium cream jar", "Small coffee mug", "Large coffee mug"]
    natural = _start_block("S01_b03_containers", "natural", 5, 16.0, items, NATURAL_TASK)
    blocks = [grasp, natural, grasp_calibration_block(), _grasp_block(n=2)]
    with _window(tmp_path, blocks, conditions={"bottle_fill_rubber": "filled"}) as win:
        screens, headers = [], []
        for _ in blocks:
            screens.append(win.prompter.message)
            headers.append(win.prompter.header)
            _key(win, Qt.Key_Right)
        assert win.current == 0 and not win.showing_terms  # no terms: no slide
    assert headers[:2] == ["Block 1/4: S01_b02_tuna_can   (0/4 blocks done)",
                           "Block 2/4: S01_b03_containers   (0/4 blocks done)"]  # fmt: skip
    assert [s.split("\n\n") for s in screens[:3]] == [
        [
            "Set out: Tuna can",
            "Task: grasp as prompted, the hand shows which fingers",
            "32 prompts, about 9 min",
            "Conditions: bottle_fill_rubber = filled",
            "Shift to start",
        ],
        [
            "Set out:\nAquaphor tube\nMedium cream jar\nSmall coffee mug\nLarge coffee mug",
            "Task: do each task as written",
            "5 prompts, about 2 min",
            "Shift to start",
        ],
        [
            "Set out:\nHardcover book\nBall",
            "Task: brief maximum grasps, as hard as is comfortable",
            "Relax in between; swap objects when prompted",
            "4 prompts, about 1 min",
            "Shift to start",
        ],
    ]
    assert screens[3] == (
        "tuna_can\nPlace: Tuna can\n2 prompts, about 1 min\n\n"
        "Put the tuna can in front of the subject.\n\nPress Shift to start"
    )


def test_a_resumed_start_screen_names_where_it_resumes(tmp_path):
    clock = Clock()
    block = _start_block("mugs", "grasp", 3, 9.0, ["Small coffee mug"], "grasp")
    with _window(tmp_path, [block], clock) as win:
        win.start_block()
        _do_prompt(win, clock)
        _go_and_end_early(win, clock)
        assert win.prompter.message.split("\n\n") == [
            "Set out: Small coffee mug", "Task: grasp",
            "Resumes at prompt 2 of 3\n2 prompts left, about 1 min", "Shift to start",
        ]  # fmt: skip


def _labelled(block: TextBlock, labels: dict[str, str | None]) -> TextBlock:
    """``block`` needing the conditions in ``labels``, each described with its label (None:
    described as before labels)."""
    block.params["conditions"] = list(labels)
    block.params["condition_questions"] = {
        name: {"ask": f"{name}?", "choices": ["filled", "empty"],
               **({"label": label} if label is not None else {})}
        for name, label in labels.items()
    }  # fmt: skip
    return block


def test_conditions_show_their_labels_on_the_start_screen_and_during_the_block(tmp_path):
    """A labelled condition reads "Rubber bottle: filled" (no "Conditions:", the label says
    what it is), before and during the block; the metadata keeps the names."""
    clock = Clock()
    labels = {"bottle_fill_rubber": "Rubber bottle", "bottle_fill_metal": "Metal bottle"}
    block = _labelled(_start_block("bottle", "grasp", 2, 9.0, ["Water bottle (rubber)"], "grasp"),
                      labels)  # fmt: skip
    conditions = {"bottle_fill_rubber": "filled", "bottle_fill_metal": "empty"}
    with _window(tmp_path, [block], clock, conditions=conditions) as win:
        assert win.prompter.message.split("\n\n") == [
            "Set out: Water bottle (rubber)", "Task: grasp", "2 prompts, about 1 min",
            "Rubber bottle: filled, Metal bottle: empty", "Shift to start",
        ]  # fmt: skip
        win.start_block()
        assert win.prompter.text_view.conditions == "Rubber bottle: filled, Metal bottle: empty"
        _key(win, Qt.Key_Enter)
    assert _records(tmp_path)[0]["metadata"]["conditions"] == conditions


@pytest.mark.parametrize(
    "questions",
    [
        {"bottle_fill_rubber": None},  # described before labels
        {"bottle_fill_rubber": "  "},  # a blank label
        "not a table",  # described badly: the names, as before
    ],
)
def test_a_condition_without_a_label_shows_its_name(tmp_path, questions):
    block = _start_block("bottle", "grasp", 2, 9.0, ["Water bottle (rubber)"], "grasp")
    if isinstance(questions, dict):
        _labelled(block, questions)
    else:
        block.params["conditions"] = ["bottle_fill_rubber"]
        block.params["condition_questions"] = questions
    with _window(tmp_path, [block], conditions={"bottle_fill_rubber": "filled"}) as win:
        assert "\n\nConditions: bottle_fill_rubber = filled\n\n" in win.prompter.message
        win.start_block()
        assert win.prompter.text_view.conditions == "bottle_fill_rubber = filled"
        _key(win, Qt.Key_Enter)


def test_some_conditions_without_a_label_keep_the_conditions_prefix(tmp_path):
    block = _labelled(_start_block("bottle", "grasp", 2, 9.0, ["Bottle"], "grasp"),
                      {"bottle_fill_rubber": "Rubber bottle", "lid": None})  # fmt: skip
    with _window(tmp_path, [block], conditions={"bottle_fill_rubber": "filled", "lid": "on"}) as w:
        assert "\n\nConditions: Rubber bottle: filled, lid = on\n\n" in w.prompter.message


def test_a_done_start_screen_says_so_on_its_own_line(tmp_path):
    """With no block-name line, "(already done)" is an item of its own, first."""
    clock = Clock()
    blocks = [_start_block("mugs", "grasp", 2, 9.0, ["Small coffee mug"], "grasp"),
              _start_block("ball", "grasp", 1, 9.0, ["Ball"], "grasp")]  # fmt: skip
    with _window(tmp_path, blocks, clock) as win:
        win.start_block()
        _do_prompt(win, clock)
        _do_prompt(win, clock)
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert win.completed == {0} and win.current == 1
        assert "already done" not in win.prompter.message
        _key(win, Qt.Key_Left)
        assert win.prompter.header == "Block 1/2: mugs   (1/2 blocks done)"
        assert win.prompter.message.split("\n\n") == [
            "(already done)", "Set out: Small coffee mug", "Task: grasp",
            "2 prompts, about 1 min", "Shift to start",
        ]  # fmt: skip


def test_the_terms_slide_comes_first_and_t_shows_it_again(tmp_path):
    """The slide (from the first block with terms) is shown when the window opens, over the
    first block's screen; Shift, Space or Enter close it without starting a block, other keys
    do nothing; T shows it again between blocks, never while a block runs."""
    clock = Clock()
    first = _grasp_block(n=2)  # an older block file: no terms
    blocks = [first, _start_block("ball", "grasp", 2, 9.0, ["Ball"], "grasp", TERMS),
              _start_block("mug", "grasp", 2, 9.0, ["Mug"], "grasp", TERMS[:1])]  # fmt: skip
    with _window(tmp_path, blocks, clock) as win:
        assert win.showing_terms and win.prompter.message == TERMS_SLIDE
        assert win.prompter.message_bold == TERMS_BOLD  # each term, on its own line
        assert win.prompter.text_mode and win.prompter.header.startswith("Block 1/3: tuna_can")
        for key in (Qt.Key_Right, Qt.Key_Left, Qt.Key_T, Qt.Key_R, Qt.Key_A):
            _key(win, key)
            assert win.showing_terms and win.current == 0 and not win.running
        idle = None
        for close in (Qt.Key_Shift, Qt.Key_Space, Qt.Key_Return):
            _key(win, close)
            assert not win.showing_terms and not win.running and win.current == 0
            assert idle is None or win.prompter.message == idle
            idle = win.prompter.message
            assert idle.startswith("tuna_can\nPlace: Tuna can")  # the first block's own screen
            assert win.prompter.message_bold == {}  # nothing bold left over
            _key(win, Qt.Key_T)
            assert win.showing_terms and win.prompter.message == TERMS_SLIDE
            assert win.prompter.message_bold == TERMS_BOLD
        _key(win, Qt.Key_Space)
        _key(win, Qt.Key_Shift)  # now Shift starts the block
        assert win.running and win.prompter.text_view.phase == PREVIEW
        _key(win, Qt.Key_T)
        assert win.running and not win.showing_terms and win.prompter.text_view.phase == PREVIEW
        _do_prompt(win, clock)
        _do_prompt(win, clock)
        _key(win, Qt.Key_T)  # at the block's DONE screen too
        assert win.running and not win.showing_terms and win.prompter.text_view.phase == DONE
        clock.advance(1.5)
        _key(win, Qt.Key_Space)
        _wait(win, clock, TAIL_S)
        assert not win.running and win.current == 1
        _key(win, Qt.Key_T)
        assert win.showing_terms and win.prompter.message == TERMS_SLIDE
    assert len(_records(tmp_path)) == 1  # the slide never started a block


def test_without_terms_there_is_no_slide_and_t_does_nothing(tmp_path):
    with _window(tmp_path, [_grasp_block(n=1), _lane("a")]) as win:
        msg = win.prompter.message
        assert not win.showing_terms and msg.startswith("tuna_can\n") and win.terms is None
        _key(win, Qt.Key_T)
        assert not win.showing_terms and win.prompter.message == msg


def test_a_malformed_start_or_terms_fails_when_the_window_is_made(tmp_path):
    cases = [
        ("start", {"set_out": [], "task": "grasp"}, r"tuna_can: params\.start\.set_out"),
        ("start", {"set_out": ["Ball"]}, r"tuna_can: params\.start"),
        ("terms", [TERMS[0], TERMS[0]], r"tuna_can: params\.terms"),
        ("terms", [{"term": "Pick up"}], r"tuna_can: params\.terms\[0\]"),
    ]
    for key, value, match in cases:
        block = _grasp_block(n=1)
        block.params[key] = value  # hand-edited after loading
        with pytest.raises(ValueError, match=match):
            with _window(tmp_path, [_grasp_block("ball", n=1), block]):
                pass
    assert not (tmp_path / "s" / "trials.jsonl").exists()  # nothing recorded


def test_esc_or_q_on_the_terms_slide_quits_without_recording(tmp_path):
    for i, key in enumerate((Qt.Key_Escape, Qt.Key_Q)):
        block = _start_block("ball", "grasp", 1, 9.0, ["Ball"], "grasp", TERMS)
        with _window(tmp_path / str(i), [block]) as win:
            win.show()
            assert win.showing_terms and win.isVisible()
            _key(win, key)
            assert not win.isVisible() and not win.running
        assert not (tmp_path / str(i) / "s" / "trials.jsonl").exists()  # nothing recorded


def test_a_lane_blocks_start_screen_is_unchanged(tmp_path):
    with _window(tmp_path, [_lane("a")]) as win:  # 0.5 s of presses + the 0.5 s tail
        assert win.prompter.message == "a\n1 presses, 1 s\n\nPress Shift to start"


def test_bold_message_parts_are_drawn_only_while_that_message_shows():
    """``message_bold`` draws the terms in bold; setting any message clears it, so every other
    screen is drawn as before (the plain path)."""
    pg.mkQApp()
    w = prompter.PrompterWidget()
    w.resize(900, 1000)
    w.message = TERMS_SLIDE
    plain = w.grab().toImage()
    w.message_bold = dict(TERMS_BOLD)
    bold = w.grab().toImage()
    assert bold != plain
    w.message = TERMS_SLIDE
    assert w.message_bold == {} and w.grab().toImage() == plain
    w.message_bold = {0: 0}  # nothing bold: the rich path, drawn without bold
    assert w.grab().toImage() != bold
    assert not w.grab().isNull()


def test_the_shipped_terms_fit_a_small_window(tmp_path):
    """At the smallest window (400 x 600) the shipped catalogue's 7 terms, in bold, shrink until
    the slide fits the area above the hand."""
    from deftemg_acquisition.tasks.catalogue import load_catalogue

    pg.mkQApp()
    terms = load_catalogue(ROOT / "protocols/grasp_natural/catalogue.toml").terms
    assert len(terms) == 7
    blocks = [_start_block("ball", "grasp", 1, 9.0, ["Ball"], "grasp",
                           [{"term": t.term, "text": t.text} for t in terms])]  # fmt: skip
    with _window(tmp_path, blocks) as win:
        w = win.prompter
        w.resize(400, 600)
        assert win.showing_terms and len(w.message_bold) == 7
        lay = w.layout_for_size()
        area = QtCore.QRectF(56, lay.header_h, w.width() - 112, lay.target_y - lay.header_h)
        sizes = []

        def document(font, width):
            sizes.append(font.pointSizeF())
            doc = document.real(font, width)
            document.last = doc
            return doc

        document.real = w._message_document
        w._message_document = document
        assert not w.grab().isNull()
        doc = document.last
        assert doc.size().height() <= area.height() and doc.idealWidth() <= area.width()
        assert len(sizes) > 1  # it had to shrink
