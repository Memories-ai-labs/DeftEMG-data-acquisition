"""The task window running text blocks: keys, saved arrays and metadata, markers, the sync
patch, the grasp calibration, and the idle screen."""

import json
import os
from contextlib import contextmanager

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui  # noqa: E402

from deftemg_acquisition.acquisition import AcquisitionSession, Armband, SessionConfig  # noqa: E402
from deftemg_acquisition.acquisition.armband.sim import SimulatedArmband  # noqa: E402
from deftemg_acquisition.tasks import prompter  # noqa: E402
from deftemg_acquisition.tasks.app import TaskWindow  # noqa: E402
from deftemg_acquisition.tasks.selfpaced import DONE, GO, IDLE, PREVIEW, TextView  # noqa: E402
from deftemg_acquisition.tasks.sequence import Prompt, Sequence  # noqa: E402
from deftemg_acquisition.tasks.textblock import (  # noqa: E402
    TEXT_INT_ARRAYS,
    TEXT_STR_ARRAYS,
    TextBlock,
    TextPrompt,
    grasp_calibration_block,
)

Qt = QtCore.Qt
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
    assert set(TEXT_INT_ARRAYS) | set(TEXT_STR_ARRAYS) | {"block_start_host_ns"} <= keys
    assert not [k for k in keys if k.startswith(("prompt_", "effort", "scroll_start"))]
    assert int(data["block_start_host_ns"]) == start
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
        # Shift starts the block again at prompt 1 (no resume yet)
        _key(win, Qt.Key_Shift)
        assert win.running and win.prompter.text_view.index == 0
        _key(win, Qt.Key_E)
        assert not win.running and win.current == 0
    recs = _records(tmp_path)
    assert [r["metadata"]["completed"] for r in recs] == [False, False]
    data = np.load(recs[0]["npz_path"])
    assert data["text_status"].tolist() == ["ok", "incomplete"]
    assert data["text_end_reason"].tolist() == ["operator", "block_end"]
    assert np.load(recs[1]["npz_path"])["text_prompt_index"].tolist() == [0]


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


def test_grasp_calibration_runs_from_its_schedule(tmp_path):
    clock = Clock()
    cal = grasp_calibration_block()
    with _window(tmp_path, [cal, _grasp_block()], clock) as win:
        n = len(cal.prompts)
        assert f"{n} grasps, {cal.duration_s + TAIL_S:.0f} s" in win.prompter.message
        win.start_block()
        start = clock.ns
        _wait(win, clock, cal.prompts[0].onset_s - 1.0)
        assert win.prompter.text_view.phase == PREVIEW
        assert win.prompter.text_view.countdown_left == pytest.approx(1.0)
        _key(win, Qt.Key_Space)  # nothing in a timed block
        _key(win, Qt.Key_R)
        _wait(win, clock, 1.0)
        assert win.prompter.text_view.phase == GO
        _wait(win, clock, cal.duration_s - cal.prompts[0].onset_s)  # at the last end
        assert win.running and win.prompter.text_view.phase == IDLE
        _wait(win, clock, TAIL_S - 0.01)
        assert win.running
        _wait(win, clock, 0.01)
        assert not win.running and win.completed == {0} and win.current == 1
    rec = _records(tmp_path)[0]
    assert rec["metadata"]["task_set"] == "grasp_calibration" and rec["metadata"]["mode"] == "timed"
    assert rec["metadata"]["completed"] is True
    data = np.load(rec["npz_path"])
    onsets = np.array([round(p.onset_s * S) for p in cal.prompts])
    assert data["text_status"].tolist() == ["ok"] * n
    assert data["text_end_reason"].tolist() == ["timer"] * n
    offsets = np.array([round(p.offset_s * S) for p in cal.prompts])
    assert (data["text_go_host_ns"] - start == onsets).all()
    assert (data["text_end_host_ns"] - start == offsets).all()
    assert (data["text_preview_host_ns"] == data["text_countdown_host_ns"]).all()


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
