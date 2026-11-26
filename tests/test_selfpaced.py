"""The self-paced runner without Qt: keys, timing, redo, early end, resume, saved arrays, and
the live view agreeing with the replayed one."""

import copy

import numpy as np
import pytest

from deftemg_acquisition.tasks.selfpaced import (
    BLOCK_END,
    CANCELLED,
    COUNTDOWN,
    DONE,
    GO,
    IDLE,
    INCOMPLETE,
    OK,
    OPERATOR,
    PREVIEW,
    REDO,
    TIMER,
    Attempt,
    SelfPacedRunner,
    TextView,
    attempts_from_arrays,
    text_view_at,
)
from deftemg_acquisition.tasks.textblock import (
    TEXT_INT_ARRAYS,
    TEXT_STR_ARRAYS,
    TextBlock,
    TextPrompt,
    grasp_calibration_block,
)

HOST0 = 7_000_000_000_000  # the block's start on the fake host clock


class Clock:
    """The injected host clock (int ns), moved by the tests."""

    def __init__(self):
        self.ns = HOST0

    def host_ns(self) -> int:
        return self.ns


def ns(t: float) -> int:
    """The fake host time ``t`` s after the block start."""
    return HOST0 + round(t * 1e9)


def _grasp_block() -> TextBlock:
    prompts = [
        TextPrompt(f"Tuna can, from the top\n{g} grasp", "tuna_can", grasp=g, approach="top",
                   variation="light", highlight=("thumb", "index"))
        for g in ("TI", "TIM", "AD")
    ]  # fmt: skip
    return TextBlock("b01_tuna_can", "grasp", "self_paced", prompts)


def _natural_block() -> TextBlock:
    prompts = [
        TextPrompt("Loosen the wing nut all the way off", "screw_panel", step="loosen"),
        TextPrompt("Tighten the wing nut", "screw_panel", step="tighten"),
        TextPrompt("Open the book, then turn pages until the time is up", "pages", duration_s=20),
    ]
    return TextBlock("b02_natural", "natural", "self_paced", prompts)


def _runner(block=None, **kw) -> tuple[SelfPacedRunner, Clock]:
    clock = Clock()
    block = _grasp_block() if block is None else block
    return SelfPacedRunner(block, host_ns=clock.host_ns, **kw), clock


def at(runner: SelfPacedRunner, clock: Clock, t: float) -> SelfPacedRunner:
    """Move the fake time to ``t`` (s after the block start) and apply what is due."""
    clock.ns = ns(t)
    runner.update()
    return runner


def _col(runner, key):
    return runner.arrays()[key].tolist()


def _do(runner, clock, t, key):
    at(runner, clock, t)
    return {"space": runner.space, "r": runner.redo, "enter": runner.end_block}[key]()


# --- the flow ---


def test_full_flow_through_a_block():
    r, c = _runner()
    assert (r.phase, r.view().phase, r.view().index, r.view().attempt) == (PREVIEW, PREVIEW, 0, 0)
    t = 0.0
    for i in range(3):
        assert _do(r, c, t + 2.0, "space") and r.phase == COUNTDOWN
        at(r, c, t + 5.0)
        assert r.phase == GO and r.view().prompt is r.block.prompts[i]
        assert _do(r, c, t + 9.0, "space")
        t += 9.0
    assert r.phase == DONE and r.view() == TextView(DONE, None, 2, 3, can_redo=True)
    assert not r.finished
    assert _do(r, c, t + 2.0, "space")
    assert r.finished and r.completed
    assert _col(r, "text_prompt_index") == [0, 1, 2]
    assert _col(r, "text_status") == [OK] * 3
    assert _col(r, "text_end_reason") == [OPERATOR] * 3
    assert _col(r, "text_preview_host_ns") == [ns(0), ns(9), ns(18)]
    assert _col(r, "text_countdown_host_ns") == [ns(2), ns(11), ns(20)]
    assert _col(r, "text_go_host_ns") == [ns(5), ns(14), ns(23)]
    assert _col(r, "text_end_host_ns") == [ns(9), ns(18), ns(27)]
    assert not _do(r, c, t + 5.0, "space")  # finished: keys do nothing


def test_markers_in_order():
    r, c = _runner(_natural_block())
    _do(r, c, 1.0, "space")
    at(r, c, 4.0)
    _do(r, c, 6.0, "r")  # redo during Go
    assert r.markers() == ["text|preview|0", "text|countdown|0", "text|go|0|screw_panel:loosen",
                           "text|end|0", "text|preview|0"]  # fmt: skip
    assert r.markers() == []  # drained
    _do(r, c, 7.0, "space")
    _do(r, c, 8.0, "enter")
    assert r.markers() == ["text|countdown|0", "text|end|0"]


def test_space_is_ignored_in_the_countdown_and_for_the_debounce_after_go():
    r, c = _runner()
    _do(r, c, 1.0, "space")
    assert not _do(r, c, 2.0, "space") and not _do(r, c, 3.5, "space")
    assert r.phase == COUNTDOWN and len(r.attempts) == 1
    at(r, c, 4.0)
    assert r.phase == GO
    assert not _do(r, c, 4.5, "space") and r.phase == GO  # a double press can't end it
    assert _do(r, c, 5.0, "space") and r.phase == PREVIEW  # debounce over (1.0 s)


def test_space_is_ignored_for_the_debounce_after_an_end():
    r, c = _runner(go_debounce_s=2.0)
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")  # end at 5.0
    assert not _do(r, c, 6.5, "space") and r.phase == PREVIEW and r.view().index == 1
    assert _do(r, c, 7.0, "space") and r.phase == COUNTDOWN


def test_double_tap_on_the_last_end_cannot_finish_the_block():
    r, c = _runner(_grasp_block(), start_index=2)
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")
    assert r.phase == DONE
    assert not _do(r, c, 5.5, "space") and not r.finished
    assert _do(r, c, 6.0, "space") and r.finished and r.completed


def test_redo_after_the_last_prompt_redoes_it():
    r, c = _runner(start_index=2)
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")
    assert r.phase == DONE
    assert _do(r, c, 5.5, "r")  # no debounce on r
    assert r.phase == PREVIEW and r.view().index == 2 and r.view().attempt == 1
    _do(r, c, 6.0, "space")
    _do(r, c, 11.0, "space")
    _do(r, c, 13.0, "space")
    assert r.finished and r.completed
    assert _col(r, "text_prompt_index") == [2, 2]
    assert _col(r, "text_status") == [REDO, OK]
    assert _col(r, "text_end_host_ns") == [ns(5), ns(11)]  # the voided end is kept
    assert _col(r, "text_preview_host_ns") == [ns(0), ns(5.5)]


@pytest.mark.parametrize("countdown_s", [3.0, 2.0])
def test_countdown_length_and_go_time(countdown_s):
    r, c = _runner(countdown_s=countdown_s)
    _do(r, c, 1.0, "space")
    at(r, c, 1.0 + countdown_s - 0.5)
    v = r.view()
    assert v.phase == COUNTDOWN and v.countdown_left == pytest.approx(0.5)
    at(r, c, 1.0 + countdown_s)
    assert r.phase == GO
    assert r.attempts[0].go_ns == r.attempts[0].countdown_ns + round(countdown_s * 1e9)


def test_a_zero_countdown_goes_at_once():
    r, c = _runner(countdown_s=0.0)
    assert _do(r, c, 1.0, "space")
    assert r.phase == GO == r.view().phase and r.attempts[0].go_ns == ns(1)


def test_go_time_is_scheduled_not_when_the_runner_noticed():
    r, c = _runner()
    _do(r, c, 1.0, "space")
    at(r, c, 4.75)  # a late update
    assert r.attempts[0].go_ns == ns(4.0)


def _to_last_go(r, c) -> float:
    """Run the natural block's first two prompts and start the fixed-duration one; its Go."""
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")
    _do(r, c, 6.0, "space")
    _do(r, c, 11.0, "space")
    _do(r, c, 12.0, "space")
    at(r, c, 15.0)
    assert r.phase == GO and r.view().index == 2
    return 15.0


def test_fixed_duration_ends_by_itself():
    r, c = _runner(_natural_block())
    go = _to_last_go(r, c)
    at(r, c, go + 19.5)
    assert r.phase == GO and r.view().time_left == pytest.approx(0.5)
    at(r, c, go + 20.25)
    assert r.phase == DONE
    last = r.attempts[-1]
    assert (last.end_ns, last.end_reason, last.status) == (ns(go + 20), TIMER, OK)
    assert not _do(r, c, go + 20.75, "space")  # debounce from the scheduled end
    assert _do(r, c, go + 21.0, "space") and r.completed


def test_space_at_the_exact_end_of_a_fixed_duration_finds_it_ended_by_the_timer():
    """A deadline is due at its ns: the timer wins a tie with Space, in the runner, the saved
    row and the view alike."""
    block = TextBlock("b03_pages", "natural", "self_paced",
                      [TextPrompt("Turn pages", "pages", duration_s=2.0)])  # fmt: skip
    r, c = _runner(block)
    _do(r, c, 1.499, "space")
    at(r, c, 4.499)
    go = r.attempts[0].go_ns
    assert r.phase == GO and go == ns(4.499)
    c.ns = go + 2_000_000_000  # Space lands exactly at Go + duration
    assert text_view_at(block, r.attempts, c.ns).phase == DONE  # the view, before any update
    assert not r.space()  # ignored: the timer ended it, and the debounce runs from that end
    a = r.attempts[0]
    assert (a.end_ns, a.end_reason, a.status) == (go + 2_000_000_000, TIMER, OK)
    assert r.phase == DONE and r.view().phase == DONE and not r.finished
    # one ns earlier, Space ends it
    r2, c2 = _runner(block)
    _do(r2, c2, 1.499, "space")
    c2.ns = r2.attempts[0].countdown_ns + 3_000_000_000 + 2_000_000_000 - 1
    assert r2.space()
    a = r2.attempts[0]
    assert (a.end_ns, a.end_reason) == (c2.ns, OPERATOR)


def test_countdown_left_follows_a_known_go_not_countdown_s():
    """A replay given the wrong countdown_s still counts down to the saved Go; only a cancelled
    countdown (no Go) needs countdown_s."""
    block = _grasp_block()
    went = Attempt(0, ns(0), ns(1), ns(4), ns(6), OPERATOR, OK)
    v = text_view_at(block, [went], ns(3), countdown_s=10.0)
    assert (v.phase, v.countdown_left) == (COUNTDOWN, pytest.approx(1.0))
    cancelled = Attempt(0, ns(0), ns(1), -1, ns(3), OPERATOR, CANCELLED)
    v = text_view_at(block, [cancelled], ns(2), countdown_s=10.0)
    assert (v.phase, v.countdown_left) == (COUNTDOWN, pytest.approx(9.0))


def test_fixed_duration_ends_early_by_space():
    r, c = _runner(_natural_block())
    go = _to_last_go(r, c)
    assert _do(r, c, go + 5.0, "space")
    last = r.attempts[-1]
    assert (last.end_ns, last.end_reason, last.status) == (ns(go + 5), OPERATOR, OK)
    assert r.phase == DONE


# --- redo and early end ---


def test_redo_after_an_end_voids_it_and_previews_the_same_prompt():
    r, c = _runner()
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")
    assert r.view().index == 1
    assert _do(r, c, 6.0, "r")
    assert r.phase == PREVIEW and r.view().index == 0
    assert [(a.index, a.status) for a in r.attempts] == [(0, REDO), (0, "")]
    assert r.attempts[0].end_ns == ns(5)  # its end is kept
    assert r.attempts[1].preview_ns == ns(6)  # the next prompt's preview was dropped
    assert not _do(r, c, 6.5, "r")  # already voided: nothing more to void
    assert r.markers()[-1] == "text|preview|0"


def test_redo_during_go():
    r, c = _runner()
    _do(r, c, 0.0, "space")
    _do(r, c, 4.0, "r")
    a, b = r.attempts
    assert (a.go_ns, a.end_ns, a.end_reason, a.status) == (ns(3), ns(4), OPERATOR, REDO)
    assert (b.index, b.preview_ns, r.phase) == (0, ns(4), PREVIEW)


def test_redo_during_the_countdown_cancels_it():
    r, c = _runner()
    _do(r, c, 0.0, "space")
    _do(r, c, 2.0, "r")
    a = r.attempts[0]
    assert (a.go_ns, a.end_ns, a.status) == (-1, ns(2), CANCELLED)
    assert r.phase == PREVIEW and r.view().index == 0
    at(r, c, 3.5)
    assert r.phase == PREVIEW  # no Go from the cancelled countdown


def test_redo_before_the_first_attempt_does_nothing():
    r, c = _runner()
    assert not _do(r, c, 1.0, "r")
    assert len(r.attempts) == 1 and r.markers() == ["text|preview|0"]


@pytest.mark.parametrize(
    "keys, row",
    [
        ([], (-1, -1)),  # in the preview
        ([(1.0, "space")], (ns(1), -1)),  # in the countdown
        ([(1.0, "space"), (4.0, None)], (ns(1), ns(4))),  # in Go
    ],
)
def test_early_block_end(keys, row):
    r, c = _runner()
    for t, key in keys:
        _do(r, c, t, key) if key else at(r, c, t)
    assert r.end_block() is False
    a = r.attempts[-1]
    assert (a.countdown_ns, a.go_ns) == row
    assert (a.end_reason, a.status) == (BLOCK_END, INCOMPLETE)
    assert r.finished and not r.completed and r.phase == IDLE and r.view().phase == IDLE
    assert not r.space() and not r.redo() and r.end_block() is False


def test_early_end_keeps_earlier_attempts():
    r, c = _runner()
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")
    _do(r, c, 6.0, "space")
    at(r, c, 9.0)
    _do(r, c, 10.0, "enter")
    assert _col(r, "text_status") == [OK, INCOMPLETE]
    assert _col(r, "text_end_reason") == [OPERATOR, BLOCK_END]


def test_end_block_after_the_last_end_completes():
    r, c = _runner(start_index=2)
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")
    assert _do(r, c, 5.5, "enter") is True
    assert r.finished and r.completed and _col(r, "text_status") == [OK]


# --- resume ---


def test_start_index_begins_at_that_prompt_and_saves_only_new_attempts():
    r, c = _runner(start_index=1)
    v = r.view()
    assert (v.phase, v.index, v.prompt) == (PREVIEW, 1, r.block.prompts[1])
    for t in (0.0, 6.0):
        _do(r, c, t, "space")
        _do(r, c, t + 5.0, "space")
    assert _do(r, c, 12.0, "space") and r.completed
    assert _col(r, "text_prompt_index") == [1, 2]  # absolute indices


def test_resume_index_after_an_early_end():
    r, c = _runner()
    assert r.resume_index() == 0
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")  # prompt 0 ok
    _do(r, c, 6.0, "space")
    _do(r, c, 11.0, "space")  # prompt 1 ok
    _do(r, c, 12.0, "r")  # ... redone
    _do(r, c, 13.0, "space")
    _do(r, c, 18.0, "enter")
    assert r.resume_index() == 1


def test_resume_index_is_unchanged_by_a_run_with_no_new_ok():
    r, c = _runner(start_index=2)
    _do(r, c, 0.0, "space")
    at(r, c, 4.0)
    _do(r, c, 5.0, "r")
    _do(r, c, 6.0, "enter")
    assert r.resume_index() == 2


def test_pair_resumes_at_tighten_after_loosen_ok():
    r, c = _runner(_natural_block())
    _do(r, c, 0.0, "space")
    _do(r, c, 5.0, "space")  # loosen ok
    _do(r, c, 6.0, "enter")
    assert r.resume_index() == 1
    r2, c2 = _runner(_natural_block())
    _do(r2, c2, 0.0, "space")
    _do(r2, c2, 4.0, "enter")  # during loosen
    assert r2.resume_index() == 0


@pytest.mark.parametrize("bad", [3, 4, -1, True, 1.0])
def test_start_index_outside_the_block_raises(bad):
    with pytest.raises(ValueError, match="start_index"):
        _runner(start_index=bad)


def test_a_timed_block_is_rejected(timed_calibration):
    with pytest.raises(ValueError, match="timed block can only be replayed"):
        _runner(timed_calibration)


@pytest.mark.parametrize("kw", [{"countdown_s": -1}, {"go_debounce_s": float("nan")}])
def test_bad_timing_raises(kw):
    with pytest.raises(ValueError):
        _runner(**kw)


# --- saved arrays ---


def _messy_run() -> SelfPacedRunner:
    """A run with every status: a cancel, a redo after an end, a redo during Go, an early end
    (5 rows: the preview left by the redo after an end is dropped)."""
    r, c = _runner()
    script = [(1.0, "space"), (2.0, "r"), (3.0, "space"), (10.0, "space"), (11.0, "r"),
              (12.0, "space"), (20.0, "space"), (21.0, "space"), (25.0, "r"), (26.0, "space"),
              (30.0, "enter")]  # fmt: skip
    for t, key in script:
        _do(r, c, t, key)
    return r


def test_arrays_have_one_length_and_the_documented_dtypes(tmp_path):
    r = _messy_run()
    arrays = r.arrays({1: ns(6.01), 3: ns(24.02)})
    assert set(arrays) == {*TEXT_INT_ARRAYS, *TEXT_STR_ARRAYS, "block_start_host_ns",
                           "text_finish_host_ns"}  # fmt: skip
    for k in TEXT_INT_ARRAYS:
        assert arrays[k].dtype == np.int64 and arrays[k].shape == (5,), k
    for k in TEXT_STR_ARRAYS:
        assert arrays[k].dtype.kind == "U" and arrays[k].shape == (5,), k
    assert (
        arrays["block_start_host_ns"].dtype == np.int64 and arrays["block_start_host_ns"] == HOST0
    )
    finish = arrays["text_finish_host_ns"]  # a scalar: the early end (Enter) at 30
    assert finish.dtype == np.int64 and finish.shape == () and finish == ns(30)
    assert arrays["text_status"].tolist() == [CANCELLED, REDO, OK, REDO, INCOMPLETE]
    assert arrays["text_prompt_index"].tolist() == [0, 0, 0, 1, 1]
    assert arrays["text_go_paint_ns"].tolist() == [-1, ns(6.01), -1, ns(24.02), -1]
    assert arrays["text_grasp"].tolist() == ["TI", "TI", "TI", "TIM", "TIM"]
    assert arrays["text_step"].tolist() == [""] * 5  # None is ""
    path = tmp_path / "trial.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as loaded:
        assert attempts_from_arrays(loaded) == r.attempts


def test_arrays_of_no_attempts_keep_their_dtypes():
    arrays = _grasp_block().block_arrays([], HOST0)
    assert arrays["text_finish_host_ns"] == -1  # not finished
    assert all(arrays[k].dtype == np.int64 and arrays[k].size == 0 for k in TEXT_INT_ARRAYS)
    assert all(arrays[k].dtype.kind == "U" and arrays[k].size == 0 for k in TEXT_STR_ARRAYS)


# --- the grasp calibration ---


def test_grasp_calibration_runs_like_a_self_paced_block():
    """Each maximum: preview, Space, countdown, Go, ends by itself after hold_s; the next is
    previewed (the object swap is the ball's preview); r redoes one; Space at DONE finishes."""
    block = grasp_calibration_block()
    r, c = _runner(block)
    assert r.view().phase == PREVIEW and r.space()  # Space starts the countdown
    assert at(r, c, 2.9).phase == COUNTDOWN
    v = at(r, c, 3.5).view()
    assert (v.phase, v.index) == (GO, 0) and v.time_left == pytest.approx(1.0)
    v = at(r, c, 4.5).view()  # Go at 3.0, the 1.5 s hold ended at 4.5
    assert (r.phase, v.phase, v.index) == (PREVIEW, PREVIEW, 1)
    assert r.attempts[0].end_ns == ns(4.5) and r.attempts[0].end_reason == TIMER
    assert _do(r, c, 5.0, "r")  # redo the first maximum
    t = 5.0
    for _ in range(len(block.prompts)):
        assert _do(r, c, t + 1.0, "space")
        at(r, c, t + 1.0 + 3.0 + 1.5)
        t += 1.0 + 3.0 + 1.5
    assert r.phase == DONE and not r.finished
    assert _do(r, c, t + 1.0, "space") and r.finished and r.completed
    assert _col(r, "text_prompt_index") == [0, 0, 1, 2, 3]
    assert _col(r, "text_status") == [REDO, OK, OK, OK, OK]
    assert _col(r, "text_end_reason") == [TIMER] * 5
    assert r.resume_index() == 4


def test_space_ends_a_maximum_early_after_the_debounce():
    r, c = _runner(grasp_calibration_block())
    _do(r, c, 0.0, "space")  # Go at 3.0
    assert not _do(r, c, 3.5, "space")  # debounce: 1.0 s after Go
    assert _do(r, c, 4.2, "space")  # before the 1.5 s hold is up
    assert r.attempts[0].end_reason == OPERATOR and r.attempts[0].end_ns == ns(4.2)


def _timed_attempts(block: TextBlock) -> list[Attempt]:
    """The attempts the old timed runner saved: each previewed (and counted down) from the
    previous prompt's end, Go at its onset, the end at its offset."""
    begin = [0.0] + [p.offset_s for p in block.prompts[:-1]]
    return [Attempt(i, ns(b), ns(b), ns(p.onset_s), ns(p.offset_s), TIMER, OK)
            for i, (b, p) in enumerate(zip(begin, block.prompts, strict=True))]  # fmt: skip


def test_a_saved_timed_calibration_replays_on_its_schedule(timed_calibration):
    block = timed_calibration
    attempts = _timed_attempts(block)
    first, last = block.prompts[0], block.prompts[-1]

    def view(t):
        return text_view_at(block, attempts, ns(t), start_ns=HOST0)

    v = view(first.onset_s - 1.25)
    assert (v.phase, v.index, v.attempt) == (PREVIEW, 0, 0)
    assert v.countdown_left == pytest.approx(1.25)
    v = view(first.onset_s + 0.5)
    assert (v.phase, v.index, v.attempt) == (GO, 0, 0)
    assert v.time_left == pytest.approx(first.duration_s - 0.5)
    v = view(first.offset_s)
    assert (v.phase, v.index) == (PREVIEW, 1) and v.countdown_left > 0
    assert view(last.offset_s).phase == IDLE  # after its last prompt: no DONE
    assert view(-0.1).phase == IDLE


@pytest.mark.parametrize("during", ["preview", "go"])
def test_a_saved_timed_calibration_ended_early_replays_idle(timed_calibration, during):
    p = timed_calibration.prompts[1]
    t = p.onset_s - 0.5 if during == "preview" else p.onset_s + 0.5
    attempts = _timed_attempts(timed_calibration)[:2]
    a = attempts[1]
    a.end_ns, a.end_reason, a.status = ns(t), BLOCK_END, INCOMPLETE
    if during == "preview":
        a.go_ns = -1
    view = text_view_at(timed_calibration, attempts, ns(t - 0.1), start_ns=HOST0)
    assert view.phase == (PREVIEW if during == "preview" else GO)
    assert text_view_at(timed_calibration, attempts, ns(t), start_ns=HOST0).phase == IDLE


# --- live and replayed views agree ---


def _replayed(r: SelfPacedRunner, t_ns: int) -> TextView:
    arrays = r.arrays()
    return text_view_at(r.block, attempts_from_arrays(arrays), t_ns,
                        start_ns=int(arrays["block_start_host_ns"]), countdown_s=r.countdown_s,
                        conditions=r.conditions,
                        finish_ns=int(arrays["text_finish_host_ns"]))  # fmt: skip


@pytest.mark.parametrize("make", [_grasp_block, _natural_block, grasp_calibration_block])
def test_live_view_agrees_with_the_replayed_view(make):
    block = make()
    clock = Clock()
    r = SelfPacedRunner(block, conditions="bottle: filled", host_ns=clock.host_ns)
    keys = {1.0: "space", 2.0: "r", 3.0: "space", 10.0: "space", 11.0: "r", 12.0: "space",
            20.0: "space", 21.0: "space", 25.0: "r", 26.0: "space", 40.0: "space",
            41.0: "space", 60.0: "space", 61.0: "r", 62.0: "space", 70.0: "space",
            80.0: "space"}  # fmt: skip
    seen = []
    for t in np.arange(0.0, 81.75, 0.25):  # Space at DONE (60 or 80) finishes; 1.5 s after
        t = float(t)
        clock.ns = ns(t)
        if t % 1.0 == 0.5:  # between ticks: the view follows the schedule without an update
            seen.append((clock.host_ns(), r.view()))
            continue
        r.update()
        if t in keys:
            {"space": r.space, "r": r.redo}[keys[t]]()
        v = r.view()
        assert v.phase == r.phase, t
        seen.append((clock.host_ns(), v))
    assert r.finished and r.completed and r.finish_ns in (ns(60), ns(80))
    phases = {v.phase for _, v in seen}
    assert phases == {PREVIEW, COUNTDOWN, GO, DONE, IDLE}
    tail = [v for t_ns, v in seen if t_ns >= r.finish_ns]
    assert len(tail) >= 7 and all(v == TextView(IDLE, total=v.total, conditions=v.conditions)
                                  for v in tail)  # fmt: skip
    assert not any(v.can_redo for v in tail)
    assert any(v.time_left is not None for _, v in seen) == (make is not _grasp_block)
    for t_ns, v in seen:
        assert v.conditions == "bottle: filled"
        if v.phase in (PREVIEW, COUNTDOWN, GO):
            assert v.attempt >= 0 and v.prompt is block.prompts[v.index]
        assert _replayed(r, t_ns) == v, (t_ns - HOST0) / 1e9


@pytest.mark.parametrize("start_index", [0, 1])
def test_can_redo_says_whether_redo_would_act(start_index):
    """At every moment (between updates too), the view's ``can_redo`` is what ``redo()``
    returns then (tried on a copy): not at the first prompt (of a resumed block too), nor in
    the preview a redo just started; in the next prompt's preview after an end, the countdown,
    Go and DONE; never once the block has finished (Enter at DONE, or Esc mid-Go: idle). The
    replayed view agrees."""
    block = TextBlock("b04_mixed", "natural", "self_paced", [
        TextPrompt("Open the jar", "jar"),
        TextPrompt("Turn pages", "pages", duration_s=2.0),
        TextPrompt("Close the jar", "jar"),
    ])  # fmt: skip
    r, c = _runner(block, start_index=start_index)
    # From 0: a cancel, a redo in Go, prompt 0 done; prompt 1 (2 s) from 13.5 ends by its timer
    # at 18.5, between updates; r at 19 redoes it from prompt 2's preview; DONE at 30, and
    # Enter finishes it at 32. From 1: Esc in prompt 2's Go at 24 ends it early.
    keys = {1.0: "space", 2.0: "r", 3.0: "space", 7.5: "r", 8.0: "space", 12.0: "space",
            13.5: "space", 19.0: "r", 20.0: "space", 26.0: "space", 30.0: "space",
            32.0: "enter"}  # fmt: skip
    if start_index == 1:
        keys[24.0] = "enter"
    seen = {False: 0, True: 0}
    for t in np.arange(0.0, 34.0, 0.25):
        t = float(t)
        c.ns = ns(t)
        if t % 1.0 != 0.5:  # at x.5 the view follows the schedule without an update
            r.update()
        v = r.view()
        assert v.can_redo == copy.deepcopy(r).redo(), (t, v)
        assert _replayed(r, c.ns) == v, t
        seen[v.can_redo] += 1
        if t in keys:
            {"space": r.space, "r": r.redo, "enter": r.end_block}[keys[t]]()
    assert all(seen.values()) and r.finished and r.completed == (start_index == 0)
    assert r.view().phase == IDLE and r.phase == IDLE


def test_can_redo_at_a_timer_end_before_the_update():
    """The next preview shown at a fixed duration's end, before the runner has ended the
    attempt (its status still ""), offers r already, as it does once updated."""
    block = TextBlock("b03_pages", "natural", "self_paced", [
        TextPrompt("Turn pages", "pages", duration_s=2.0),
        TextPrompt("Close the book", "pages"),
    ])  # fmt: skip
    r, c = _runner(block)
    _do(r, c, 1.0, "space")
    at(r, c, 4.5)
    assert r.phase == GO
    c.ns = r.attempts[0].go_ns + 2_000_000_000  # the timer's end, no update yet
    assert r.attempts[0].status == "" and len(r.attempts) == 1
    v = r.view()
    assert (v.phase, v.index, v.can_redo) == (PREVIEW, 1, True)
    r.update()
    assert r.attempts[0].status == OK and r.view() == v
    assert r.redo() and r.view().can_redo is False  # the redo's own preview: nothing to redo


def test_a_timed_block_never_offers_r(timed_calibration):
    attempts = _timed_attempts(timed_calibration)
    for t in np.arange(0.0, 40.0, 0.5):
        v = text_view_at(timed_calibration, attempts, ns(float(t)), start_ns=HOST0)
        assert not v.can_redo, t


def test_view_before_the_block_starts_is_idle(timed_calibration):
    r, _ = _runner()
    assert text_view_at(r.block, r.attempts, HOST0 - 1).phase == IDLE
    assert text_view_at(timed_calibration, [], HOST0 - 1, start_ns=HOST0).phase == IDLE
    with pytest.raises(ValueError, match="start_ns"):
        text_view_at(timed_calibration, [], HOST0)


def test_view_after_an_end_previews_the_next_prompt_with_its_row():
    block = _grasp_block()
    done = Attempt(0, ns(0), ns(1), ns(4), ns(6), OPERATOR, OK)
    v = text_view_at(block, [done], ns(7))
    assert (v.phase, v.index, v.attempt) == (PREVIEW, 1, 1)


def test_go_view_needs_its_attempt_row():
    with pytest.raises(ValueError, match="attempt"):
        TextView(GO, _grasp_block().prompts[0], 0, 1)
    TextView(GO, _grasp_block().prompts[0], 0, 1, attempt=0)
    TextView(PREVIEW, _grasp_block().prompts[0], 0, 1)  # a hand-made preview may have none
