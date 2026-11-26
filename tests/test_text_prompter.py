"""The prompter in text mode: text blocks drawn without lanes, with the grasp on the hand."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402
from pyqtgraph.Qt import QtCore, QtGui  # noqa: E402

from deftemg_acquisition.tasks.prompter import (  # noqa: E402
    BACKGROUND,
    DIGIT_COLORS,
    FINGERS_CM,
    METER_H,
    PrompterOverlay,
    PrompterWidget,
    hand_paths,
    tip_points,
)
from deftemg_acquisition.tasks.selfpaced import (  # noqa: E402
    COUNTDOWN,
    DONE,
    GO,
    IDLE,
    PREVIEW,
    TextView,
)
from deftemg_acquisition.tasks.sequence import DIGITS  # noqa: E402
from deftemg_acquisition.tasks.textblock import (  # noqa: E402
    GRASP_HIGHLIGHTS,
    HAND,
    TextBlock,
    TextPrompt,
)
from deftemg_acquisition.viz.screen import font_points  # noqa: E402

TEXT = "Tuna can, from the top\nThumb-index grasp\nLight grasp"
LIFT = "Tennis ball, from the side\nThumb-index-middle grasp\nPICK UP\nPick up, place 30 cm away"
LIFT_PROMPT = TextPrompt(
    LIFT, "tennis_ball", grasp="TIM", highlight=("thumb", "index"), bold_line=2
)  # its lift line bold


def _prompt(highlight=("thumb", "index"), **kw) -> TextPrompt:
    return TextPrompt(TEXT, "tuna_can", grasp="TI", approach="top", highlight=highlight, **kw)


def _block(*prompts: TextPrompt) -> TextBlock:
    return TextBlock("tuna_can", "grasp", "self_paced", list(prompts) or [_prompt()])


def _widget(hand="right", size=(500, 800), **kw) -> PrompterWidget:
    pg.mkQApp()
    w = PrompterWidget(hand=hand, **kw)
    w.resize(*size)
    return w


def _render(w: PrompterWidget) -> QtGui.QImage:
    img = QtGui.QImage(w.size(), QtGui.QImage.Format_ARGB32)
    w.render(img)
    return img


def _lit(img: QtGui.QImage, x: float, y: float) -> bool:
    """The highlight colour (sky blue) at a point, faint or strong."""
    r, g, b, _ = img.pixelColor(round(x), round(y)).getRgb()
    return b > 90 and b - r > 50


def _palm(w: PrompterWidget) -> tuple[float, float]:
    """A point on the back of the hand, below the knuckles, on screen."""
    lay = w.layout_for_size()
    mirror = 1 if lay.hand == "right" else -1
    return lay.width / 2 + mirror * 0.8 * lay.cm, lay.hand_top + 9.9 * lay.cm


@pytest.mark.parametrize("hand", ["right", "left"])
def test_tip_points_are_inside_the_hand_under_their_lanes(hand):
    w = _widget(hand)
    lay = w.layout_for_size()
    silhouette, _ = hand_paths(lay)
    tips = tip_points(lay)
    assert set(tips) == set(FINGERS_CM) == set(DIGITS)
    for d, (x, y) in tips.items():
        assert silhouette.contains(QtCore.QPointF(x, y)), d
        assert abs(x - lay.lane_x(d)) < lay.lane_w / 2, d
        assert lay.hand_top <= y < lay.height, d
    # Ordered across the screen as the lanes are (thumb on the left for the right hand)
    xs = [tips[d][0] for d in DIGITS]
    assert xs == sorted(xs, reverse=hand == "left")


@pytest.mark.parametrize("hand", ["right", "left"])
@pytest.mark.parametrize("grasp", ["TI", "TIM", "AD", "Power", "none"])
def test_text_view_highlights_the_grasp_on_the_hand(hand, grasp):
    """TI marks the thumb and index tips, TIM adds the middle, AD all five, Power the whole
    hand; a prompt without a grasp marks nothing."""
    highlight = GRASP_HIGHLIGHTS.get(grasp, ())
    w = _widget(hand)
    w.sequence = _block(_prompt(highlight))
    w.text_view = TextView(PREVIEW, _prompt(highlight), index=0, total=1)
    img = _render(w)
    tips = tip_points(w.layout_for_size())
    marked = DIGITS if highlight == HAND else highlight
    assert [d for d in DIGITS if _lit(img, *tips[d])] == list(marked)
    assert _lit(img, *_palm(w)) == (highlight == HAND)


@pytest.mark.parametrize("phase", [PREVIEW, COUNTDOWN, GO])
def test_highlight_is_on_ahead_of_and_during_the_task(phase):
    w = _widget()
    w.text_view = TextView(phase, _prompt(), index=0, total=3, attempt=0, countdown_left=2.5)
    img = _render(w)
    tips = tip_points(w.layout_for_size())
    assert _lit(img, *tips["thumb"]) and _lit(img, *tips["index"])


@pytest.mark.parametrize("phase", [IDLE, DONE])
def test_no_highlight_without_a_prompt_under_way(phase):
    w = _widget()
    w.text_view = TextView(phase, _prompt(), index=0, total=3)
    img = _render(w)
    tips = tip_points(w.layout_for_size())
    assert not any(_lit(img, *tips[d]) for d in DIGITS)
    assert w._highlight_cache is None


def test_highlight_image_is_drawn_once_per_size_hand_and_grasp():
    w = _widget()
    w.text_view = TextView(PREVIEW, _prompt(), index=0, total=2)
    _render(w)
    cached = w._highlight_cache
    assert cached is not None
    w.text_view = TextView(GO, _prompt(), index=0, total=2, attempt=0)  # same grasp
    _render(w)
    assert w._highlight_cache is cached
    w.text_view = TextView(PREVIEW, _prompt(("thumb", "index", "middle")), index=1, total=2)
    _render(w)
    assert w._highlight_cache is not cached  # another grasp
    cached = w._highlight_cache
    w.resize(450, 760)
    _render(w)
    assert w._highlight_cache is not cached  # another size
    cached = w._highlight_cache
    w.hand = "left"
    _render(w)
    assert w._highlight_cache is not cached  # the other hand
    cached = w._highlight_cache
    _render(w)
    assert w._highlight_cache is cached


def test_text_mode_draws_no_lanes_rings_effort_or_lane_overlays():
    """A text block (no digit prompts, no ``implied_digits``) paints without touching the lane
    code: no lane lines, target rings or effort bar, and lane overlays are not called (nor
    removed)."""
    calls = []

    class LaneOverlay(PrompterOverlay):
        def paint_under(self, p, lay, widget):
            calls.append("under")

        def paint_over(self, p, lay, widget):
            calls.append("over")

    w = _widget()
    overlay = LaneOverlay()
    w.overlays = [overlay]
    w.show_effort, w.effort, w.effort_zones = True, (0.5, "low"), {"low": (0.2, 0.4)}
    w.sequence = _block()
    w.t = 1.0
    assert w.text_mode and w.text_view is None  # a text block alone is enough
    idle = _render(w)
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=0)
    img = _render(w)
    assert calls == [] and w.overlays == [overlay]
    assert w._ring_levels() == []
    lay = w.layout_for_size()
    bg = QtGui.QColor(BACKGROUND).getRgb()
    # The target row (rings) and the effort bar's place are empty
    for d in DIGITS:
        for dy in (0, lay.radius("high")):
            x = lay.lane_x(d)
            assert img.pixelColor(round(x), round(lay.target_y - dy)).getRgb() == bg
    assert lay.meter_h == METER_H  # room was made for the bar, but it is not drawn
    w.show_effort = False
    assert _render(w) == img
    # No capsule anywhere above the hand: no digit colour (GO is green, so the idle screen)
    colours = {QtGui.QColor(c).getRgb()[:3] for c in DIGIT_COLORS.values()}
    above = (idle.pixelColor(x, y).getRgb()[:3] for x in range(0, w.width(), 5)
             for y in range(0, int(lay.hand_top), 5))  # fmt: skip
    assert not colours & set(above)


def test_hand_stays_where_it_is_in_text_mode():
    """With no grasp to show, the hand in text mode is the lane prompter's hand, pixel for
    pixel."""
    lane = _widget()
    ref = _render(lane)
    w = _widget()
    w.sequence = _block(_prompt(()))
    w.text_view = TextView(GO, _prompt(()), index=0, total=1, attempt=0)
    img = _render(w)
    lay = w.layout_for_size()
    for y in range(int(lay.hand_top) - 2, w.height(), 3):
        for x in range(0, w.width(), 3):
            assert img.pixelColor(x, y) == ref.pixelColor(x, y), (x, y)


def _brightest(img: QtGui.QImage, top: float, bottom: float) -> int:
    return max(
        max(img.pixelColor(x, y).getRgb()[:3])
        for x in range(0, img.width(), 2)
        for y in range(int(top), int(bottom), 2)
    )


def test_prompt_text_is_dimmed_in_the_preview_and_go_is_green():
    w = _widget()
    lay = w.layout_for_size()
    w.text_view = TextView(PREVIEW, _prompt(()), index=0, total=4)
    preview = _render(w)
    w.text_view = TextView(GO, _prompt(()), index=0, total=4, attempt=0)
    go = _render(w)
    middle = (lay.header_h + 120, lay.hand_top - 60)  # the prompt text, under the state line
    assert _brightest(go, *middle) > _brightest(preview, *middle) + 60
    green = QtGui.QColor("#009E73").getRgb()[:3]
    state = (go.pixelColor(x, y).getRgb()[:3] for x in range(0, w.width(), 2)
             for y in range(int(lay.header_h), int(lay.header_h + 120), 2))  # fmt: skip
    assert green in set(state)


@pytest.mark.parametrize(
    "view",
    [
        TextView(IDLE),
        TextView(PREVIEW, _prompt(), index=0, total=5, conditions="bottle: filled"),
        TextView(PREVIEW, _prompt(HAND), index=3, total=4, countdown_left=2.2),  # timed block
        TextView(COUNTDOWN, _prompt(), index=1, total=5, countdown_left=0.4),
        TextView(GO, _prompt(), index=4, total=5, attempt=7, time_left=12.3),
        TextView(GO, LIFT_PROMPT, index=2, total=5, attempt=2),
        TextView(DONE, None, index=4, total=5),
    ],
)
@pytest.mark.parametrize("size", [(500, 800), (400, 600)])
def test_every_text_view_paints(view, size):
    """Every phase paints (a raise in paint fails the test via the Qt exception hook), with a
    message over it too."""
    w = _widget(size=size)
    w.sequence, w.text_view = _block(), view
    w.header = "Block 2/9: tuna_can"
    _render(w)
    w.message = "Resumes at prompt 3 of 9. " * 4
    w.warning = "glove disconnected"
    _render(w)


class _SpyPainter:
    """A painter on an image that records each ``drawText``: the text, its font, and where the
    text landed (its bounding rect)."""

    def __init__(self, img: QtGui.QImage):
        self._p = QtGui.QPainter(img)
        self.texts: list[tuple[str, QtGui.QFont, QtCore.QRectF]] = []

    def __getattr__(self, name):
        return getattr(self._p, name)

    def drawText(self, rect, flags, text):  # noqa: N802 (Qt's name)
        self.texts.append((text, self._p.font(), self._p.boundingRect(rect, flags, text)))
        self._p.drawText(rect, flags, text)


def _drawn(w: PrompterWidget, view: TextView) -> list[tuple[str, QtGui.QFont, QtCore.QRectF]]:
    """What the text view's painter writes, in order."""
    w.text_view = view
    img = QtGui.QImage(w.size(), QtGui.QImage.Format_ARGB32)
    spy = _SpyPainter(img)
    w._paint_text_view(spy, w.layout_for_size())
    spy.end()
    return spy.texts


def test_a_prompt_without_a_bold_line_is_one_regular_text():
    w = _widget()
    texts = _drawn(w, TextView(GO, _prompt(), index=0, total=4, attempt=0))
    assert [(t, f.bold()) for t, f, _ in texts if "Tuna" in t] == [(TEXT, False)]


@pytest.mark.parametrize("bold", [0, 2, 3])
@pytest.mark.parametrize("phase", [PREVIEW, GO])
@pytest.mark.parametrize("size", [(500, 800), (400, 600)])
def test_the_bold_line_alone_is_bold_and_the_stack_fits(bold, phase, size):
    """Each line drawn on its own, in order, only ``bold_line`` bold; the stack sits between
    the state line and the hint, inside the side margins, lines not overlapping, also in the
    smallest window the prompter takes (where it shrinks)."""
    w = _widget(size=size)
    assert (w.width(), w.height()) == size
    prompt = TextPrompt(LIFT, "tennis_ball", grasp="TIM", bold_line=bold)
    texts = _drawn(w, TextView(phase, prompt, index=1, total=4, attempt=1))
    lines = LIFT.split("\n")
    state = next(r for t, _, r in texts if t in ("GO", "Rest. Next:"))
    hint = next(r for t, _, r in texts if t.startswith(("Space", "r:")))
    drawn = [(t, f, r) for t, f, r in texts if t in lines]
    assert [(t, f.bold()) for t, f, _ in drawn] == [(ln, i == bold) for i, ln in enumerate(lines)]
    for (_, _, r), (_, _, below) in zip(drawn, drawn[1:], strict=False):
        assert r.bottom() <= below.top() + 1  # stacked in order
    assert drawn[0][2].top() >= state.bottom() - 1 and drawn[-1][2].bottom() <= hint.top() + 1
    assert all(r.left() >= 32 - 1 and r.right() <= w.width() - 32 + 1 for _, _, r in drawn)


def test_the_bold_line_shrinks_with_the_rest_in_a_small_window():
    """One size for the whole stack, the largest that fits: the smallest window the prompter
    takes (400 x 600) draws every line smaller."""
    prompt = LIFT_PROMPT
    sizes = []
    for size in [(900, 1000), (400, 600)]:
        w = _widget(size=size)
        assert (w.width(), w.height()) == size
        texts = _drawn(w, TextView(GO, prompt, index=1, total=4, attempt=1))
        sizes.append({f.pointSizeF() for t, f, _ in texts if t in LIFT.split("\n")})
    (big,), (small,) = sizes
    assert big == font_points(26) and small < big


@pytest.mark.parametrize(
    "view, hint",
    [
        (TextView(PREVIEW, _prompt(), index=2, total=4, can_redo=True),
         "Space to start   r: redo the last prompt"),
        # the block's (or a resumed block's) first prompt, the preview a redo just started
        (TextView(PREVIEW, _prompt(), index=0, total=4), "Space to start"),
        # a timed block's preview (replayed only): no hint, whatever can_redo says
        (TextView(PREVIEW, _prompt(), index=2, total=4, countdown_left=2.2, can_redo=True),
         None),
        (TextView(GO, _prompt(), index=2, total=4, attempt=2, can_redo=True),
         "Space when finished"),
        (TextView(DONE, None, index=3, total=4, can_redo=True),
         "Space to finish the block, r to redo the last one"),
    ],
)  # fmt: skip
def test_the_hint_offers_r_only_when_a_redo_would_act(view, hint):
    texts = [t for t, _, _ in _drawn(_widget(), view)]
    hints = [t for t in texts if "Space" in t or "r:" in t]
    assert hints == ([hint] if hint else [])
    if view.countdown_left is not None:
        assert "Rest. Next, in 3 s:" in texts


def test_text_view_rejects_an_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        TextView("running")


@pytest.fixture
def clock(monkeypatch):
    """The prompter's host clock, set by the test (ns)."""
    now = {"ns": 10_000_000_000}
    monkeypatch.setattr("deftemg_acquisition.tasks.prompter.host_ns", lambda: now["ns"])
    return now


def test_first_go_frame_is_logged_per_attempt(clock):
    w = _widget()
    w.sequence = _block()
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=0)
    _render(w)
    assert w.go_paint_ns == {}  # not running: nothing logged
    w.begin_run(time.perf_counter())
    assert w.onset_paint_ns.size == 0 and w.offset_paint_ns.size == 0
    w.text_view = TextView(PREVIEW, _prompt(), index=0, total=1)
    _render(w)
    assert w.go_paint_ns == {}
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=0)
    clock["ns"] = 11_000_000_000
    _render(w)
    assert w.go_paint_ns == {0: 11_000_000_000}  # the host time of the first Go frame
    clock["ns"] = 11_016_000_000
    _render(w)
    assert w.go_paint_ns == {0: 11_000_000_000}  # the first frame only
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=1)  # a redo
    clock["ns"] = 15_000_000_000
    _render(w)
    assert w.go_paint_ns == {0: 11_000_000_000, 1: 15_000_000_000}
    w.begin_run(time.perf_counter())
    assert w.go_paint_ns == {}  # a new run starts clean


def test_sync_patch_flashes_at_go(clock):
    from deftemg_acquisition.tasks.prompter import SYNC_FLASH_S, SYNC_PATCH_PX

    w = _widget(sync_patch=True)
    w.sequence = _block()
    w.begin_run(time.perf_counter())

    def patch(img):
        return img.pixelColor(w.width() - SYNC_PATCH_PX // 2, SYNC_PATCH_PX // 2).getRgb()[:3]

    w.text_view = TextView(PREVIEW, _prompt(), index=0, total=1)
    assert patch(_render(w)) == (0, 0, 0) and w.flash_paint_ns == []
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=0)
    go_ns = clock["ns"] = 20_000_000_000
    assert patch(_render(w)) == (255, 255, 255)
    assert w.go_paint_ns == {0: go_ns} and w.flash_paint_ns == [go_ns]
    clock["ns"] = go_ns + int(SYNC_FLASH_S * 1e9) - 1_000_000  # 1 ms before the flash ends
    assert patch(_render(w)) == (255, 255, 255) and len(w.flash_paint_ns) == 1  # still on
    clock["ns"] = go_ns + int(SYNC_FLASH_S * 1e9)
    assert patch(_render(w)) == (0, 0, 0)
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=1)  # the redo's Go
    clock["ns"] = go_ns + 2_000_000_000
    assert patch(_render(w)) == (255, 255, 255)
    assert w.flash_paint_ns == [go_ns, go_ns + 2_000_000_000]


def test_lane_blocks_are_unchanged_by_text_mode():
    """A lane block with no text view still draws lanes and rings (text_mode is off)."""
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    w = _widget()
    w.sequence = Sequence("s", "single", [Prompt(1.0, 1.0, ("index",), ("high",))])
    assert not w.text_mode
    assert w._ring_levels() == ["low", "high"]
    img = _render(w)
    lay = w.layout_for_size()
    bg = QtGui.QColor(BACKGROUND).getRgb()
    x, y, r = lay.lane_x("index"), lay.target_y, lay.radius("high")
    span = range(-int(r) - 2, int(r) + 3)
    box = [img.pixelColor(round(x + dx), round(y + dy)).getRgb() for dx in span for dy in span]
    assert any(c != bg for c in box)  # the index lane's target rings
    assert PrompterOverlay.text_mode is False


def test_a_stale_text_view_does_not_turn_a_lane_block_into_text():
    """The sequence's type decides the mode: a lane block with a text view left set still
    draws lanes and gets its prompts' paint arrays from ``begin_run``."""
    from deftemg_acquisition.tasks.sequence import Prompt, Sequence

    prompts = [Prompt(1.0, 1.0, ("index",), ("high",)), Prompt(3.0, 1.0, ("ring",), ("low",))]
    w = _widget()
    w.sequence = Sequence("s", "single", prompts)
    ref = _render(w)
    w.text_view = TextView(GO, _prompt(), index=0, total=1, attempt=0)  # left from a text block
    assert not w.text_mode
    assert _render(w) == ref  # lanes, no text view, no highlight
    w.begin_run(time.perf_counter())
    assert w.onset_paint_ns.tolist() == [-1, -1] and w.offset_paint_ns.tolist() == [-1, -1]
    _render(w)
    assert w.go_paint_ns == {}  # nor are Go frames logged
    w.sequence = None  # no sequence: the text view is drawn
    assert w.text_mode
