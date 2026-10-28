"""Scrolling finger-press prompter: lanes of capsules falling onto a target row above a hand.

Each prompt is drawn as a capsule in its digit's lane. Circle size gives the force level and
capsule length gives the hold time. The leading edge reaches the target row (two rings per
lane, one per force level) at the prompt's onset; while the press is held the capsule shrinks
into the target and vanishes at the offset. A digit the task implies
(``Sequence.implied_digits``: the thumb in pinch and squeeze) gets neither capsules nor target
rings: its lane stays empty.

The hand is drawn from an egocentric (dorsal) view, so for the right hand the thumb is on the
left and the pinky on the right; the left hand is mirrored.

With ``show_effort``, one effort bar runs across the lanes under the target row (the row moves
up to make room): its fill is ``effort`` (0 = rest, 1 = the calibration maximum of what is
prompted; ``tasks.effort``), with a zone per force level (``effort_zones``). During a press the
prompted level's zone is outlined and the fill is green inside it, vermillion above it, grey
below it.

Other code can draw in the lanes too (``overlays``: :class:`PrompterOverlay`, e.g. a model's
output on the prompts' timeline) and be told when each frame has been painted (``on_paint``).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from deftemg_acquisition.clock import host_ns

from .sequence import DIGITS, LEVELS, Sequence

log = logging.getLogger(__name__)

# Okabe-Ito colours: distinct for common colour-vision deficiencies.
DIGIT_COLORS = {
    "thumb": "#E69F00",
    "index": "#56B4E9",
    "middle": "#009E73",
    "ring": "#F0E442",
    "pinky": "#CC79A7",
}
BACKGROUND = "#111418"
FOREGROUND = "#E8E8E8"
# Fraction of lane width. "medium" only appears when replaying recordings made before the
# switch to two levels; "max" in calibration blocks (as hard as is comfortable).
LEVEL_RADIUS = {"low": 0.18, "medium": 0.28, "high": 0.38, "max": 0.47}
METER_H = 40  # room under the target row for the effort bar (bar, its labels, gaps)
METER_COLOURS = {"below": "#8C8C8C", "inside": "#009E73", "above": "#D55E00"}
SYNC_PATCH_PX = 56  # sync patch: square in the top-right corner for a photodiode or camera
SYNC_FLASH_S = 0.1  # how long it stays white after each prompt onset

# --- Hand model ------------------------------------------------------------------------------
# A right hand seen from above (dorsal view), in cm, x to the right, y down from the top of the
# middle fingertip, whose knuckle is at x = 0 so the middle finger points straight up its lane.
# Proportions follow typical adult anthropometry; the other fingers splay so that their tips
# sit LANE_CM apart, one under each lane. The left hand is the mirror image.
LANE_CM = 2.9  # fingertip spacing = lane width
HAND_VISIBLE_CM = 10.5  # how much of the hand is on screen (cropped just below the knuckles)
HAND_HEIGHT_FRACTION = 0.27  # share of the window height given to the hand
DEFAULT_LOOKAHEAD_S = 6.0  # seconds of upcoming prompts visible above the target row
# digit: (knuckle x, knuckle y, length knuckle-to-tip, base width, tip width)
FINGERS_CM = {
    "index": (-1.8, 8.6, 7.4, 2.0, 1.7),
    "middle": (0.0, 8.2, 8.2, 2.0, 1.7),
    "ring": (1.8, 8.6, 7.7, 1.9, 1.6),
    "pinky": (3.5, 9.4, 6.2, 1.65, 1.4),
    "thumb": (-1.7, 15.5, 10.0, 2.3, 1.9),  # measured from the base of the thumb (CMC joint)
}
# Palm outline: start point, then cubic segments as (control 1, control 2, end).
PALM_CM = [
    (-2.7, 8.2),
    ((-3.0, 9.8), (-3.3, 11.0), (-3.2, 12.5)),  # thumb side, over the thenar bulge
    ((-3.1, 14.5), (-2.7, 16.5), (-2.4, 18.0)),
    ((-2.4, 20.0), (-2.4, 22.0), (-2.4, 24.0)),  # wrist, off-screen
    ((0.9, 24.0), (0.9, 24.0), (4.1, 24.0)),
    ((4.1, 22.0), (4.1, 20.0), (4.1, 18.0)),
    ((4.3, 16.0), (5.1, 13.5), (5.0, 11.5)),  # little-finger side
    ((4.9, 10.3), (4.5, 9.6), (4.2, 9.0)),
    ((2.9, 7.9), (-0.1, 7.4), (-2.7, 8.2)),  # across the knuckles
]


@dataclass(frozen=True)
class Layout:
    """Screen geometry for a widget of size ``width`` x ``height`` (pixels, y down)."""

    width: float
    height: float
    hand: str = "right"
    lookahead_s: float = DEFAULT_LOOKAHEAD_S
    header_h: float = 48.0
    meter_h: float = 0.0  # room for the effort bar under the target row (METER_H when shown)

    @property
    def cm(self) -> float:
        """Pixels per cm of hand: sized by HAND_HEIGHT_FRACTION, and fitting the width."""
        by_height = self.height * HAND_HEIGHT_FRACTION / HAND_VISIBLE_CM
        by_width = self.width / ((len(DIGITS) + 0.6) * LANE_CM)
        return min(by_height, by_width)

    @property
    def lane_w(self) -> float:
        return LANE_CM * self.cm

    @property
    def lanes_x0(self) -> float:
        return (self.width - self.lane_w * len(DIGITS)) / 2

    def lane_index(self, digit: str) -> int:
        """Screen lane for a digit: thumb leftmost for the right hand, rightmost for the left."""
        i = DIGITS.index(digit)
        return i if self.hand == "right" else len(DIGITS) - 1 - i

    def lane_x(self, digit: str) -> float:
        return self.lanes_x0 + (self.lane_index(digit) + 0.5) * self.lane_w

    def radius(self, level: str) -> float:
        return LEVEL_RADIUS[level] * self.lane_w

    @property
    def hand_top(self) -> float:
        """Screen y of the top of the hand model (the middle fingertip)."""
        return self.height - HAND_VISIBLE_CM * self.cm

    @property
    def target_y(self) -> float:
        return self.hand_top - self.radius("high") - 14 - self.meter_h

    @property
    def speed(self) -> float:
        """Scroll speed in pixels per second."""
        return (self.target_y - self.header_h) / self.lookahead_s

    def capsule(self, onset_in_s: float, offset_in_s: float) -> tuple[float, float] | None:
        """Vertical extent (top, bottom centre y) of a prompt given time until onset/offset.

        Returns None when the prompt is not on screen.
        """
        lead = self.target_y - onset_in_s * self.speed
        trail = self.target_y - offset_in_s * self.speed
        if trail >= self.target_y or lead < self.header_h - self.radius("high"):
            return None
        return trail, min(lead, self.target_y)


def _qcolor(hex_color: str, alpha: int = 255) -> QtGui.QColor:
    c = QtGui.QColor(hex_color)
    c.setAlpha(alpha)
    return c


def _finger(
    knuckle_x: float, knuckle_y: float, tip_x: float, length: float, w_base: float, w_tip: float
) -> tuple[QtGui.QPainterPath, QtGui.QPainterPath]:
    """A tapered finger with a rounded tip, and its nail, in hand cm.

    The finger keeps its length and is angled from the knuckle so its tip lands at ``tip_x``.
    """
    dx = tip_x - knuckle_x
    rise = math.sqrt(max(length**2 - dx**2, 0.0))
    finger = QtGui.QPainterPath()
    finger.moveTo(-w_base / 2, 1.0)  # starts inside the palm so the union is seamless
    finger.lineTo(-w_tip / 2, -length + w_tip / 2)
    finger.arcTo(QtCore.QRectF(-w_tip / 2, -length, w_tip, w_tip), 180, -180)
    finger.lineTo(w_base / 2, 1.0)
    finger.closeSubpath()
    nail_w = w_tip * 0.62
    nail = QtGui.QPainterPath()
    nail.addRoundedRect(
        QtCore.QRectF(-nail_w / 2, -length + 0.25, nail_w, 1.25), nail_w * 0.45, nail_w * 0.45
    )
    tr = QtGui.QTransform()
    tr.translate(knuckle_x, knuckle_y)
    tr.rotate(math.degrees(math.atan2(dx, rise)))
    return tr.map(finger), tr.map(nail)


def hand_paths(lay: Layout) -> tuple[QtGui.QPainterPath, QtGui.QPainterPath]:
    """Hand silhouette and nail outlines in screen pixels, fingertips under their lanes."""
    palm = QtGui.QPainterPath()
    palm.moveTo(*PALM_CM[0])
    for c1, c2, end in PALM_CM[1:]:
        palm.cubicTo(*c1, *c2, *end)
    palm.closeSubpath()

    mirror = 1 if lay.hand == "right" else -1
    to_screen = QtGui.QTransform()
    to_screen.translate(lay.width / 2, lay.hand_top)
    to_screen.scale(mirror * lay.cm, lay.cm)
    # Union in screen pixels: Qt flattens curves when combining paths, and at cm scale that
    # would turn the rounded fingertips into a few straight segments.
    silhouette, nails = to_screen.map(palm), QtGui.QPainterPath()
    for d, (kx, ky, length, wb, wt) in FINGERS_CM.items():
        tip_x = mirror * (lay.lane_x(d) - lay.width / 2) / lay.cm
        finger, nail = _finger(kx, ky, tip_x, length, wb, wt)
        silhouette = silhouette.united(to_screen.map(finger))
        nails.addPath(to_screen.map(nail))
    return silhouette.simplified(), nails


class PrompterOverlay:
    """Something more to draw in the prompter (``PrompterWidget.overlays``). ``paint_under``
    runs after the lane lines, before the capsules; ``paint_over`` after the target rings, before
    the effort bar and the hand. Each gets the painter (saved and restored around the call), the
    layout and the widget (``widget.t``: the block's time, None between blocks). One that raises
    is logged and removed."""

    def paint_under(self, p: QtGui.QPainter, lay: Layout, widget: PrompterWidget) -> None:
        pass

    def paint_over(self, p: QtGui.QPainter, lay: Layout, widget: PrompterWidget) -> None:
        pass


class PrompterWidget(QtWidgets.QWidget):
    """Draws a :class:`Sequence` at time ``t`` (seconds since block start, set by the owner)."""

    def __init__(
        self,
        hand: str = "right",
        lookahead_s: float = DEFAULT_LOOKAHEAD_S,
        sync_patch: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.sync_patch = sync_patch
        self.flash_paint_ns: list[int] = []  # host time each sync flash was first painted
        self._flash_on = False
        self.hand = hand
        self.lookahead_s = lookahead_s
        self.sequence: Sequence | None = None
        self.t: float | None = None  # None: not running, draw an idle screen
        # While running, t is taken from this perf_counter origin at the moment of painting,
        # so each frame shows where prompts are when it is drawn, not when it was requested.
        self.t0_perf: float | None = None
        # Host time (ns) of the first frame drawn at/after each prompt's onset and offset,
        # i.e. when the capsule was first drawn reaching the target and first drawn gone.
        self.onset_paint_ns = np.empty(0, np.int64)
        self.offset_paint_ns = np.empty(0, np.int64)
        self._onsets_s = np.empty(0)
        self._offsets_s = np.empty(0)
        self._hand_cache: tuple[tuple, tuple[QtGui.QPainterPath, QtGui.QPainterPath]] | None = None
        self.header = ""
        self.message = ""  # large centred text, e.g. instructions between blocks
        self.show_effort = False  # the effort bar (and room for it under the target row)
        # (effort, prompted level) while running: effort None = no reading yet (empty bar),
        # level None = between presses (no zone outlined); None: no bar this block
        self.effort: tuple[float | None, str | None] | None = None
        self.effort_zones: dict[str, tuple[float, float]] = {}
        self.overlays: list[PrompterOverlay] = []  # drawn in the lanes, in order
        self.on_paint = None  # called after each paint (e.g. to time what reached the screen)
        self.setMinimumSize(400, 600)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BACKGROUND))
        self.setPalette(pal)

    def begin_run(self, t0_perf: float) -> None:
        """Start animating ``self.sequence`` from ``t0_perf`` (a time.perf_counter value)."""
        n = len(self.sequence.prompts)
        self._onsets_s = np.array([p.onset_s for p in self.sequence.prompts])
        self._offsets_s = np.array([p.offset_s for p in self.sequence.prompts])
        self.onset_paint_ns = np.full(n, -1, np.int64)
        self.offset_paint_ns = np.full(n, -1, np.int64)
        self.flash_paint_ns = []
        self._flash_on = False
        self.t0_perf = t0_perf
        self.update()

    def end_run(self) -> None:
        self.t0_perf = None
        self.t = None
        self.update()

    def layout_for_size(self) -> Layout:
        meter_h = METER_H if self.show_effort else 0.0
        return Layout(self.width(), self.height(), self.hand, self.lookahead_s, meter_h=meter_h)

    def _hand(self, lay: Layout) -> tuple[QtGui.QPainterPath, QtGui.QPainterPath]:
        """Hand paths, rebuilt only when the size or hand changes (path unions are slow)."""
        key = (lay.width, lay.height, lay.hand)
        if self._hand_cache is None or self._hand_cache[0] != key:
            self._hand_cache = (key, hand_paths(lay))
        return self._hand_cache[1]

    def paintEvent(self, _event) -> None:
        if self.t0_perf is not None:
            self.t = time.perf_counter() - self.t0_perf
            now_ns = host_ns()
            for times_s, painted in (
                (self._onsets_s, self.onset_paint_ns),
                (self._offsets_s, self.offset_paint_ns),
            ):
                painted[(times_s <= self.t) & (painted < 0)] = now_ns
        lay = self.layout_for_size()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # Lanes
        p.setPen(QtGui.QPen(_qcolor(FOREGROUND, 25), 1))
        for i in range(len(DIGITS) + 1):
            x = lay.lanes_x0 + i * lay.lane_w
            p.drawLine(QtCore.QPointF(x, lay.header_h), QtCore.QPointF(x, lay.target_y))

        implied = set(self.sequence.implied_digits) if self.sequence is not None else set()
        self._paint_overlays("paint_under", p, lay)

        # Prompts
        if self.sequence is not None and self.t is not None:
            p.save()
            p.setClipRect(QtCore.QRectF(0, lay.header_h, self.width(), self.height()))
            p.setPen(QtCore.Qt.NoPen)
            for pr in self.sequence.prompts:
                ext = lay.capsule(pr.onset_s - self.t, pr.offset_s - self.t)
                if ext is None:
                    continue
                top, bottom = ext
                for d, lv in zip(pr.digits, pr.levels, strict=True):
                    if d in implied:
                        continue
                    x, r = lay.lane_x(d), lay.radius(lv)
                    p.setBrush(_qcolor(DIGIT_COLORS[d], 235))
                    p.drawRoundedRect(
                        QtCore.QRectF(x - r, top - r, 2 * r, bottom - top + 2 * r), r, r
                    )
            p.restore()

        # Target row: one ring per force level per lane; drawn over the
        # prompts so the size reference stays visible during a press.
        p.setBrush(QtCore.Qt.NoBrush)
        levels = self._ring_levels()
        for d in (d for d in DIGITS if d not in implied):
            p.setPen(QtGui.QPen(_qcolor(DIGIT_COLORS[d], 130), 1.5))
            for lv in levels:
                r = lay.radius(lv)
                p.drawEllipse(QtCore.QPointF(lay.lane_x(d), lay.target_y), r, r)

        self._paint_overlays("paint_over", p, lay)
        if self.show_effort and self.effort is not None and self.t is not None:
            self._paint_effort(p, lay)

        # Hand
        silhouette, nails = self._hand(lay)
        p.setBrush(_qcolor(FOREGROUND, 18))
        p.setPen(QtGui.QPen(_qcolor(FOREGROUND, 200), 2))
        p.drawPath(silhouette)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(_qcolor(FOREGROUND, 90), 1.2))
        p.drawPath(nails)

        if self.sync_patch:
            self._paint_sync_patch(p)

        # Text
        p.setPen(_qcolor(FOREGROUND))
        font = p.font()
        font.setPointSizeF(13)
        p.setFont(font)
        p.drawText(QtCore.QRectF(16, 0, self.width() - 32, lay.header_h),
                   QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, self.header)  # fmt: skip
        if self.message:
            area = QtCore.QRectF(56, lay.header_h, self.width() - 112, lay.target_y - lay.header_h)
            flags = QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap
            for size in range(20, 9, -1):  # smaller text until it fits a small window
                font.setPointSizeF(size)
                p.setFont(font)
                text = p.boundingRect(area, flags, self.message)  # the backdrop fits the text
                if text.height() <= area.height() and text.width() <= area.width():
                    break
            p.fillRect(text.adjusted(-16, -12, 16, 12), _qcolor(BACKGROUND, 220))
            p.drawText(area, flags, self.message)
        p.end()
        if self.on_paint is not None:
            try:
                self.on_paint()
            except Exception:  # instrumentation must never break the display
                log.exception("prompter on_paint failed; it is turned off")
                self.on_paint = None

    def _paint_overlays(self, step: str, p: QtGui.QPainter, lay: Layout) -> None:
        for overlay in list(self.overlays):
            p.save()
            try:
                getattr(overlay, step)(p, lay, self)
            except Exception:  # an optional display must never stop a recording
                log.exception("prompter overlay %r failed; it is removed", overlay)
                self.overlays.remove(overlay)
            finally:
                p.restore()

    def effort_bar(self, lay: Layout) -> QtCore.QRectF:
        """Where the effort bar's track is (its fill runs from the left edge, 0 to 1)."""
        x0 = lay.lanes_x0 + 6
        return QtCore.QRectF(x0, lay.target_y + lay.radius("high") + 10, 5 * lay.lane_w - 12, 16)

    def effort_state(self) -> str | None:
        """ "below", "inside" or "above" the prompted level's zone; None between presses."""
        value, level = self.effort or (None, None)
        if value is None or level not in self.effort_zones:
            return None
        lo, hi = self.effort_zones[level]
        return "inside" if lo <= value <= hi else "above" if value > hi else "below"

    def _paint_effort(self, p: QtGui.QPainter, lay: Layout) -> None:
        value, level = self.effort
        bar = self.effort_bar(lay)
        p.save()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(_qcolor(FOREGROUND, 22))  # track
        p.drawRoundedRect(bar.adjusted(-4, -4, 4, 4), 6, 6)
        x0, w = bar.left(), bar.width()
        for lv, (lo, hi) in self.effort_zones.items():
            p.setBrush(_qcolor(FOREGROUND, 70 if lv == level else 40))
            p.drawRect(QtCore.QRectF(x0 + lo * w, bar.top(), (hi - lo) * w, bar.height()))
        if value is not None:
            state = self.effort_state() or "below"
            p.setBrush(_qcolor(METER_COLOURS[state], 235))
            fill = QtCore.QRectF(x0, bar.top() + 4, min(max(value, 0.0), 1.0) * w, bar.height() - 8)
            p.drawRoundedRect(fill, 3, 3)
        if level in self.effort_zones:  # the prompted zone, outlined
            lo, hi = self.effort_zones[level]
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(_qcolor(FOREGROUND), 2))
            p.drawRect(QtCore.QRectF(x0 + lo * w, bar.top() - 1, (hi - lo) * w, bar.height() + 2))
        font = p.font()
        font.setPointSizeF(9)
        p.setFont(font)
        p.setPen(_qcolor(FOREGROUND, 200))
        for lv, (lo, hi) in self.effort_zones.items():
            p.drawText(QtCore.QRectF(x0 + lo * w, bar.bottom() + 3, (hi - lo) * w, 14),
                       QtCore.Qt.AlignCenter, lv)  # fmt: skip
        p.restore()

    def _ring_levels(self) -> list[str]:
        """Target rings: one per force level the sequence uses (never raises in paint)."""
        if self.sequence is None:
            return list(LEVELS)
        try:
            return list(self.sequence.level_names)  # the sequence's level coding
        except ValueError:  # levels of several codings: the ones it uses
            used = {lv for pr in self.sequence.prompts for lv in pr.levels}
            return [lv for lv in LEVEL_RADIUS if lv in used] or list(LEVELS)

    def _paint_sync_patch(self, p: QtGui.QPainter) -> None:
        """White for SYNC_FLASH_S from each prompt onset, otherwise black.

        The capsule reaches the target in the same frame, so a photodiode (or slow-motion video)
        on the patch measures when prompts actually appear. The paint time of each flash is
        logged for comparison with the scheduled onset.
        """
        on = False
        if self.sequence is not None and self.t is not None:
            on = any(
                pr.onset_s <= self.t < pr.onset_s + SYNC_FLASH_S for pr in self.sequence.prompts
            )
        if on and not self._flash_on:
            self.flash_paint_ns.append(host_ns())
        self._flash_on = on
        rect = QtCore.QRectF(self.width() - SYNC_PATCH_PX, 0, SYNC_PATCH_PX, SYNC_PATCH_PX)
        p.fillRect(rect, QtGui.QColor("white" if on else "black"))
