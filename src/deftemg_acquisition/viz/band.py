"""Wrist cross-section showing EMG RMS at each electrode of the armband.

Geometry: 18 electrode pods sit evenly around the band in 19 positions (360/19 degrees apart);
the band opening between ch18 and ch1 takes the 19th, as wide as a missing pod. The puck sits
between ch9 and ch10, and the opening is directly opposite it, centred on the volar side.
Numbering runs ch10 -> ch18 down one side to the opening, then ch1 -> ch9 up the other side back
to the puck.

Default wearing: right arm, puck on the dorsal (back) surface, cable pointing up the arm. The
cross-section is drawn as seen from the elbow looking toward the hand (the subject's own view,
like the prompter): dorsal up, volar (palm) down, and for the right arm radial (thumb) left and
ulnar (pinky) right. Channel numbers are assumed to increase counterclockwise from this
viewpoint (ch10 on the left of the puck); swap PUCK_NEIGHBOURS if they turn out to run the other
way. On the left arm the band looks the same from this viewpoint; only the radial/ulnar labels
swap.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from deftemg_acquisition.acquisition.protocol import BAND_SLOTS, N_EMG_CHANNELS

PUCK_DEG = 90.0  # puck at the top (dorsal), angles counterclockwise from the right (screen)
POD_STEP_DEG = 360.0 / BAND_SLOTS
PUCK_NEIGHBOURS = (9, 10)  # pods either side of the puck: ch9 clockwise, ch10 counterclockwise
WRIST_ASPECT = 0.72  # wrist depth / width (the cross-section is roughly elliptical)

BACKGROUND = "#111418"
FOREGROUND = "#E8E8E8"
COLORMAP = pg.colormap.get("inferno")


def channel_angles_deg() -> np.ndarray:
    """Angle of each channel (index 0 = ch1) around the band, counterclockwise from the right.

    The pod counterclockwise of the puck (ch10) sits half a pod from it; numbers increase
    counterclockwise and continue from ch18 to ch1 across the band opening (one empty position).
    """
    first = PUCK_DEG + POD_STEP_DEG / 2
    ch = np.arange(1, N_EMG_CHANNELS + 1)
    k = ch - PUCK_NEIGHBOURS[1]  # positions counterclockwise from ch10
    k = np.where(k < 0, k + BAND_SLOTS, k)  # ch1..ch9 come after the opening
    return (first + k * POD_STEP_DEG) % 360


def opening_angle_deg() -> float:
    """Angle of the band opening, between ch18 and ch1."""
    a = channel_angles_deg()
    return float((a[-1] + POD_STEP_DEG * (BAND_SLOTS - N_EMG_CHANNELS + 1) / 2) % 360)


def side_labels(arm: str) -> dict[str, str]:
    """Anatomical labels for the four sides of the cross-section."""
    radial, ulnar = "Radial (thumb)", "Ulnar (pinky)"
    left, right = (radial, ulnar) if arm == "right" else (ulnar, radial)
    return {"top": "Dorsal", "bottom": "Volar (palm)", "left": left, "right": right}


class AutoScale:
    """Colour scale that follows the largest channel RMS: rises at once, decays over ~2 s."""

    def __init__(self, floor_uv: float = 10.0, decay_per_update: float = 0.985):
        self.value = floor_uv
        self.floor_uv = floor_uv
        self.decay = decay_per_update

    def update(self, rms: np.ndarray) -> float:
        target = max(self.floor_uv, 1.1 * float(np.max(rms)) if len(rms) else 0.0)
        self.value = target if target > self.value else max(target, self.value * self.decay)
        return self.value


def band_rms(x: np.ndarray) -> np.ndarray:
    """RMS per channel of (n, 18) EMG in µV, after removing each channel's mean."""
    if len(x) == 0:
        return np.zeros(N_EMG_CHANNELS)
    x = x.astype(np.float64)
    return np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))


class BandRmsWidget(QtWidgets.QWidget):
    """Wrist cross-section with one disc per electrode, filled by its RMS (inferno colormap).

    Each disc's outline uses the channel's trace colour; a translucent shape inside the wrist
    has its radius proportional to RMS, showing where around the wrist the activity is.
    """

    def __init__(self, arm: str = "right", channel_colors=None):
        super().__init__()
        self.arm = arm
        self.channel_colors = channel_colors
        self.rms = np.zeros(N_EMG_CHANNELS)
        self.scale_uv = 50.0  # RMS mapped to the top of the colormap
        self.window_ms = 200
        self.setMinimumSize(300, 260)
        self._angles = np.radians(channel_angles_deg())

    def set_rms(self, rms: np.ndarray, scale_uv: float, window_ms: float) -> None:
        self.rms, self.scale_uv, self.window_ms = rms, max(scale_uv, 1e-6), window_ms
        self.update()

    def _point(self, cx, cy, rx, ry, theta, k=1.0) -> QtCore.QPointF:
        return QtCore.QPointF(cx + k * rx * math.cos(theta), cy - k * ry * math.sin(theta))

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(BACKGROUND))
        w, h = self.width(), self.height()
        fg = QtGui.QColor(FOREGROUND)
        dim = QtGui.QColor(FOREGROUND)
        dim.setAlpha(150)
        font = p.font()

        # Layout: two title lines, puck above the ring, side labels left/right, legend below.
        title_h, legend_h, side_w = 34, 18, 52
        avail_w = w - 2 * side_w
        avail_h = h - title_h - legend_h
        # ring + pod labels take ~1.45 rx sideways and the puck adds ~0.55 rx on top
        rx = max(30.0, min(avail_w / 2 / 1.45, avail_h / (2 * WRIST_ASPECT * 1.45 + 0.55)))
        ry = rx * WRIST_ASPECT
        pod_r = max(4.5, rx * 0.10)
        cx = w / 2
        cy = title_h + rx * 0.55 + ry * 1.45 + (avail_h - (2 * ry * 1.45 + rx * 0.55)) / 2

        # Wrist outline
        p.setPen(QtGui.QPen(dim, 1.5))
        p.setBrush(QtGui.QColor(255, 255, 255, 12))
        p.drawEllipse(QtCore.QPointF(cx, cy), rx * 0.86, ry * 0.86)

        # RMS shape: radius from the centre proportional to RMS (clipped at the scale)
        level = np.clip(self.rms / self.scale_uv, 0, 1)
        points = [
            self._point(cx, cy, rx, ry, th, 0.12 + 0.72 * lv)
            for th, lv in zip(self._angles, level, strict=True)
        ]
        p.setPen(QtGui.QPen(QtGui.QColor(255, 170, 60, 200), 1.5))
        p.setBrush(QtGui.QColor(255, 140, 40, 70))
        p.drawPolygon(QtGui.QPolygonF(points))

        # Band opening (between ch18 and ch1)
        op = math.radians(opening_angle_deg())
        p.setPen(QtGui.QPen(QtGui.QColor("#e15759"), 2, QtCore.Qt.DashLine))
        p.drawLine(self._point(cx, cy, rx, ry, op, 0.9), self._point(cx, cy, rx, ry, op, 1.16))

        # Electrodes, numbered just outside the ring
        font.setPointSizeF(max(7.0, min(10.0, pod_r * 0.9)))
        p.setFont(font)
        for i, (th, lv) in enumerate(zip(self._angles, level, strict=True)):
            c = self._point(cx, cy, rx, ry, th)
            outline = self.channel_colors[i] if self.channel_colors is not None else fg
            p.setPen(QtGui.QPen(QtGui.QColor(outline), 2))
            p.setBrush(COLORMAP.map(float(lv), mode="qcolor"))
            p.drawEllipse(c, pod_r, pod_r)
            off = pod_r * 2.1
            lab = QtCore.QPointF(c.x() + off * math.cos(th), c.y() - off * math.sin(th))
            p.setPen(fg)
            box = QtCore.QRectF(lab.x() - 11, lab.y() - 7, 22, 14)
            p.drawText(box, QtCore.Qt.AlignCenter, str(i + 1))

        # Puck: above the ring and its numbers, joined to the band between ch9 and ch10
        top = self._point(cx, cy, rx, ry, math.radians(PUCK_DEG))
        puck_w, puck_h = rx * 0.7, max(14.0, pod_r * 2.0)
        puck = QtCore.QRectF(cx - puck_w / 2, top.y() - pod_r * 3.2 - puck_h, puck_w, puck_h)
        p.setPen(QtGui.QPen(fg, 1.5))
        p.drawLine(QtCore.QPointF(cx, puck.bottom()), QtCore.QPointF(cx, top.y() - pod_r * 0.4))
        p.setBrush(QtGui.QColor("#3a3f47"))
        p.drawRoundedRect(puck, 4, 4)
        p.drawText(puck, QtCore.Qt.AlignCenter, "puck")

        # Anatomical sides
        font.setPointSizeF(9)
        p.setFont(font)
        p.setPen(dim)
        sides = side_labels(self.arm)
        two_line = {k: v.replace(" (", "\n(") for k, v in sides.items()}
        p.drawText(
            QtCore.QRectF(2, cy - 16, side_w - 4, 32), QtCore.Qt.AlignCenter, two_line["left"]
        )
        p.drawText(
            QtCore.QRectF(w - side_w + 2, cy - 16, side_w - 4, 32), QtCore.Qt.AlignCenter,
            two_line["right"],
        )  # fmt: skip
        below = cy + ry + pod_r * 3.0
        p.drawText(QtCore.QRectF(0, below, w, 16), QtCore.Qt.AlignHCenter, sides["bottom"])

        # Title and legend
        p.setPen(fg)
        p.drawText(
            QtCore.QRectF(6, 2, w - 12, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"Band RMS · last {self.window_ms:.0f} ms",
        )  # fmt: skip
        p.setPen(dim)
        p.drawText(
            QtCore.QRectF(6, 17, w - 12, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"{self.arm} arm, seen from the elbow, {sides['top'].lower()} up",
        )  # fmt: skip
        p.drawText(
            QtCore.QRectF(6, h - legend_h, w - 12, legend_h),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"colour 0–{self.scale_uv:.0f} µV · red dash: band opening",
        )
        p.end()
