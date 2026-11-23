"""Drawings of the Wuji Glove calibration poses, in the prompter's hand style (``tasks.prompter``).

The prompter's right hand seen from above, in cm, with the fingers posed: a flexed finger
looks shorter from above (``curled_tip``), and the thumb, a straight digit from its base, is
pointed at whatever it touches. The digits a pose uses are filled in their prompter colours,
and a ring marks where two digits touch. The left hand is the mirror image.
"""

from __future__ import annotations

import math

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from deftemg_acquisition.tasks.prompter import (
    DIGIT_COLORS,
    FINGERS_CM,
    FOREGROUND,
    PALM_CM,
    _finger,
    _qcolor,
)

FINGERS = ("index", "middle", "ring", "pinky")
SPREAD_CM = 2.9  # open hand: fingertips this far apart, as in the prompter's lanes
OPEN_TIP_X = {"thumb": -2 * SPREAD_CM, "index": -SPREAD_CM, "middle": 0.0,
              "ring": SPREAD_CM, "pinky": 2 * SPREAD_CM}  # fmt: skip
SHORTEN = 0.6  # a finger bent 90 degrees at the knuckle shows this much less of its length
WRIST_Y = 18.5  # cm: the drawing stops here


def curled_tip(digit: str, curl: float) -> tuple[float, float]:
    """Tip (cm) of a digit flexed by ``curl`` (0 straight, 1 bent 90 degrees), seen from above."""
    kx, ky, length, *_ = FINGERS_CM[digit]
    dx = OPEN_TIP_X[digit] - kx
    dy = -math.sqrt(max(length**2 - dx**2, 0.0))
    seen = 1 - SHORTEN * curl
    return kx + dx * seen, ky + dy * seen


def _along(digit: str, curl: float, frac: float) -> tuple[float, float]:
    """The point ``frac`` of the way from a digit's knuckle to its (curled) tip."""
    kx, ky = FINGERS_CM[digit][:2]
    tx, ty = curled_tip(digit, curl)
    return kx + frac * (tx - kx), ky + frac * (ty - ky)


def pose_tips(pose: str) -> tuple[dict[str, tuple[float, float]], set[str], tuple | None]:
    """A calibration pose: each digit's tip (cm), the digits it uses, and where two touch."""
    curls = dict.fromkeys(FINGERS, 0.1)  # relaxed
    thumb = curled_tip("thumb", 0.0)
    active: set[str] = set()
    contact = None
    if pose == "flat_open":
        curls = dict.fromkeys(FINGERS, 0.0)
        active = {"thumb", *FINGERS}
    elif pose.startswith("pinch_") and pose[6:] in FINGERS:
        finger = pose[6:]
        curls[finger] = 0.7
        fx, fy = curled_tip(finger, 0.7)
        thumb = contact = (fx - 0.4, fy + 0.3)
        active = {"thumb", finger}
    elif pose == "four_finger_bend_90":
        curls = dict.fromkeys(FINGERS, 1.0)
        active = set(FINGERS)
    elif pose == "thumb_to_index_dip":
        curls["index"] = 0.3
        x, y = _along("index", 0.3, 0.65)
        thumb = contact = (x - 0.5, y)
        active = {"thumb", "index"}
    elif pose == "index_to_thumb_emf":
        thumb = (-4.8, 7.0)
        tips = {d: curled_tip(d, c) for d, c in curls.items()}
        tips["index"] = (-3.9, 6.5)  # bent over, pointing at the thumb's side
        return {**tips, "thumb": thumb}, {"thumb", "index"}, thumb
    tips = {d: curled_tip(d, c) for d, c in curls.items()}
    return {**tips, "thumb": thumb}, active, contact


class HandPoseWidget(QtWidgets.QWidget):
    """Draws ``pose`` (a calibration step name; ``set_pose``) for the ``side`` hand."""

    def __init__(self, side: str = "right"):
        super().__init__()
        self.side = side
        self.pose: str | None = None
        self.setMinimumHeight(240)

    def set_pose(self, pose: str) -> None:
        if pose != self.pose:
            self.pose = pose
            self.update()

    def paintEvent(self, _event) -> None:
        if not self.pose:
            return
        tips, active, contact = pose_tips(self.pose)
        palm = QtGui.QPainterPath()
        palm.moveTo(*PALM_CM[0])
        for c1, c2, end in PALM_CM[1:]:
            palm.cubicTo(*c1, *c2, *end)
        palm.closeSubpath()
        fingers = {}
        for d, (tx, ty) in tips.items():
            kx, ky, _, wb, wt = FINGERS_CM[d]
            fingers[d] = _finger(kx, ky, tx, math.hypot(tx - kx, ty - ky), wb, wt)

        rect = palm.boundingRect()
        for finger, _ in fingers.values():
            rect = rect.united(finger.boundingRect())
        rect.setBottom(WRIST_Y)
        s = 0.92 * min(self.width() / rect.width(), self.height() / rect.height())
        tr = QtGui.QTransform()
        tr.translate(self.width() / 2, self.height() / 2)
        tr.scale(s if self.side == "right" else -s, s)
        tr.translate(-rect.center().x(), -rect.center().y())

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setClipRect(tr.mapRect(rect))
        # Union in pixels: Qt flattens curves when combining paths (as in the prompter).
        silhouette = tr.map(palm)
        for finger, _ in fingers.values():
            silhouette = silhouette.united(tr.map(finger))
        p.setBrush(_qcolor(FOREGROUND, 18))
        p.setPen(QtGui.QPen(_qcolor(FOREGROUND, 200), 2))
        p.drawPath(silhouette.simplified())
        for d, (finger, nail) in fingers.items():
            if d in active:
                p.setBrush(_qcolor(DIGIT_COLORS[d], 120))
                p.setPen(QtGui.QPen(_qcolor(DIGIT_COLORS[d]), 2))
                p.drawPath(tr.map(finger))
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(_qcolor(FOREGROUND, 90), 1.2))
            p.drawPath(tr.map(nail))
        if contact is not None:
            p.setPen(QtGui.QPen(_qcolor("#FFFFFF"), 2.5))
            p.drawEllipse(tr.map(QtCore.QPointF(*contact)), 0.55 * s, 0.55 * s)
        p.end()
