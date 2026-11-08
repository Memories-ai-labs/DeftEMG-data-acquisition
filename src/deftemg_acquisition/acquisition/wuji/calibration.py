"""Wuji Glove hand model calibration, run before recording (``WujiGloveRecorder(calibrate=True)``).

Adapted from the SDK's ``examples/python/wuji_glove/5.calibration.py``. The SDK keeps one
hand model per SDK user and hand side (``left_hand.urdf`` / ``right_hand.urdf``); a glove of
that side connected under that user loads it. The default SDK user cannot calibrate, so
``use_user`` creates a new user, named with the time, for each calibration.
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from deftemg_acquisition.tasks.prompter import BACKGROUND, FOREGROUND

from .hand_pose import HandPoseWidget

log = logging.getLogger(__name__)

POSE_GUIDES = {
    "flat_open": "Straighten all fingers, palm flat, wrist still.",
    "pinch_index": "Touch thumb tip to index fingertip.",
    "pinch_middle": "Touch thumb tip to middle fingertip.",
    "pinch_ring": "Touch thumb tip to ring fingertip.",
    "pinch_pinky": "Touch thumb tip to pinky fingertip.",
    "four_finger_bend_90": "Bend four fingers about 90 degrees.",
    "thumb_to_index_dip": "Press thumb near the index DIP joint base.",
    "index_to_thumb_emf": "Point index fingertip toward thumb sensor side.",
}


def use_user(manager: Any, name: str | None) -> dict[str, Any]:
    """Create and switch to a new SDK user ``<name>_<YYYYmmdd-HHMMSS>`` (``name`` default
    ``deftemg``): every recording is calibrated afresh, so each calibration gets its own user."""
    display_name = f"{name or 'deftemg'}_{datetime.now():%Y%m%d-%H%M%S}"
    user = manager.create_user(display_name, description="DeftEMG calibration before recording")
    log.info("wuji: created SDK user %r", display_name)
    return manager.switch_user(user["user_id"])


def progress_value(fb: dict[str, Any]) -> float:
    """The current pose's progress, 0..1."""
    try:
        if "progress" in fb:
            return max(0.0, min(1.0, float(fb["progress"])))
        if fb.get("state") == "done":
            return 1.0
        elapsed, target = {
            "collecting": ("collect_elapsed", "collect_target"),
            "waiting_stable": ("hold_elapsed", "hold_target"),
        }.get(fb.get("state"), ("", ""))
        t = float(fb.get(target, 0.0) or 0.0)
        return max(0.0, min(1.0, float(fb.get(elapsed, 0.0) or 0.0) / t)) if t > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


STATE_TEXT = {
    "waiting_movement": "Open your palm, then move into the pose",
    "waiting_stable": "Hold still…",
    "collecting": "Collecting: keep holding",
}


class CalibrationWindow(QtWidgets.QWidget):
    """The calibration of one glove in a window like the prompter's: the pose to make, its
    progress and the SDK's corrections. The SDK runs on a worker thread; Cancel / Esc / closing
    the window cancel it. ``done`` carries the result, or the exception."""

    feedback = QtCore.Signal(object)
    done = QtCore.Signal(object)

    def __init__(self, glove: Any, side: str, skip_constraints: bool, timeout_s: float):
        super().__init__()
        self.glove, self.side = glove, side
        self.skip_constraints, self.timeout_s = skip_constraints, timeout_s
        self.result: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._cancel = False

        self.setWindowTitle(f"Wuji Glove calibration: {side} hand")
        self.setStyleSheet(
            f"QWidget {{ background: {BACKGROUND}; color: {FOREGROUND}; }}"
            "QProgressBar { border: 1px solid #444; border-radius: 4px; height: 18px;"
            " text-align: center; } QProgressBar::chunk { background: #009E73; }"
            "QPushButton { border: 1px solid #666; border-radius: 4px; padding: 6px 18px; }"
        )
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        lay.setSpacing(14)
        self.header = self._label(f"{side.capitalize()} hand ({glove.serial_number})", 13,
                                  "#9A9A9A")  # fmt: skip
        self.pose = self._label("Starting…", 30, bold=True)
        self.figure = HandPoseWidget(side)
        self.guide = self._label("", 12, "#9A9A9A")
        self.state = self._label("", 16, "#E69F00")
        self.current = QtWidgets.QProgressBar(maximum=1000, textVisible=False)
        self.overall = QtWidgets.QProgressBar(format="pose %v of %m")
        self.hints = self._label("", 13, "#D55E00")
        self.hints.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        lay.addWidget(self.header)
        lay.addWidget(self.pose)
        lay.addWidget(self.figure, 3)
        for w in (self.guide, self.state, self.current, self.overall):
            lay.addWidget(w)
        lay.addWidget(self.hints, 1)
        cancel = QtWidgets.QPushButton("Cancel (Esc)")
        cancel.clicked.connect(self.close)
        lay.addWidget(cancel, 0, QtCore.Qt.AlignRight)
        self.resize(900, 860)

        self.feedback.connect(self._show)
        self.done.connect(self._finished)
        threading.Thread(target=self._run, daemon=True, name="wuji-calibrate").start()

    def _label(self, text: str, size: float, color: str | None = None,
               bold: bool = False) -> QtWidgets.QLabel:  # fmt: skip
        label = QtWidgets.QLabel(text, alignment=QtCore.Qt.AlignCenter, wordWrap=True)
        font = label.font()
        font.setPointSizeF(size)
        font.setBold(bold)
        label.setFont(font)
        if color:
            label.setStyleSheet(f"color: {color};")
        return label

    def _run(self) -> None:
        async def run() -> Any:  # the SDK's calibrate needs a running event loop
            self._loop, self._task = asyncio.get_running_loop(), asyncio.current_task()
            if self._cancel:  # cancelled before this started
                raise asyncio.CancelledError
            return await self.glove.calibrate(
                skip_constraints=self.skip_constraints, timeout_s=self.timeout_s,
                on_feedback=self.feedback.emit,
            )  # fmt: skip

        try:
            self.done.emit(dict(asyncio.run(run())))
        except BaseException as e:  # handed to the main thread, which raises it
            self.done.emit(e)

    def _show(self, fb: dict[str, Any]) -> None:
        step = int(fb.get("step_index", 0) or 0)
        total = int(fb.get("step_total", 0) or 0)
        name, state = str(fb.get("step_name", "")), str(fb.get("state", ""))
        self.pose.setText(f"{step + 1}/{total} · {name.replace('_', ' ').capitalize()}")
        self.figure.set_pose(name)
        self.guide.setText(POSE_GUIDES.get(name, "Move into the pose and hold still."))
        if state == "done":
            last = step + 1 >= total
            self.state.setText("Finishing calibration…" if last else "Collected. Open your palm")
        else:
            self.state.setText(STATE_TEXT.get(state, state))
        self.current.setValue(int(progress_value(fb) * 1000))
        self.overall.setMaximum(max(total, 1))
        self.overall.setValue(step + 1 if state == "done" else step)
        self.hints.setText("\n".join(diagnostic_messages(fb)[:8]))

    def _finished(self, result: Any) -> None:
        self.result = result
        self.close()

    def keyPressEvent(self, ev) -> None:
        if ev.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev) -> None:
        if self.result is None:  # still running: cancel, close once the worker reports back
            self._cancel = True
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)
            self.state.setText("Cancelling…")
            ev.ignore()
            return
        super().closeEvent(ev)


def diagnostic_messages(fb: dict[str, Any]) -> list[str]:
    messages = []
    if fb.get("variance_ok") is False:
        messages.append("motion: hand is moving too much; hold wrist and fingers still")

    metrics = fb.get("metrics") or []
    grouped: dict[str, list[str]] = {}
    for metric in metrics if isinstance(metrics, list) else []:
        if not isinstance(metric, dict):
            continue
        label = str(metric.get("label", "metric"))
        unit = str(metric.get("unit", ""))
        scale = 1000.0 if unit == "m" else 1.0
        unit = "mm" if unit == "m" else unit
        error = float(metric.get("error", 0.0) or 0.0) * scale
        if error <= 0.0001:
            continue
        finger = metric.get("finger") or "global"
        finger_b = metric.get("finger_b")
        group = f"{finger}-{finger_b}" if finger_b else str(finger)
        hint = metric.get("hint", "adjust pose")
        grouped.setdefault(group, []).append(f"{label}: off by {error:.1f}{unit}; {hint}")

    for group in sorted(grouped):
        messages.append(f"{group}:")
        messages.extend(f"  {message}" for message in grouped[group])
    if fb.get("constraints_ok") is False and not grouped:
        messages.append("pose: not inside target; adjust the current pose")
    return messages


def calibrate(glove: Any, side: str, *, skip_constraints: bool = False,
              timeout_s: float = 900.0) -> dict[str, Any]:  # fmt: skip
    """Run the SDK's hand model calibration on one connected glove in a ``CalibrationWindow``
    (blocks until it is done or cancelled; raises if it failed or was cancelled)."""
    pg.mkQApp("Wuji Glove calibration")
    win = CalibrationWindow(glove, side, skip_constraints, timeout_s)
    loop = QtCore.QEventLoop()
    win.done.connect(loop.quit, QtCore.Qt.QueuedConnection)  # after the window took the result
    win.show()
    win.raise_()
    win.activateWindow()
    loop.exec()
    if isinstance(win.result, asyncio.CancelledError):
        raise RuntimeError(f"wuji {side}: calibration cancelled")
    if isinstance(win.result, BaseException):
        raise RuntimeError(f"wuji {side}: calibration failed: {win.result}") from win.result
    log.info("wuji %s calibrated: %s", side, win.result.get("calibrated_urdf"))
    return win.result
