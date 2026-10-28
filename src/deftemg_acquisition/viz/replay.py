"""Replay a recorded trial: prompts, EMG viewer with prompt markers, and (later) camera footage.

    pixi run replay                                   # first trial of the latest session
    pixi run replay data/raw/<session> --trial 2
    pixi run replay data/raw/<session>/trial_001_<label>.npz --speed 0.5

The window shows, all on one timeline:
- the prompter as the subject saw it (prompter trials only), with the effort bar where the
  block recorded one,
- the EMG traces, spectrum and IMU as in the live viewer, with a line at the start of each
  prompted press, coloured by digit, appearing as the press starts,
- a camera panel, empty until camera recording is added (see :class:`VideoSource`).

Other code can replay more with the trial (``panels``: :class:`ReplayPanel`, e.g. a model's
output) and add its own options (``parser`` and ``run``).

Keys: Space play/pause, Left/Right -/+1 s (Shift: 5 s), comma/period previous/next prompt,
Home/End start/end, Esc quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from deftemg_acquisition.data import trials
from deftemg_acquisition.tasks.effort import ZONES
from deftemg_acquisition.tasks.prompter import DEFAULT_LOOKAHEAD_S, DIGIT_COLORS, PrompterWidget
from deftemg_acquisition.tasks.sequence import Prompt, Sequence
from deftemg_acquisition.viz.panels import (
    BandRms,
    DisplayControls,
    ImuPlots,
    SpectrumPlot,
    TracePlot,
    auto_spacing,
)

FRAME_MS = 16
SPECTRUM_EVERY_S = 0.25
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


# --- Data -------------------------------------------------------------------------------------


LEVEL_TAGS = {"low": "L", "medium": "M", "high": "H", "max": "MAX"}


@dataclass
class PromptMark:
    onset_s: float  # when the prompt reached the target (drawn time if logged), trial time
    offset_s: float
    label: str
    color: str


@dataclass
class EffortTrace:
    """The effort bar as the subject saw it (blocks recorded with ``--effort-meter``)."""

    t: np.ndarray  # trial time of each reading
    value: np.ndarray  # effort, 0 rest to 1 the calibration maximum (NaN: no reading)
    level: list[str | None]  # the prompted level being held, None between presses
    zones: dict[str, tuple[float, float]]

    def at(self, t: float) -> tuple[float | None, str | None] | None:
        """The bar's (value, level) at trial time ``t``: the last reading before it."""
        i = int(np.searchsorted(self.t, t, side="right")) - 1
        if i < 0:
            return None
        v = float(self.value[i])
        return (None if np.isnan(v) else v), self.level[i]


@dataclass
class ReplayData:
    """One trial on a single timeline: seconds since its first EMG sample."""

    name: str
    npz_path: Path
    t0_ns: int  # host time of the first EMG sample
    emg_t: np.ndarray
    emg_uv: np.ndarray
    imu_t: np.ndarray
    gyro: np.ndarray
    acc: np.ndarray
    prompts: list[PromptMark]
    sequence: Sequence | None  # for redrawing the prompter
    scroll_start_s: float | None  # trial time of the prompter's t = 0
    hand: str
    arm: str  # arm wearing the band
    lookahead_s: float
    effort: EffortTrace | None = None
    trial_id: str = ""  # "<session>/<trial>" (trials.TrialInfo.id)

    @property
    def duration_s(self) -> float:
        return float(self.emg_t[-1]) if len(self.emg_t) else 0.0


def resolve_trial(path: Path | None, trial: int, data_dir: Path) -> Path:
    """Trial .npz from a file, a session folder (+ 1-based trial number), or the latest session."""
    if path is not None and path.suffix == ".npz":
        return path
    if path is None:
        sessions = sorted(p for p in data_dir.iterdir() if (p / "trials.jsonl").exists())
        if not sessions:
            raise SystemExit(f"no sessions with trials in {data_dir}")
        path = sessions[-1]
    records = trials.read_records(path)
    if not 1 <= trial <= len(records):
        raise SystemExit(f"{path} has trials 1-{len(records)}, not {trial}")
    return path / Path(records[trial - 1]["npz_path"]).name


def load_trial(npz_path: Path) -> ReplayData:
    trial = trials.load_trial(Path(npz_path), keep_arrays=True)
    info, meta = trial.info, trial.info.metadata
    t0 = int(trial.emg_t_ns[0])

    def rel(ns):
        return (np.asarray(ns, dtype=np.int64) - t0) / 1e9

    prompts, sequence, scroll_start_s, effort = [], None, None, None
    if trial.prompts is not None:
        p = trial.prompts
        seq_prompts = []
        for i in range(len(p)):
            cols = np.flatnonzero(p.digits[i])
            digits = tuple(p.digit_names[c] for c in cols)
            levels = tuple(p.level_names[p.level[i, c]] for c in cols)
            label = "+".join(digits) + " " + "/".join(LEVEL_TAGS.get(lv, lv) for lv in levels)
            prompts.append(
                PromptMark(rel(p.onset_ns[i]), rel(p.offset_ns[i]), label, DIGIT_COLORS[digits[0]])
            )
            seq_prompts.append(
                Prompt(float(p.onset_s[i]), float(p.offset_s[i] - p.onset_s[i]), digits, levels)
            )
        # The recorded coding, so the target rings are the ones the subject saw.
        sequence = Sequence(info.label, meta.get("task_set", ""), seq_prompts,
                            {"level_names": list(p.level_names)})  # fmt: skip
        scroll_start_s = float(rel(trial.scroll_start_ns))
        arr = trial.arrays
        if "effort" in arr and len(arr["effort"]):
            zones = (meta.get("effort_meter") or {}).get("zones") or ZONES
            effort = EffortTrace(rel(arr["effort_host_ns"]), arr["effort"].astype(float),
                                 [p.level_names[c] if c >= 0 else None
                                  for c in arr["effort_level"].astype(int)],
                                 {k: tuple(v) for k, v in zones.items()})  # fmt: skip

    return ReplayData(
        name=info.label,
        npz_path=info.npz_path,
        t0_ns=t0,
        emg_t=rel(trial.emg_t_ns),
        emg_uv=trial.emg_uv.astype(np.float32),
        imu_t=rel(trial.imu_t_ns),
        gyro=trial.gyro,
        acc=trial.acc,
        prompts=prompts,
        sequence=sequence,
        scroll_start_s=scroll_start_s,
        hand=meta.get("hand", "right"),
        arm=info.arm,
        lookahead_s=float(meta.get("lookahead_s", DEFAULT_LOOKAHEAD_S)),
        effort=effort,
        trial_id=info.id,
    )


# --- Camera (framework only) ------------------------------------------------------------------


class VideoSource(Protocol):
    """A camera recording that can be shown in step with the EMG.

    To add camera support: implement this for a video file plus its alignment to the host clock
    (e.g. a frame timestamp log, or an offset found with the --sync-patch flash), and return it
    from :func:`find_video_sources`.
    """

    name: str

    def frame_at(self, t_ns: int) -> QtGui.QImage | None:
        """The frame showing host time ``t_ns``, or None outside the recording."""
        ...


def find_video_sources(npz_path: Path) -> list[VideoSource]:
    """Camera recordings belonging to a trial. None exist yet."""
    return []


class CameraPanel(QtWidgets.QWidget):
    """Shows the current frame of each video source, or a placeholder when there are none."""

    def __init__(self, sources: list[VideoSource]):
        super().__init__()
        self.sources = sources
        self.frames: list[QtGui.QImage | None] = [None] * len(sources)
        self.setMinimumHeight(160)

    def show_time(self, t_ns: int) -> None:
        if self.sources:
            self.frames = [s.frame_at(t_ns) for s in self.sources]
            self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#000000"))
        if not self.sources:
            p.setPen(QtGui.QColor("#888888"))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "No camera footage")
            return
        w = self.width() / len(self.sources)
        for i, (src, frame) in enumerate(zip(self.sources, self.frames, strict=True)):
            box = QtCore.QRectF(i * w, 0, w, self.height())
            if frame is None:
                p.setPen(QtGui.QColor("#888888"))
                p.drawText(box, QtCore.Qt.AlignCenter, f"{src.name}: no frame")
                continue
            img = frame.scaled(int(box.width()), int(box.height()), QtCore.Qt.KeepAspectRatio)
            p.drawImage(
                QtCore.QPointF(
                    box.x() + (box.width() - img.width()) / 2, (box.height() - img.height()) / 2
                ),
                img,
            )


# --- Window -----------------------------------------------------------------------------------


class ReplayPanel:
    """More to replay with the trial (``ReplayWindow``'s ``panels``): ``widget`` (if any) goes
    under the EMG traces, ``height`` px tall to start with, ``caption`` (if any) under the
    prompter, ``status`` in the transport bar."""

    widget: QtWidgets.QWidget | None = None
    height: int = 380
    caption: str = ""

    def attach(self, window: ReplayWindow) -> None:
        """Once, before the window is laid out (``window.prompter`` is None for a trial without
        prompts): e.g. add a prompter overlay, set ``caption``."""

    def draw(self, now: float, window_s: float) -> None:
        """At each render: ``now`` s into the trial, ``window_s`` s of EMG shown."""

    def status(self, now: float) -> str:
        """Text for the transport bar at ``now``."""
        return ""


class ReplayWindow(QtWidgets.QMainWindow):
    def __init__(
        self, data: ReplayData, window_s: float = 5.0, speed: float = 1.0, autoplay: bool = True,
        panels: list[ReplayPanel] | tuple[ReplayPanel, ...] = (),
    ):  # fmt: skip
        super().__init__()
        self.data = data
        self.setWindowTitle(f"Replay – {data.name}")
        # Trial time at the right edge of the traces. Starting at 0 with an empty screen, new
        # data enters at the right edge and slides left, exactly as in the live viewer.
        self.now = 0.0
        self.playing = False
        self._last_tick = time.perf_counter()
        self._last_spectrum = -1.0

        pg.setConfigOptions(antialias=False)
        self.controls = DisplayControls(window_s, self.auto_spacing, pause=False)
        self.traces = TracePlot()
        self.spectrum = SpectrumPlot()
        self.imu_plots = ImuPlots()
        self.camera = CameraPanel(find_video_sources(data.npz_path))
        self.band = BandRms(data.arm, self.controls)
        for pr in data.prompts:
            # Appears when the press starts, then scrolls left with the EMG.
            self.traces.add_marker(pr.onset_s, pr.label, pr.color, width=1.5)

        self.prompter = None
        if data.sequence is not None:
            self.prompter = PrompterWidget(data.hand, data.lookahead_s)
            self.prompter.sequence = data.sequence
            self.prompter.setMinimumSize(260, 400)
            if data.effort is not None:  # the bar the subject saw
                self.prompter.show_effort = True
                self.prompter.effort_zones = dict(data.effort.zones)
        self.panels = list(panels)
        for pn in self.panels:
            pn.attach(self)

        # Transport
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems([f"{s:g}x" for s in SPEEDS])
        self.speed_combo.setCurrentIndex(SPEEDS.index(speed) if speed in SPEEDS else 3)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, int(data.duration_s * 1000))
        self.slider.sliderMoved.connect(lambda ms: self.seek(ms / 1000))
        self.time_label = QtWidgets.QLabel()
        self.prompt_label = QtWidgets.QLabel()
        self.panel_labels = [QtWidgets.QLabel() for _ in self.panels]
        transport = QtWidgets.QHBoxLayout()
        for w in (self.play_btn, QtWidgets.QLabel("Speed"), self.speed_combo):
            transport.addWidget(w)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.time_label)
        transport.addSpacing(16)
        transport.addWidget(self.prompt_label)
        for label in self.panel_labels:
            transport.addSpacing(16)
            transport.addWidget(label)

        top_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        top_right.addWidget(self.camera)
        top_right.addWidget(self.band.widget)
        top_right.setSizes([160, 320])
        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        for w in (top_right, self.spectrum, self.imu_plots.gyro, self.imu_plots.acc):
            right.addWidget(w)
        right.setSizes([340, 190, 150, 150])
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        captions = [pn.caption for pn in self.panels if pn.caption]
        if self.prompter is not None and captions:
            left = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(left)
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(self.prompter, 1)
            for text in captions:
                caption = QtWidgets.QLabel(text)
                caption.setWordWrap(True)
                column.addWidget(caption)
            split.addWidget(left)
        elif self.prompter is not None:
            split.addWidget(self.prompter)
        below = [pn for pn in self.panels if pn.widget is not None]
        if below:  # under the EMG traces
            middle = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            middle.addWidget(self.traces)
            for pn in below:
                middle.addWidget(pn.widget)
            middle.setSizes([520] + [pn.height for pn in below])
            split.addWidget(middle)
        else:
            split.addWidget(self.traces)
        split.addWidget(right)
        split.setSizes([320, 760, 420] if self.prompter is not None else [900, 500])

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.controls)
        layout.addWidget(split, 1)
        layout.addLayout(transport)
        self.setCentralWidget(central)
        self.resize(1600, 950)

        for sig in (
            self.controls.window_spin.valueChanged,
            self.controls.spacing_spin.valueChanged,
            self.controls.offset_box.toggled,
            self.controls.spec_combo.currentIndexChanged,
        ):
            sig.connect(lambda *_: self.render(force_spectrum=True))
        QtCore.QTimer(self, timeout=self._tick, interval=FRAME_MS).start()
        self.render(force_spectrum=True)
        if autoplay:
            self.toggle_play()

    # --- playback ----------------------------------------------------------------------------

    @property
    def speed(self) -> float:
        return SPEEDS[self.speed_combo.currentIndex()]

    def toggle_play(self) -> None:
        if not self.playing and self.now >= self.data.duration_s:
            self.now = 0.0
        self.playing = not self.playing
        self._last_tick = time.perf_counter()
        self.play_btn.setText("Pause" if self.playing else "Play")

    def seek(self, t: float) -> None:
        self.now = float(np.clip(t, 0.0, self.data.duration_s))
        self.render(force_spectrum=True)

    def jump_prompt(self, direction: int) -> None:
        onsets = [p.onset_s for p in self.data.prompts]
        if direction > 0:
            later = [t for t in onsets if t > self.now + 1e-3]
            if later:
                self.seek(later[0])
        else:
            earlier = [t for t in onsets if t < self.now - 1e-3]
            if earlier:
                self.seek(earlier[-1])

    def auto_spacing(self) -> None:
        d = self.data
        sel = (d.emg_t >= self.now - self.controls.window_s) & (d.emg_t <= self.now)
        # With little on screen (e.g. at the start), size from the first window of the trial.
        x = d.emg_uv[sel] if sel.sum() >= 1000 else d.emg_uv[: int(self.controls.window_s * 2000)]
        self.controls.spacing_spin.setValue(auto_spacing(x))

    def _tick(self) -> None:
        tick = time.perf_counter()
        if self.playing:
            self.now += (tick - self._last_tick) * self.speed
            if self.now >= self.data.duration_s:
                self.now = self.data.duration_s
                self.toggle_play()
            self.render()
        self._last_tick = tick

    # --- drawing -----------------------------------------------------------------------------

    def render(self, force_spectrum: bool = False) -> None:
        d, now, win = self.data, self.now, self.controls.window_s
        a, b = np.searchsorted(d.emg_t, [now - win, now])
        self.traces.draw(
            d.emg_t[a:b] - now, d.emg_uv[a:b], self.controls.spacing, win,
            self.controls.offset_box.isChecked(),
        )  # fmt: skip
        self.traces.place_markers(now)
        ia, ib = np.searchsorted(d.imu_t, [now - win, now])
        self.imu_plots.draw(d.imu_t[ia:ib] - now, d.gyro[ia:ib], d.acc[ia:ib], win)
        if force_spectrum or abs(now - self._last_spectrum) >= SPECTRUM_EVERY_S:
            sa = np.searchsorted(d.emg_t, now - 2.0)
            self.spectrum.draw(d.emg_uv[sa:b], self.controls.spec_combo.currentIndex())
            self._last_spectrum = now
        ra = np.searchsorted(d.emg_t, now - self.band.window_s)
        self.band.draw(d.emg_uv[ra:b])
        if self.prompter is not None:
            self.prompter.t = now - d.scroll_start_s
            self.prompter.header = d.name
            if d.effort is not None:
                self.prompter.effort = d.effort.at(now)
            self.prompter.update()
        for pn, label in zip(self.panels, self.panel_labels, strict=True):
            pn.draw(now, win)
            label.setText(pn.status(now))
        self.camera.show_time(d.t0_ns + int(now * 1e9))

        self.slider.blockSignals(True)
        self.slider.setValue(int(now * 1000))
        self.slider.blockSignals(False)
        self.time_label.setText(f"{now:6.2f} / {d.duration_s:.2f} s")
        active = [p.label for p in d.prompts if p.onset_s <= now < p.offset_s]
        self.prompt_label.setText(f"prompt: {', '.join(active) or 'rest'}" if d.prompts else "")

    # --- keys --------------------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        Qt, key = QtCore.Qt, event.key()
        step = 5.0 if event.modifiers() & Qt.ShiftModifier else 1.0
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Left:
            self.seek(self.now - step)
        elif key == Qt.Key_Right:
            self.seek(self.now + step)
        elif key == Qt.Key_Comma:
            self.jump_prompt(-1)
        elif key == Qt.Key_Period:
            self.jump_prompt(+1)
        elif key == Qt.Key_Home:
            self.seek(0)
        elif key == Qt.Key_End:
            self.seek(self.data.duration_s)
        elif key == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", type=Path, nargs="?", help="trial .npz or session folder")
    ap.add_argument("--trial", type=int, default=1, help="trial number in a session (1-based)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--window", type=float, default=5.0, help="seconds of data shown")
    ap.add_argument("--speed", type=float, default=1.0, choices=SPEEDS)
    ap.add_argument("--paused", action="store_true", help="open paused instead of playing")
    ap.add_argument(
        "--arm", choices=["right", "left"], help="arm wearing the band (default: from session)"
    )
    return ap


def run(args: argparse.Namespace, path: Path | None = None, panels=None) -> None:
    """Replay for ``parser()``'s ``args`` (``path``: instead of ``args.path``) until the window
    is closed; ``panels(data)``, if given, makes the window's :class:`ReplayPanel` list."""
    npz = resolve_trial(path or args.path, args.trial, args.data_dir)
    print(f"Replaying {npz}")
    app = pg.mkQApp("EMG replay")
    data = load_trial(npz)
    if args.arm:
        data.arm = args.arm
    win = ReplayWindow(data, args.window, args.speed, autoplay=not args.paused,
                       panels=panels(data) if panels is not None else ())  # fmt: skip
    win.show()
    sys.exit(app.exec())


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
