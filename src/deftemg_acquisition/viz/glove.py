"""The live viewer's glove panels, for a Wuji Glove's LSL streams.

They read the streams published by ``WujiGloveRecorder(live=True)`` (``acquire`` / ``prompter``
with ``--wuji``), type ``Glove_<stream>``. Each panel
stays hidden until its stream appears, so armband-only sessions look as before. Under the band
RMS:

- **Tactile hand** (``tactile_point_cloud``, ``tip_poses``): every taxel in 3D (wrist frame),
  coloured by its latest pressure (grey: no reading), fingertips labelled. Drag to rotate,
  scroll to zoom, middle- or ctrl-drag to pan; **colour max** sets the pressure shown as full
  colour, **Reset view** re-centres. Needs PyOpenGL; left out without it.
- **Joint angles** (``hand_joint_angles``): one joint of each finger (the menu: joint 1 is the
  first of the finger's angles in the glove's order), in degrees, over the EMG time window.
- **Tactile (zone mean)** (``tactile_zones``): mean pressure per zone (taxels with a reading),
  thumb to pinky and palm, over the EMG time window.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
import pylsl
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from deftemg_acquisition.acquisition.lsl import STREAM_PREFIX
from deftemg_acquisition.acquisition.wuji.wuji_recorder import live_stream_type
from deftemg_acquisition.viz.panels import MAX_WINDOW_S
from deftemg_acquisition.viz.viewer import Ring, ViewerPanel

log = logging.getLogger(__name__)

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
ZONES = [*FINGERS, "palm"]
COLORS = ["#e15759", "#f28e2b", "#edc948", "#59a14f", "#4e79a7", "#b07aa1"]
NO_READING = (0.45, 0.45, 0.45, 0.35)  # RGBA of a taxel with no reading
RATE_CAP_HZ = 150  # trace capacity per second (the glove streams at about 120 Hz)


def _labels(info: pylsl.StreamInfo) -> list[str]:
    ch = info.desc().child("channels").child("channel")
    out = []
    while not ch.empty():
        out.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return out


def field_columns(labels: list[str], field: str) -> np.ndarray:
    """Columns of a frame field in a glove stream's flattened row (``wuji_recorder``'s
    ``flatten_fields``: ``field`` alone, or ``field_0``, ``field_1`` ...)."""
    return np.array([i for i, lab in enumerate(labels)
                     if lab == field or lab.rpartition("_")[0] == field], int)  # fmt: skip


class _StreamFinder:
    """One background thread listing the LSL streams every second, for every ``GloveInlet``.

    Used instead of a ``ContinuousResolver`` per stream: several of those in one process were
    seen to miss streams that appeared after they started, for a minute or more. A full
    listing found them every time.
    """

    def __init__(self):
        self._wanted: set[str] = set()
        self._found: dict[str, pylsl.StreamInfo] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def want(self, stype: str) -> None:
        with self._lock:
            self._wanted.add(stype)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="glove-find")
                self._thread.start()

    def take(self, stype: str) -> pylsl.StreamInfo | None:
        """The stream of type ``stype`` once found (then no longer searched for)."""
        with self._lock:
            info = self._found.pop(stype, None)
            if info is not None:
                self._wanted.discard(stype)
            return info

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._wanted:
                    self._thread = None
                    return
            for info in pylsl.resolve_streams(1.0):
                with self._lock:
                    if info.type() in self._wanted and info.name().startswith(STREAM_PREFIX):
                        self._found.setdefault(info.type(), info)


_finder = _StreamFinder()


class GloveInlet:
    """One glove stream, found in the background. Each new frame goes through ``select``
    (made from the stream's channel labels once found: (n, channels) -> (n, k)) into
    ``ring``, which keeps ``capacity`` rows."""

    def __init__(
        self, stream: str, select: Callable[[list[str]], Callable[[np.ndarray], np.ndarray]],
        capacity: int,
    ):  # fmt: skip
        self.stype = live_stream_type(stream)
        self._make_select = select
        self._capacity = capacity
        self.inlet: pylsl.StreamInlet | None = None
        self.labels: list[str] = []
        self.ring: Ring | None = None
        self.n = 0  # frames received
        _finder.want(self.stype)

    def poll(self) -> bool:
        """Pull new frames; True when the stream was found just now."""
        found = False
        if self.inlet is None:
            info = _finder.take(self.stype)
            if info is None:
                return False
            self.inlet = pylsl.StreamInlet(
                info, max_buflen=5, processing_flags=pylsl.proc_clocksync
            )
            self.labels = _labels(self.inlet.info())
            self._select = self._make_select(self.labels)
            k = self._select(np.zeros((1, len(self.labels)), np.float32)).shape[1]
            self.ring = Ring(self._capacity, k)
            found = True
        x, t = self.inlet.pull_chunk(timeout=0.0)
        if t:
            self.ring.extend(np.asarray(t), self._select(np.asarray(x, np.float32)))
            self.n += len(t)
        return found


def _header(title: str) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout, QtWidgets.QHBoxLayout]:
    widget = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(widget)
    lay.setContentsMargins(0, 0, 0, 0)
    head = QtWidgets.QHBoxLayout()
    lay.addLayout(head)
    widget.setVisible(False)  # until its stream appears
    return widget, lay, head


class GloveTraces(ViewerPanel):
    """Traces from one glove stream over the viewer's time window. ``select(labels)`` gives
    the function mapping frames (n, channels) to the values kept (n, k); ``show(y)`` maps
    those to the (n, len(names)) traces drawn."""

    def __init__(
        self, title: str, stream: str, names: list[str], units: str,
        select: Callable[[list[str]], Callable[[np.ndarray], np.ndarray]],
        show: Callable[[np.ndarray], np.ndarray] = lambda y: y, height: int = 200,
    ):  # fmt: skip
        self.title, self.show, self.height = title, show, height
        self.inlet = GloveInlet(stream, select, int(MAX_WINDOW_S * RATE_CAP_HZ))
        self.widget, lay, self.head = _header(title)
        self.label = QtWidgets.QLabel()
        self.head.addWidget(self.label)
        self.head.addStretch()
        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", units)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.setLabel("bottom", "s")
        self.plot.addLegend(offset=(5, 5), labelTextSize="7pt", colCount=3)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setMouseEnabled(x=False, y=True)
        self.plot.setMinimumHeight(120)  # the side column's splitter would squeeze it away
        self.curves = [
            self.plot.plot(pen=pg.mkPen(COLORS[i % len(COLORS)], width=1.5), name=n)
            for i, n in enumerate(names)
        ]
        lay.addWidget(self.plot)
        self.label.setText(f"<b>{title}</b> &nbsp; <i>live</i>")

    def poll(self) -> None:
        if self.inlet.poll():
            self.widget.setVisible(True)

    def draw(self, now: float, window_s: float) -> None:
        if self.inlet.ring is None:
            return
        t, y = self.inlet.ring.latest(int(window_s * RATE_CAP_HZ))
        keep = t >= now - window_s
        t, y = t[keep] - now, self.show(y[keep])
        for i, c in enumerate(self.curves):
            c.setData(t, y[:, i])
        self.plot.setXRange(-window_s, 0, padding=0)


def zone_means(labels: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    """``tactile_zones`` frames -> mean pressure per zone (``ZONES`` order) over the taxels
    with a reading (-1 marks none; 0 when a zone has none)."""
    cols = [field_columns(labels, z) for z in ZONES]

    def select(x: np.ndarray) -> np.ndarray:
        out = np.zeros((len(x), len(ZONES)), np.float32)
        for i, c in enumerate(cols):
            v = x[:, c]
            valid = v >= 0
            out[:, i] = np.where(valid, v, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        return out

    return select


def joint_angles_deg(labels: list[str]) -> Callable[[np.ndarray], np.ndarray]:
    """``hand_joint_angles`` frames -> the angles (rad on the glove) in degrees."""
    cols = field_columns(labels, "angles")
    return lambda x: np.degrees(x[:, cols])


class JointAngles(GloveTraces):
    """One joint of each finger, chosen in the panel's menu, in degrees."""

    def __init__(self, height: int = 200):
        super().__init__("Glove joint angles", "hand_joint_angles", FINGERS, "deg",
                         joint_angles_deg, show=self._joint, height=height)  # fmt: skip
        self.joint = QtWidgets.QComboBox()
        self.joint.addItems([f"joint {j + 1}" for j in range(5)])
        self.head.addWidget(self.joint)

    def _joint(self, y: np.ndarray) -> np.ndarray:
        """(n, fingers x angles per finger) -> (n, fingers): the chosen joint of each."""
        per = max(1, y.shape[1] // len(FINGERS))
        j = min(self.joint.currentIndex(), per - 1)
        return y[:, [f * per + j for f in range(len(FINGERS))]]


def cloud_colors(pressure: np.ndarray, vmax: float, cmap: pg.ColorMap) -> np.ndarray:
    """RGBA (float) per taxel: ``cmap`` over 0..``vmax``; grey where there's no reading."""
    valid = pressure >= 0
    colors = np.empty((len(pressure), 4), np.float32)
    colors[~valid] = NO_READING
    if valid.any():
        colors[valid] = cmap.map(np.clip(pressure[valid] / vmax, 0, 1), "float")
    return colors


def _latest_fields(*fields: str) -> Callable[[list[str]], Callable[[np.ndarray], np.ndarray]]:
    """Keep only the columns of ``fields``, in that order."""

    def make(labels: list[str]) -> Callable[[np.ndarray], np.ndarray]:
        cols = np.concatenate([field_columns(labels, f) for f in fields])
        return lambda x: x[:, cols]

    return make


class TactileHand(ViewerPanel):
    """The glove's taxels in 3D (wrist frame), coloured by their latest pressure."""

    def __init__(self, height: int = 420):
        import pyqtgraph.opengl as gl  # needs PyOpenGL: imported only when the panel is used

        self.height = height
        self.cloud = GloveInlet("tactile_point_cloud", _latest_fields("position", "pressure"), 2)
        self.tips = GloveInlet("tip_poses", _latest_fields("position"), 2)
        self.cmap = pg.colormap.get("viridis")
        self._centred = False

        self.widget, lay, head = _header("Tactile hand")
        self.label = QtWidgets.QLabel()
        head.addWidget(self.label)
        head.addStretch()
        head.addWidget(QtWidgets.QLabel("colour max"))
        self.vmax = QtWidgets.QDoubleSpinBox()
        self.vmax.setRange(0.01, 10.0)
        self.vmax.setSingleStep(0.05)
        self.vmax.setValue(0.3)
        head.addWidget(self.vmax)
        reset = QtWidgets.QPushButton("Reset view")
        reset.clicked.connect(self.reset_view)
        head.addWidget(reset)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((28, 28, 32))
        self.view.setMinimumHeight(260)  # the side column's splitter would squeeze it away
        self.scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=6, pxMode=True)
        self.scatter.setGLOptions("translucent")
        self.view.addItem(self.scatter)
        self.tip_text = [gl.GLTextItem(text=f, color=(255, 255, 255, 220)) for f in FINGERS]
        for t in self.tip_text:
            self.view.addItem(t)
        lay.addWidget(self.view)
        self._status("waiting for stream…")

    def _status(self, text: str) -> None:
        self.label.setText(f"<b>Glove tactile hand</b> &nbsp; <i>{text}</i>")

    def reset_view(self) -> None:
        self._centred = False

    def poll(self) -> None:
        if self.cloud.poll():
            self.widget.setVisible(True)
        self.tips.poll()

    def latest(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The latest cloud: positions (mm) and pressures of the points present."""
        if self.cloud.ring is None or self.cloud.ring.n == 0:
            return None
        row = self.cloud.ring.latest(1)[1][0]
        n = len(row) // 4  # x, y, z per point, then a pressure per point
        pos, p = row[: 3 * n].reshape(n, 3) * 1000.0, row[3 * n :]
        ok = np.isfinite(pos).all(axis=1)
        return pos[ok], p[ok]

    def draw(self, now: float, window_s: float) -> None:
        cloud = self.latest()
        if cloud is None:
            return
        pos, p = cloud
        self.scatter.setData(pos=pos, color=cloud_colors(p, self.vmax.value(), self.cmap))
        if not self._centred and len(pos):
            self.view.setCameraPosition(
                pos=pg.Vector(*pos.mean(axis=0)), distance=float(np.ptp(pos, axis=0).max() * 2.2),
                elevation=30, azimuth=-60,
            )  # fmt: skip
            self._centred = True
        if self.tips.ring is not None and self.tips.ring.n:
            tips = self.tips.ring.latest(1)[1][0].reshape(-1, 3) * 1000.0
            for t, xyz in zip(self.tip_text, tips, strict=False):
                t.setData(pos=xyz)
        valid = p >= 0
        self._status(f"live · {int(valid.sum())} taxels · peak {p[valid].max(initial=0):.3f}")


def glove_panels(hand: bool = True) -> list[ViewerPanel]:
    """The glove panels for the viewer's side column (needs a QApplication). ``hand``: the
    3D tactile hand too (left out, with a warning, without PyOpenGL)."""
    panels: list[ViewerPanel] = []
    if hand:
        try:
            panels.append(TactileHand())
        except ImportError as e:
            log.warning("glove: no 3D tactile hand (%s); install PyOpenGL for it", e)
    panels.append(JointAngles())
    panels.append(
        GloveTraces("Glove tactile (zone mean)", "tactile_zones", ZONES, "pressure", zone_means)
    )
    return panels
