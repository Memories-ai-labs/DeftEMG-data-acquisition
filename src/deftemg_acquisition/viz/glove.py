"""The live viewer's glove panel: a Wuji Glove's tactile map, joint angles and contact trace.

It reads the glove's LSL streams (``Glove_tactile``, ``Glove_hand_joint_angles``), published by
``WujiGloveRecorder(live=True)`` (``acquire`` / ``prompter`` with ``--wuji``, ``--wuji-simulate``
or ``--wuji-replay``), and stays hidden until one of them appears, so armband-only sessions
look as before. On the panel:

- **Tactile map:** the latest tactile frame, 768 values as a 24 x 32 grid (rows top to bottom),
  0 to 1; grey where the glove has no taxel (-1).
- **Joint angles:** the latest frame's angles (rad), five per finger, thumb to pinky.
- **Contact:** mean tactile value over the taxels, on the same time axis as the EMG traces.
"""

from __future__ import annotations

import numpy as np
import pylsl
import pyqtgraph as pg

from deftemg_acquisition.acquisition.live import STREAM_PREFIX
from deftemg_acquisition.acquisition.wuji.wuji_recorder import live_stream_type
from deftemg_acquisition.viz.panels import MAX_WINDOW_S
from deftemg_acquisition.viz.viewer import Ring, ViewerPanel

TACTILE_GRID = (24, 32)
FINGER_COLORS = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7", "#b07aa1"]  # thumb .. pinky
NO_TAXEL = -0.05  # image level of absent taxels: below 0, drawn grey
RATE_CAP_HZ = 150  # contact trace capacity per second


def _labels(info: pylsl.StreamInfo) -> list[str]:
    ch = info.desc().child("channels").child("channel")
    out = []
    while not ch.empty():
        out.append(ch.child_value("label"))
        ch = ch.next_sibling()
    return out


class _LatestInlet:
    """One glove LSL stream: found in the background; keeps the newest frame and its time."""

    def __init__(self, stream: str):
        pred = f"type='{live_stream_type(stream)}' and starts-with(name,'{STREAM_PREFIX}')"
        self._resolver = pylsl.ContinuousResolver(pred=pred)
        self.inlet: pylsl.StreamInlet | None = None
        self.labels: list[str] = []
        self.n = 0  # frames received

    def poll(self) -> tuple[np.ndarray, np.ndarray]:
        """New frames since the last poll: (timestamps, rows)."""
        empty = np.zeros(0), np.zeros((0, len(self.labels)), np.float32)
        if self.inlet is None:
            found = self._resolver.results()
            if not found:
                return empty
            self.inlet = pylsl.StreamInlet(
                found[0], max_buflen=5, processing_flags=pylsl.proc_clocksync
            )
            self.labels = _labels(self.inlet.info())
            empty = np.zeros(0), np.zeros((0, len(self.labels)), np.float32)
        x, t = self.inlet.pull_chunk(timeout=0.0)
        if not t:
            return empty
        self.n += len(t)
        return np.asarray(t), np.asarray(x, np.float32)


class GlovePanel(ViewerPanel):
    """Viewer side panel for a Wuji Glove's live streams (hidden until one is found)."""

    height = 380

    def __init__(self):
        self.tactile = _LatestInlet("tactile")
        self.angles = _LatestInlet("hand_joint_angles")
        self.contact = Ring(int(MAX_WINDOW_S * RATE_CAP_HZ), 1)
        self._tactile_frame: np.ndarray | None = None
        self._angle_frame: np.ndarray | None = None

        self.widget = pg.GraphicsLayoutWidget()
        self.widget.setVisible(False)
        self.title = self.widget.addLabel("Glove", row=0, col=0, colspan=2)
        vb = self.widget.addViewBox(row=1, col=0, lockAspect=True, enableMouse=False)
        vb.invertY(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        lut = pg.colormap.get("inferno").getLookupTable(nPts=256)
        lut[0] = (70, 70, 70)  # absent taxels
        self.image.setLookupTable(lut)
        self.image.setLevels((NO_TAXEL, 1.0))
        vb.addItem(self.image)

        self.angle_plot = self.widget.addPlot(row=1, col=1, title="joint angles (rad)")
        self.angle_plot.setMouseEnabled(False, False)
        self.angle_plot.hideAxis("bottom")
        self.angle_plot.setYRange(-0.6, 1.8)
        self.bars = pg.BarGraphItem(x=[], height=[], width=0.8)
        self.angle_plot.addItem(self.bars)

        self.contact_plot = self.widget.addPlot(row=2, col=0, colspan=2, title="contact")
        self.contact_plot.setMouseEnabled(False, False)
        self.contact_plot.setLabel("left", "mean")
        self.contact_curve = self.contact_plot.plot(pen=pg.mkPen("#edc948", width=1.5))
        self.widget.ci.layout.setRowStretchFactor(1, 3)
        self.widget.ci.layout.setRowStretchFactor(2, 2)

    def poll(self) -> None:
        t, x = self.tactile.poll()
        if len(t):
            valid = x >= 0
            mean = np.where(valid, x, 0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
            self.contact.extend(t, mean[:, None])
            self._tactile_frame = x[-1]
        _, a = self.angles.poll()
        if len(a):
            self._angle_frame = a[-1]
        if not self.widget.isVisible() and (self.tactile.inlet or self.angles.inlet):
            self.widget.setVisible(True)

    def draw(self, now: float, window_s: float) -> None:
        if not self.widget.isVisible():
            return
        if self._tactile_frame is not None:
            f = np.where(self._tactile_frame < 0, NO_TAXEL, self._tactile_frame)
            shape = TACTILE_GRID if f.size == np.prod(TACTILE_GRID) else (1, f.size)
            self.image.setImage(f.reshape(shape), autoLevels=False)
        if self._angle_frame is not None:
            idx = [i for i, label in enumerate(self.angles.labels) if label.startswith("angles")]
            heights = self._angle_frame[idx]
            per = max(1, len(heights) // len(FINGER_COLORS))
            finger = np.arange(len(heights)) // per
            self.bars.setOpts(
                x=np.arange(len(heights)) + 0.8 * finger, height=heights,
                brushes=[FINGER_COLORS[k % len(FINGER_COLORS)] for k in finger],
            )  # fmt: skip
        t, c = self.contact.latest(int(window_s * RATE_CAP_HZ))
        keep = t - now >= -window_s
        self.contact_curve.setData(t[keep] - now, c[keep, 0])
        self.contact_plot.setXRange(-window_s, 0, padding=0)
        recent = t[t >= t[-1] - 1.0] if len(t) else t
        rate = len(recent) - 1 if len(recent) > 1 else 0
        self.title.setText(f"Wuji glove · tactile {rate} Hz · {self.tactile.n} frames")
