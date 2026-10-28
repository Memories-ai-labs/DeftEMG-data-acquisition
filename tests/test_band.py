import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from deftemg_acquisition.viz.band import (  # noqa: E402
    PUCK_DEG,
    AutoScale,
    band_rms,
    channel_angles_deg,
    opening_angle_deg,
    side_labels,
)


def test_layout_matches_manual():
    a = channel_angles_deg()
    # ch9 and ch10 flank the puck at the top (dorsal), half a pod either side
    step = 360 / 19
    assert a[8] == pytest.approx(PUCK_DEG - step / 2) and a[9] == pytest.approx(PUCK_DEG + step / 2)
    # numbers increase counterclockwise (seen from the elbow), 360/19 degrees apart (one position
    # is the opening), all distinct
    steps = np.diff(np.unwrap(np.radians(a)))
    np.testing.assert_allclose(np.degrees(steps), 360 / 19)
    assert len(set(np.round(a, 6))) == 18
    # ch10 -> ch18 run down the left side; the opening (ch18 | ch1) is centred on volar
    assert all(90 < x < 270 for x in a[9:18])
    assert opening_angle_deg() == pytest.approx(270)
    # ch18 and ch1 sit one position either side of the opening (an empty position between them)
    assert a[17] == pytest.approx(270 - step) and a[0] == pytest.approx(270 + step)


def test_side_labels_follow_arm():
    assert side_labels("right")["left"].startswith("Radial")
    assert side_labels("left")["left"].startswith("Ulnar")
    assert side_labels("left")["top"] == "Dorsal"


def test_band_rms_and_autoscale():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 18)) * 5 + 100  # DC offset is removed
    x[:, 3] *= 4
    rms = band_rms(x)
    assert rms[3] == pytest.approx(20, rel=0.05) and rms[0] == pytest.approx(5, rel=0.05)
    scale = AutoScale(floor_uv=10)
    assert scale.update(rms) == pytest.approx(1.1 * rms.max())
    quiet = np.full(18, 1.0)
    later = [scale.update(quiet) for _ in range(500)]
    assert later[0] < 1.1 * rms.max() and later[-1] == pytest.approx(10)  # decays to the floor


def test_widget_renders():
    import pyqtgraph as pg

    from deftemg_acquisition.viz.band import BandRmsWidget
    from deftemg_acquisition.viz.panels import CHANNEL_COLORS

    pg.mkQApp()
    w = BandRmsWidget("left", CHANNEL_COLORS)
    w.resize(320, 300)
    w.set_rms(np.linspace(0, 60, 18), 50, 200)
    assert not w.grab().isNull()


def test_rms_scale_defaults_to_a_fixed_maximum():
    import pyqtgraph as pg

    from deftemg_acquisition.viz.panels import DEFAULT_RMS_MAX_UV, BandRms, DisplayControls

    pg.mkQApp()
    controls = DisplayControls(5.0, lambda: None)
    band = BandRms("right", controls)
    band.draw(np.random.default_rng(0).normal(size=(400, 18)) * 200)
    assert not controls.rms_auto_box.isChecked() and band.widget.scale_uv == DEFAULT_RMS_MAX_UV
    controls.rms_auto_box.setChecked(True)
    band.draw(np.random.default_rng(0).normal(size=(400, 18)) * 200)
    assert band.widget.scale_uv > 100  # auto follows the most active channel
