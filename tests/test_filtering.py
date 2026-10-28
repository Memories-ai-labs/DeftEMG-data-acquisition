import numpy as np
import pytest

from deftemg_acquisition.analysis.filtering import (
    FilterConfig,
    design_filter,
    filter_emg,
    notch_frequencies,
)
from deftemg_acquisition.errors import ConfigError

FS = 2000.0


def tone(f, n, amp=1.0):
    return amp * np.sin(2 * np.pi * f * np.arange(n) / FS)


def amp_at(y, f):
    """Amplitude of the f Hz component of y (a whole number of cycles in the window)."""
    t = np.arange(len(y)) / FS
    return 2 * abs(np.mean(y * np.exp(-2j * np.pi * f * t)))


def test_notch_frequencies():
    assert notch_frequencies(FilterConfig(), FS) == [60.0]
    assert notch_frequencies(FilterConfig(harmonics=[3, 5, 9]), FS) == [60, 180, 300, 540]
    odd = notch_frequencies(FilterConfig(harmonics="odd"), FS)
    assert odd[:4] == [60, 180, 300, 420] and odd[-1] == 900  # 1020 is above fs/2
    assert len(notch_frequencies(FilterConfig(harmonics="all"), FS)) == 16  # 60 .. 960
    assert notch_frequencies(FilterConfig(notch_hz=None, harmonics="odd"), FS) == []
    assert design_filter(FilterConfig(highpass_hz=None, notch_hz=None), FS) is None


def test_default_removes_offset_drift_and_mains_keeps_emg_band():
    n = 20 * int(FS)
    rng = np.random.default_rng(0)
    emg = rng.standard_normal(n) * 10
    x = -4000 + 300 * tone(0.5, n) + 200 * tone(60, n) + tone(100, n, 50) + emg
    y = filter_emg(x[:, None], FS)[:, 0]
    mid = slice(int(FS), -int(FS))
    assert abs(y[mid].mean()) < 1  # offset gone without de-meaning first
    assert amp_at(y[mid], 0.5) < 0.5  # drift gone
    assert amp_at(y[mid], 60) < 1  # 200 µV hum gone
    assert amp_at(y[mid], 100) == pytest.approx(50, rel=0.02)  # EMG band untouched


def test_harmonics_are_notched_only_when_asked():
    n = 10 * int(FS)
    x = (tone(60, n, 100) + tone(180, n, 20) + tone(300, n, 20) + tone(200, n, 20))[:, None]
    mid = slice(int(FS), -int(FS))
    plain = filter_emg(x, FS)[mid, 0]
    comb = filter_emg(x, FS, FilterConfig(harmonics="odd"))[mid, 0]
    assert amp_at(plain, 180) == pytest.approx(20, rel=0.02)
    assert amp_at(comb, 180) < 0.2 and amp_at(comb, 300) < 0.2
    assert amp_at(comb, 200) == pytest.approx(20, rel=0.02)  # between harmonics: kept


def test_segments_are_filtered_separately():
    n = 4000
    x = np.zeros((2 * n, 1))
    x[n:] = 5000.0  # a 5 mV step at a gap between two segments
    joined = filter_emg(x, FS)
    split = filter_emg(x, FS, segments=[(0, n), (n, 2 * n)])
    assert np.abs(joined).max() > 1000  # filtering across the gap rings on the step
    assert np.abs(split).max() < 1e-6


def test_causal_starts_at_rest_level():
    x = np.full((4000, 2), -4000.0) + tone(100, 4000, 20)[:, None]
    y = filter_emg(x, FS, FilterConfig(zero_phase=False))
    assert np.abs(y).max() < 60  # no multi-mV ring from the offset at the first sample


def test_config_validation():
    with pytest.raises(ConfigError):
        FilterConfig(harmonics="even")
    with pytest.raises(ConfigError):
        FilterConfig(harmonics=[1])
    with pytest.raises(ConfigError):
        design_filter(FilterConfig(highpass_hz=1500), FS)
