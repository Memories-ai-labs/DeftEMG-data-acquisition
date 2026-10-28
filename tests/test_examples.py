"""The shipped example session and protocol files stay loadable."""

from pathlib import Path

from deftemg_acquisition.data.trials import load_trial, scan_trials, select_trials
from deftemg_acquisition.tasks.sequence import Sequence

ROOT = Path(__file__).resolve().parents[1]


def test_example_session_loads():
    session = ROOT / "examples" / "S01"
    usable, skipped = select_trials(scan_trials([session]))
    assert len(usable) == 2 and [reason for _, reason in skipped] == ["calibration block"]
    trial = load_trial(usable[0].npz_path)
    assert trial.emg_uv.shape[1] == 18 and len(trial.prompts.onset_ns) == 8


def test_protocol_files_load():
    paths = sorted((ROOT / "protocols").rglob("*.json"))
    assert len(paths) == 62
    for path in paths:
        assert Sequence.load(path).prompts, path
