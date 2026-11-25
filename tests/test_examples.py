"""The shipped example session and protocol files stay loadable."""

from pathlib import Path

from deftemg_acquisition.cli import make_task_sequences
from deftemg_acquisition.cli.make_task_sequences import PLAN
from deftemg_acquisition.data.trials import load_trial, scan_trials, select_trials
from deftemg_acquisition.tasks.sequence import Sequence
from deftemg_acquisition.tasks.textblock import TextBlock, load_block

ROOT = Path(__file__).resolve().parents[1]
DEMO_TASKS = ROOT / "protocols" / "demo_tasks"


def test_example_session_loads():
    session = ROOT / "examples" / "S01"
    usable, skipped = select_trials(scan_trials([session]))
    assert len(usable) == 2 and [reason for _, reason in skipped] == ["calibration block"]
    trial = load_trial(usable[0].npz_path)
    assert trial.emg_uv.shape[1] == 18 and len(trial.prompts.onset_ns) == 8


def test_protocol_files_load():
    # every block file, lane or text; a generated set's plan.json is not a block
    paths = sorted(p for p in (ROOT / "protocols").rglob("*.json") if p.name != PLAN)
    assert len(paths) == 65
    blocks = {path: load_block(path) for path in paths}
    for path, block in blocks.items():
        assert block.prompts, path
    text = sorted(p.relative_to(ROOT).as_posix() for p, b in blocks.items()
                  if isinstance(b, TextBlock))  # fmt: skip
    assert text == [f"protocols/demo_tasks/S01/{n}.json"
                    for n in ("b01_tennis_ball", "b02_mug", "b03_jar_and_pages")]  # fmt: skip
    assert sum(isinstance(b, Sequence) for b in blocks.values()) == 62


def test_demo_task_blocks_are_their_catalogue_generated(tmp_path):
    """The demo text blocks are what their catalogue gives with the seed they were made with."""
    out = tmp_path / "demo_tasks"
    make_task_sequences.main(
        ["--catalogue", str(DEMO_TASKS / "catalogue.toml"), "--out", str(out), "--seed", "1"]
    )
    shipped = sorted(p.relative_to(DEMO_TASKS) for p in DEMO_TASKS.glob("S??/*.json"))
    assert shipped == sorted(p.relative_to(out) for p in out.glob("S??/*.json"))
    for rel in shipped:
        assert (out / rel).read_text() == (DEMO_TASKS / rel).read_text(), rel
