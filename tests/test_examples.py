"""The shipped example session and protocol files stay loadable."""

import subprocess
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


def shipped_protocol_files() -> list[Path]:
    """The block files in ``protocols/`` that git tracks (a set made there by following the
    README is not one of them), or every one found if git can't tell. A set's plan.json is not
    a block."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z", "--", "protocols"],
            cwd=ROOT, capture_output=True, check=True, text=True,
        ).stdout  # fmt: skip
        paths = [ROOT / f for f in out.split("\0") if f.endswith(".json")]
    except (OSError, subprocess.CalledProcessError):
        paths = []
    if not paths:  # not a git checkout (an unpacked source copy)
        paths = list((ROOT / "protocols").rglob("*.json"))
    return sorted(p for p in paths if p.name != PLAN)


def test_protocol_files_load():
    # every shipped block file, lane or text
    paths = shipped_protocol_files()
    assert len(paths) == 65
    blocks = {path: load_block(path) for path in paths}
    for path, block in blocks.items():
        assert block.prompts, path
    text = sorted(p.relative_to(ROOT).as_posix() for p, b in blocks.items()
                  if isinstance(b, TextBlock))  # fmt: skip
    assert text == [f"protocols/demo_tasks/S01/{n}.json"
                    for n in ("b01_tennis_ball", "b02_mug", "b03_jar_and_pages")]  # fmt: skip
    assert sum(isinstance(b, Sequence) for b in blocks.values()) == 62


def test_demo_task_blocks_are_their_catalogue_generated(tmp_path, capsys):
    """The demo text blocks are what their catalogue gives with the seed they were made with."""
    out = tmp_path / "demo_tasks"
    make_task_sequences.main(
        ["--catalogue", str(DEMO_TASKS / "catalogue.toml"), "--out", str(out), "--seed", "1"]
    )
    summary = capsys.readouterr().out  # its one session and calibration, in the singular
    assert "1 calibration " in summary and "1 session," in summary
    shipped = sorted(p.relative_to(DEMO_TASKS) for p in DEMO_TASKS.glob("S??/*.json"))
    assert shipped == sorted(p.relative_to(out) for p in out.glob("S??/*.json"))
    for rel in shipped:
        assert (out / rel).read_text() == (DEMO_TASKS / rel).read_text(), rel


def test_demo_task_blocks_show_the_new_screens():
    """The demo blocks carry what the prompter's screens need: the bold lead, the start screen's
    lines and the terms slide."""
    blocks = {p.stem: load_block(p) for p in sorted(DEMO_TASKS.glob("S01/*.json"))}
    mug = blocks["b02_mug"]
    assert mug.params["start"] == {
        "set_out": ["Coffee mug"],
        "task": "grasp as prompted, the hand shows which fingers",
    }
    for p in mug.prompts:
        assert p.bold_line == 2 and p.text.split("\n")[2] == "STAY ON TABLE", p
    assert blocks["b03_jar_and_pages"].params["start"] == {
        "set_out": ["Jar", "Book"],
        "task": "do each task as written",
    }
    terms = [t["term"] for t in mug.params["terms"]]
    assert terms[:3] == ["Light grasp", "Grasp and squeeze", "Stay on table"]
    assert all(b.params["terms"] == mug.params["terms"] for b in blocks.values())
    assert all(p.bold_line is None for p in blocks["b03_jar_and_pages"].prompts)
