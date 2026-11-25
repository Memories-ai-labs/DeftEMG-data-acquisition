"""The prompter command with text blocks: loading, the first calibration of each kind, session
conditions, and whole runs with the simulated armband and with a replayed recording."""

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg  # noqa: E402

from deftemg_acquisition.cli import make_task_sequences, prompter  # noqa: E402
from deftemg_acquisition.data.trials import read_records  # noqa: E402
from deftemg_acquisition.tasks.app import TaskWindow  # noqa: E402
from deftemg_acquisition.tasks.sequence import Prompt, Sequence, calibration_for  # noqa: E402
from deftemg_acquisition.tasks.textblock import GRASP_CALIBRATION, TextBlock, TextPrompt  # noqa: E402

CATALOGUE = Path(__file__).parents[1] / "protocols/grasp_natural/catalogue.toml"
BOTTLES = ["--condition", "bottle_fill_rubber=filled", "--condition", "bottle_fill_metal=empty"]


def _text(name: str, conditions=(), questions=None) -> TextBlock:
    params = {"conditions": list(conditions)}
    if questions is not None:
        params["condition_questions"] = questions
    prompts = [TextPrompt("Ball, power grasp", "ball", grasp="Power", highlight="hand")]
    return TextBlock(name, "grasp", "self_paced", prompts, params)


def _lane(name: str, task_set: str = "single_pinch") -> Sequence:
    return Sequence(name, task_set, [Prompt(0.2, 0.2, ("index",), ("high",))])


def _names(blocks) -> list[str]:
    return [b.name for _, b in blocks]


def test_the_first_block_of_each_kind_gets_its_calibration():
    lane, text = (Path("l.json"), _lane("l")), (Path("t.json"), _text("t"))
    only_lane = prompter.with_calibrations([lane, lane], 2)
    today = calibration_for("single_pinch", 2)  # exactly what a lane-only session always had
    assert [b.to_dict() for _, b in only_lane] == [today.to_dict(), *[lane[1].to_dict()] * 2]
    assert only_lane[0][0] == Path(f"built-in/{today.name}.json")
    assert _names(prompter.with_calibrations([text, text])) == [GRASP_CALIBRATION, "t", "t"]
    cal = today.name
    assert _names(prompter.with_calibrations([lane, text, lane], 2)) == [
        cal, "l", GRASP_CALIBRATION, "t", "l"]  # fmt: skip
    assert _names(prompter.with_calibrations([text, lane, text], 2)) == [
        GRASP_CALIBRATION, "t", cal, "l", "t"]  # fmt: skip


def test_condition_questions_come_from_the_blocks_that_use_them():
    q = {"bottle": {"ask": "Bottle filled?", "choices": ["filled", "empty"]}}
    blocks = [(Path("a.json"), _text("a", ["bottle"], q)), (Path("b.json"), _text("b")),
              (Path("c.json"), _text("c", ["lid"])), (Path("d.json"), _lane("d"))]  # fmt: skip
    assert prompter.condition_questions(blocks) == {"bottle": q["bottle"], "lid": None}
    other = {"bottle": {"ask": "Bottle filled?", "choices": ["full", "empty"]}}
    blocks.append((Path("e.json"), _text("e", ["bottle"], other)))
    with pytest.raises(SystemExit, match=r"bottle: a\.json and e\.json ask it differently"):
        prompter.condition_questions(blocks)
    with pytest.raises(SystemExit, match="need an ask and choices"):
        prompter.condition_questions([(Path("f.json"), _text("f", ["x"], {"x": ["a"]}))])
    for bad in ([], "", "Bottle filled?"):  # present, but not a table of conditions
        with pytest.raises(SystemExit, match=r"g\.json: condition_questions must be a table"):
            prompter.condition_questions([(Path("g.json"), _text("g", ["x"], bad))])


def test_ask_conditions():
    q = {"bottle": {"ask": "Bottle filled?", "choices": ["filled", "empty"]}, "lid": None}
    said: list[str] = []
    answers = iter(["", "full", "empty"])
    asked: list[str] = []

    def ask(text):
        asked.append(text)
        return next(answers)

    got = prompter.ask_conditions(q, {"lid": "on", "botle": "x"}, True, ask, said.append)
    assert got == {"bottle": "empty", "lid": "on"}
    assert asked == ["Bottle filled? [filled/empty]: "] * 3
    assert said[0] == "--condition botle: no block here uses it; ignored (did you mean bottle?)"
    assert said[1:] == ["bottle: give a value", "bottle: 'full' is not one of filled, empty"]
    assert prompter.ask_conditions(q, {"bottle": "filled", "lid": "off"}, False) == {
        "bottle": "filled", "lid": "off"}  # fmt: skip
    with pytest.raises(SystemExit, match="bottle: 'full' is not one of filled, empty"):
        prompter.ask_conditions(q, {"bottle": "full", "lid": "on"}, False)
    with pytest.raises(SystemExit, match="no terminal to ask: .* for lid"):
        prompter.ask_conditions(q, {"bottle": "filled"}, False)
    assert prompter.ask_conditions({}, {"bottle": "filled"}, False, say=said.append) == {}


def test_a_given_condition_no_block_uses_is_noted():
    """Ignored with a note, which names a close condition only when it looks like a typo of
    one: not for the other bottle of a pair given together, nor for a merely similar name."""
    filled = {"ask": "Bottle filled?", "choices": ["filled", "empty"]}
    q = {"bottle_fill_rubber": filled}

    def notes(given: dict[str, str]) -> list[str]:
        said: list[str] = []
        prompter.ask_conditions(q, given, False, say=said.append)
        return said

    note = "--condition {}: no block here uses it; ignored"
    both = {"bottle_fill_rubber": "filled", "bottle_fill_metal": "empty"}
    assert notes(both) == [note.format("bottle_fill_metal")]
    typo = {"bottle_fill_rubber": "filled", "bottle_fill_rubbr": "empty"}
    assert notes(typo) == [note.format("bottle_fill_rubbr")]  # the close one was given too
    alone = {"bottle_fill_rubbr": "filled"}
    with pytest.raises(SystemExit):  # the condition itself is still unanswered
        notes(alone)
    said: list[str] = []
    prompter.ask_conditions(q, alone, True, lambda _: "filled", said.append)
    assert said == [note.format("bottle_fill_rubbr") + " (did you mean bottle_fill_rubber?)"]
    said.clear()
    prompter.ask_conditions(q, {"bottle_fill_metal": "x"}, True, lambda _: "filled", said.append)
    assert said == [note.format("bottle_fill_metal")]  # similar (0.74), but not a typo


class RunOneBlock:
    """Stands in for the QApplication of ``prompter.main``: ``exec`` notes the window's blocks,
    runs its first block for a moment, ends it early and closes the window."""

    def __init__(self):
        self.blocks: list[list[str]] = []

    def exec(self):
        from pyqtgraph.Qt import QtWidgets

        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, TaskWindow) and w.isVisible():
                self.blocks.append(_names(w.sequences))
                w.start_block()
                time.sleep(0.3)
                w.end_block(completed=False)
                w.close()
        return 0


def _main(monkeypatch, argv: list[str]) -> RunOneBlock:
    pg.mkQApp()
    app = RunOneBlock()
    monkeypatch.setattr(pg, "mkQApp", lambda *a, **k: app)
    prompter.main(["--no-stream", *argv])
    return app


@pytest.fixture(scope="module")
def task_set(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("set") / "G01"
    make_task_sequences.main(["--catalogue", str(CATALOGUE), "--out", str(out), "--seed", "3"])
    return out


def test_a_generated_session_runs_simulated(task_set, tmp_path, monkeypatch, capsys):
    s01 = task_set / "S01"
    app = _main(monkeypatch, ["--simulate", "--data-dir", str(tmp_path), "--session", "x",
                              *BOTTLES, str(s01)])  # fmt: skip
    names = [json.loads(f.read_text())["name"] for f in sorted(s01.glob("*.json"))]
    assert app.blocks == [[GRASP_CALIBRATION, *names]]
    meta = json.loads((tmp_path / "x" / "session.json").read_text())["metadata"]
    assert meta["conditions"] == {"bottle_fill_rubber": "filled"}  # S01 has one bottle only
    out = capsys.readouterr().out.splitlines()
    assert "--condition bottle_fill_metal: no block here uses it; ignored" in out  # no hint
    (rec,) = read_records(tmp_path / "x")
    assert (
        rec["metadata"]["task_set"] == GRASP_CALIBRATION and rec["metadata"]["completed"] is False
    )


class RunBottle:
    """Stands in for the QApplication of ``prompter.main``: ``exec`` runs the first block that
    uses a session condition for a moment, ends it early and closes the window."""

    def exec(self):
        from pyqtgraph.Qt import QtWidgets

        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, TaskWindow) and w.isVisible():
                w.current = next(
                    i for i, (_, b) in enumerate(w.sequences) if b.params.get("conditions")
                )
                w.start_block()
                time.sleep(0.2)
                w.end_block(completed=False)
                w.close()
        return 0


def test_a_generated_block_with_a_condition_records_it(task_set, tmp_path, monkeypatch):
    """A generated text block that uses a session condition, run through the command: the
    session's answer is in the session metadata and in the block's trial."""
    pg.mkQApp()
    monkeypatch.setattr(pg, "mkQApp", lambda *a, **k: RunBottle())
    prompter.main(["--no-stream", "--simulate", "--no-calibration", "--data-dir", str(tmp_path),
                   "--session", "x", *BOTTLES, str(task_set / "S01")])  # fmt: skip
    meta = json.loads((tmp_path / "x" / "session.json").read_text())["metadata"]
    assert meta["conditions"] == {"bottle_fill_rubber": "filled"}
    (rec,) = read_records(tmp_path / "x")
    assert rec["label"] == "S01_b02_water_bottle_rubber"
    assert rec["metadata"]["conditions"] == {"bottle_fill_rubber": "filled"}


def test_a_folder_with_its_plan_loads_and_lane_blocks_mix_in(tmp_path, monkeypatch):
    folder = tmp_path / "blocks"
    folder.mkdir()
    (folder / "plan.json").write_text(json.dumps({"seed": 1, "sessions": []}))
    _lane("a_lane").save(folder / "a.json")
    _text("b_text").save(folder / "b.json")
    assert [p.name for p in prompter.expand([folder])] == ["a.json", "b.json"]
    app = _main(monkeypatch, ["--simulate", "--effort-meter", "--data-dir", str(tmp_path),
                              "--session", "x", str(folder)])  # fmt: skip
    cal = calibration_for("single_pinch").name
    assert app.blocks == [[cal, "a_lane", GRASP_CALIBRATION, "b_text"]]
    meta = json.loads((tmp_path / "x" / "session.json").read_text())["metadata"]
    assert "conditions" not in meta  # no block uses one


def test_an_unanswered_condition_stops_before_the_session_opens(task_set, tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = ["--simulate", "--data-dir", str(tmp_path), str(task_set / "S01")]
    with pytest.raises(SystemExit, match="no terminal to ask: pass --condition"):
        _main(monkeypatch, run)
    with pytest.raises(SystemExit, match="'half' is not one of filled, empty"):
        _main(monkeypatch, ["--condition", "bottle_fill_rubber=half", *run])
    with pytest.raises(SystemExit, match="bottle_fill_metal: given twice"):
        _main(monkeypatch, [*BOTTLES, "--condition", "bottle_fill_metal=half", *run])
    assert not any(tmp_path.iterdir())  # no session folder


def test_a_timed_block_file_stops_before_the_session_opens(
    timed_calibration, tmp_path, monkeypatch
):
    path = tmp_path / "blocks" / "old_calibration.json"
    path.parent.mkdir()
    timed_calibration.save(path)
    data = tmp_path / "data"
    with pytest.raises(SystemExit, match=r"old_calibration\.json: a timed block can only be "
                       r"replayed, not run"):  # fmt: skip
        _main(monkeypatch, ["--simulate", "--data-dir", str(data), str(path)])
    assert not data.exists()  # no session folder


def test_a_recording_with_a_text_block_replays_as_the_armband(tmp_path, monkeypatch):
    path = tmp_path / "t.json"
    _text("t").save(path)
    _main(monkeypatch, ["--simulate", "--no-calibration", "--data-dir", str(tmp_path),
                        "--session", "rec", str(path)])  # fmt: skip
    recorded = str(tmp_path / "rec")
    app = _main(monkeypatch, ["--replay", recorded, "--no-calibration", "--data-dir",
                              str(tmp_path), "--session", "again", str(path)])  # fmt: skip
    assert app.blocks == [["t"]]
    meta = json.loads((tmp_path / "again" / "session.json").read_text())["metadata"]
    assert meta["replay"] == [recorded]
    (rec,) = read_records(tmp_path / "again")
    assert rec["metadata"]["task"] == "text_prompter" and rec["n_emg"] > 0
