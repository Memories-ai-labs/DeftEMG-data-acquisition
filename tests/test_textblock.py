"""Text-prompt blocks: JSON round trip, loading lane files unchanged, the grasp calibration."""

import copy
import json
import tomllib
from pathlib import Path

import pytest

from deftemg_acquisition.tasks.catalogue import CatalogueError, parse_catalogue
from deftemg_acquisition.tasks.sequence import (
    CALIBRATION,
    THUMB_SETS,
    Sequence,
    calibration_for,
    calibration_sequence,
)
from deftemg_acquisition.tasks.textblock import (
    CALIBRATION_DEFAULTS,
    GRASP_CALIBRATION,
    TextBlock,
    TextPrompt,
    calibration_for_block,
    calibration_settings,
    grasp_calibration_block,
    load_block,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = sorted((ROOT / "protocols" / "demo").glob("*.json"))
SHIPPED = ROOT / "protocols" / "grasp_natural" / "catalogue.toml"


def _grasp_block(**params) -> TextBlock:
    prompts = [
        TextPrompt(
            "Tuna can, from the top\nThumb-index grasp\nLight grasp",
            "tuna_can",
            grasp="TI",
            approach="top",
            variation="light",
            highlight=("thumb", "index"),
            est_s=4.0,
        ),
        TextPrompt(
            "Tuna can, from the top\nPower grasp\nGrasp and squeeze",
            "tuna_can",
            grasp="Power",
            approach="top",
            variation="squeeze",
            highlight="hand",
            est_s=6.0,
        ),
    ]
    return TextBlock("b01_tuna_can", "grasp", "self_paced", prompts, {"seed": 7, **params})


def _natural_block() -> TextBlock:
    prompts = [
        TextPrompt(
            "Loosen the wing nut all the way off", "screw_panel", step="loosen",
            group="screw_panel", est_s=25.0,
        ),
        TextPrompt(
            "Tighten the wing nut", "screw_panel", step="tighten", group="screw_panel", est_s=25.0
        ),
        TextPrompt(
            "Open the book, then turn pages until the time is up", "pages", group="pages",
            duration_s=20.0,
        ),
    ]  # fmt: skip
    return TextBlock("b02_natural", "natural", "self_paced", prompts, {"timing": {"reset_s": 2}})


@pytest.mark.parametrize("make", [_grasp_block, _natural_block, grasp_calibration_block])
def test_text_block_roundtrip(tmp_path, make):
    block = make()
    path = tmp_path / "sub" / "block.json"
    block.save(path)
    d = json.loads(path.read_text())
    assert d["kind"] == "text_block" and d["task"] == "text_prompter"
    assert {"name", "task_set", "mode", "params", "prompts"} <= set(d)
    back = TextBlock.load(path)
    assert back == block
    assert load_block(path) == block
    assert isinstance(back.prompts[0].highlight, tuple | str)


def test_highlight_saved_as_list_or_hand(tmp_path):
    d = _grasp_block().to_dict()
    assert d["prompts"][0]["highlight"] == ["thumb", "index"]
    assert d["prompts"][1]["highlight"] == "hand"
    assert _natural_block().to_dict()["prompts"][0]["highlight"] == []


@pytest.mark.parametrize("path", DEMO, ids=lambda p: p.name)
def test_load_block_reads_lane_files_unchanged(path):
    assert DEMO, "no demo lane blocks"
    block = load_block(path)
    assert isinstance(block, Sequence)
    assert block == Sequence.load(path)
    assert block.to_dict() == Sequence.load(path).to_dict()
    assert not block.is_calibration


def test_load_block_rejects_unknown_kind(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({**_grasp_block().to_dict(), "kind": "something_else"}))
    with pytest.raises(ValueError, match="unknown block kind"):
        load_block(path)


def test_load_block_rejects_a_text_block_without_kind(tmp_path):
    """Not the lane path's KeyError: a clear error naming the missing kind."""
    d = _grasp_block().to_dict()
    del d["kind"]
    path = tmp_path / "x.json"
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match='needs "kind"'):
        load_block(path)


def test_is_calibration():
    assert calibration_sequence().is_calibration
    assert calibration_for("pinch").is_calibration
    assert grasp_calibration_block().is_calibration
    assert not _grasp_block().is_calibration and not _natural_block().is_calibration


def test_estimate_and_duration():
    block = _grasp_block()
    assert block.duration_s is None
    assert block.estimate_s() == pytest.approx((3 + 4 + 5) + (3 + 6 + 5))
    assert block.estimate_s(1) == pytest.approx(3 + 6 + 5)  # a resumed block: remaining only
    natural = _natural_block()  # reset_s from params; a fixed duration counts as the task
    assert natural.estimate_s() == pytest.approx((3 + 25 + 2) * 2 + (3 + 20 + 2))
    cal = grasp_calibration_block()
    assert cal.estimate_s() == cal.duration_s


def test_estimate_rejects_start_index_outside_the_block():
    block = _grasp_block()  # two prompts
    for bad in (2, -1, True, 1.0):
        with pytest.raises(ValueError, match="start_index"):
            block.estimate_s(bad)
    with pytest.raises(ValueError, match="timed block always starts at prompt 0"):
        grasp_calibration_block().estimate_s(1)


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d.update(task_set="calibration"), "task_set"),
        (lambda d: d.update(mode="paced"), "mode"),
        (lambda d: d.update(prompts=[]), "at least one prompt"),
        (lambda d: d["prompts"][0].update(highlight=["thumb", "toe"]), "unknown digit"),
        (lambda d: d["prompts"][0].update(highlight="palm"), "highlight"),
        (lambda d: d["prompts"][0].update(highlight=["index", "index"]), "repeated"),
        (lambda d: d["prompts"][0].update(text=" "), "text"),
        (lambda d: d["prompts"][0].update(colour="red"), "unknown key"),
        (lambda d: d["prompts"][0].update(onset_s=1.0), "self-paced"),
        (lambda d: d["prompts"][0].update(duration_s=0), "duration_s"),
        (lambda d: d.update(kind=None), "not a text block"),
        (lambda d: d.update(task="scroll_prompter"), "task is 'text_prompter'"),
        (lambda d: d.update(colour="red"), "unknown key"),
        (lambda d: d.pop("mode"), "missing mode"),
        (lambda d: d.update(prompts="none"), "list of tables"),
        (lambda d: d.update(name=""), "name"),
        (lambda d: d.update(task_set="grasp_calibration"), "must be 'timed'"),
        (lambda d: d["prompts"][0].update(grasp=3), "grasp must be a string or None"),
        (lambda d: d["prompts"][0].update(step=["loosen"]), "step must be a string or None"),
        (lambda d: d["prompts"][0].update(object=None), "object"),
        (lambda d: d["prompts"][0].update(est_s=True), "est_s"),
        (lambda d: d["prompts"][0].update(est_s="4"), "est_s"),
        (lambda d: d["prompts"][0].update(est_s=-1), "est_s"),
        (lambda d: d["prompts"][0].update(est_s=float("inf")), "est_s"),
        (lambda d: d["prompts"][0].update(duration_s=float("nan")), "duration_s"),
        (lambda d: d["prompts"][0].update(highlight=5), "highlight"),
        (lambda d: d["prompts"][0].update(highlight={"thumb": 1}), "highlight"),
        (lambda d: d["prompts"][0].update(highlight=[1, 2]), "highlight"),
        (lambda d: d.update(params=[]), "params must be a table"),
        (lambda d: d["params"].update(timing=[3]), "params.timing must be a table"),
        (lambda d: d["params"].update(timing={"reset_s": "5"}), "params.timing.reset_s"),
        (lambda d: d["params"].update(timing={"countdown_s": float("inf")}), "countdown_s"),
        (lambda d: d["params"].update(calibration={"hold": 2}), "params.calibration: unknown"),
        (lambda d: d["params"].update(calibration={"gap_s": -1}), "params.calibration.gap_s"),
    ],
)
def test_bad_text_blocks_raise(change, match):
    d = copy.deepcopy(_grasp_block().to_dict())
    change(d)
    with pytest.raises(ValueError, match=match):
        TextBlock.from_dict(d)


def test_timed_block_needs_times_in_order():
    d = grasp_calibration_block().to_dict()
    d["prompts"][1]["onset_s"] = None
    with pytest.raises(ValueError, match="onset_s and duration_s"):
        TextBlock.from_dict(d)
    d = grasp_calibration_block().to_dict()
    d["prompts"].reverse()
    with pytest.raises(ValueError, match="before prompt 0 ends"):
        TextBlock.from_dict(d)
    d = grasp_calibration_block().to_dict()
    d["prompts"][1]["onset_s"] = d["prompts"][0]["onset_s"] + 1.0  # inside the 1.5 s hold
    with pytest.raises(ValueError, match="not overlapping"):
        TextBlock.from_dict(d)
    d["prompts"][1]["onset_s"] = d["prompts"][0]["onset_s"] + 1.5  # right after it: allowed
    assert TextBlock.from_dict(d).prompts[1].onset_s == 5.5


def test_bad_calibration_in_a_block_file_fails_at_load(tmp_path):
    d = _grasp_block().to_dict()
    d["params"]["calibration"] = {"maxima": [{"grasp": "TI", "object": "ball"}]}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(d))
    with pytest.raises(
        ValueError, match=r"b01_tuna_can: params\.calibration\.maxima\[0\]: missing text"
    ):
        load_block(path)


def test_block_arrays_is_not_filled_in_yet():
    with pytest.raises(NotImplementedError):
        _grasp_block().block_arrays([])


# ---- the grasp calibration ----


def test_grasp_calibration_schedule():
    cal = grasp_calibration_block()
    assert cal.name == cal.task_set == GRASP_CALIBRATION
    assert cal.mode == "timed"
    assert [(p.grasp, p.object) for p in cal.prompts] == [
        ("TI", "hardcover_book"),
        ("TIM", "hardcover_book"),
        ("AD", "hardcover_book"),
        ("Power", "ball"),
    ]
    assert [p.onset_s for p in cal.prompts] == [4.0, 8.5, 13.0, 23.5]
    assert {p.duration_s for p in cal.prompts} == {1.5}
    assert cal.duration_s == 25.0  # about 26 s
    assert [p.highlight for p in cal.prompts] == [
        ("thumb", "index"),
        ("thumb", "index", "middle"),
        ("thumb", "index", "middle", "ring", "pinky"),
        "hand",
    ]
    assert cal.prompts[0].text.startswith("Hardcover book")
    assert cal.prompts[-1].text.startswith("Ball")  # the swap's preview names the ball
    assert "swap" in cal.params["instructions"]


def test_grasp_calibration_times_match_the_lane_calibration():
    """Same lead, rest, hold and gap as the lane calibration; only the swap adds time."""
    lane = calibration_sequence()
    cal = grasp_calibration_block()
    assert cal.prompts[0].onset_s == lane.prompts[0].onset_s
    assert cal.prompts[0].duration_s == lane.prompts[0].duration_s
    lane_step = lane.prompts[1].onset_s - lane.prompts[0].onset_s
    steps = [b.onset_s - a.onset_s for a, b in zip(cal.prompts, cal.prompts[1:], strict=False)]
    assert steps[:2] == [lane_step, lane_step]
    assert steps[2] == pytest.approx(lane_step + CALIBRATION_DEFAULTS["swap_gap_s"])


def test_grasp_calibration_reps_repeat_each_object_before_the_swap():
    cal = grasp_calibration_block(reps=2)
    assert [p.grasp for p in cal.prompts] == ["TI", "TIM", "AD"] * 2 + ["Power"] * 2
    # the second power maximum, right after the first, gets rest_s (3 s) more than the gap
    assert [p.onset_s for p in cal.prompts] == [4.0, 8.5, 13.0, 17.5, 22.0, 26.5, 37.0, 44.5]
    assert cal.duration_s == 46.0
    objects = [p.object for p in cal.prompts]
    assert sum(a != b for a, b in zip(objects, objects[1:], strict=False)) == 1  # one swap
    with pytest.raises(ValueError, match="reps"):
        grasp_calibration_block(reps=0)


def test_grasp_calibration_overrides():
    maxima = [
        {"grasp": "Power", "object": "ball", "text": "Ball\nPower grasp"},
        {"grasp": "TI", "object": "ball", "text": "Ball\nThumb-index grasp"},
    ]
    cal = grasp_calibration_block({"hold_s": 2.0, "gap_s": 4.0, "rest_s": 0, "maxima": maxima})
    assert [p.onset_s for p in cal.prompts] == [1.0, 7.0]  # one object: no swap gap
    assert cal.duration_s == 9.0
    assert cal.params["calibration"]["maxima"][1]["highlight"] == ["thumb", "index"]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"hold": 2.0}, "unknown key"),  # a misspelt hold_s is an error, not ignored
        ({"hold_s": 0}, "hold_s"),
        ({"gap_s": -1}, "gap_s"),
        ({"swap_gap_s": "6"}, "swap_gap_s"),
        ({"gap_s": float("inf")}, "gap_s"),
        ({"rest_s": float("nan")}, "rest_s"),
        ({"hold_s": True}, "hold_s"),
        ({"maxima": "TI"}, "maxima"),
        ({"maxima": ["TI"]}, "expected a table"),
        ({"maxima": []}, "maxima"),
        ({"maxima": [{"grasp": "TI", "object": "ball"}]}, "missing"),
        ({"maxima": [{"grasp": "TI", "object": "ball", "text": "x", "hold_s": 1}]}, "unknown"),
        ({"maxima": [{"grasp": "TI", "object": "ball", "text": 3}]}, "text"),
        ({"maxima": [{"grasp": "Lateral", "object": "key", "text": "x"}]}, "needs a highlight"),
    ],
)
def test_bad_calibration_settings_raise(overrides, match):
    with pytest.raises(ValueError, match=match):
        calibration_settings(overrides)


def test_calibration_for_block_matches_calibration_for_on_lane_blocks():
    lane_blocks = [Sequence.load(p) for p in DEMO] + [calibration_sequence()]
    task_sets = {b.task_set for b in lane_blocks}
    assert {"single", "multi", *THUMB_SETS, CALIBRATION} <= task_sets
    for block in lane_blocks:
        for reps in (1, 2):
            got = calibration_for_block(block, reps)
            assert isinstance(got, Sequence)
            assert got == calibration_for(block.task_set, reps)
    assert calibration_for_block(lane_blocks[0]) == calibration_for(lane_blocks[0].task_set)


def test_calibration_for_block_gives_the_grasp_calibration_for_text_blocks():
    assert calibration_for_block(_grasp_block()) == grasp_calibration_block()
    assert calibration_for_block(_natural_block(), 2) == grasp_calibration_block(reps=2)
    block = _grasp_block(calibration={"swap_gap_s": 10})  # embedded catalogue overrides
    cal = calibration_for_block(block)
    assert cal.is_calibration and cal.prompts[-1].onset_s == 27.5
    assert calibration_for_block(cal) == cal  # a calibration block keeps its settings


# ---- the catalogue's [calibration] table ----


@pytest.fixture
def data():
    return copy.deepcopy(tomllib.loads(SHIPPED.read_text()))


def test_catalogue_calibration_table_overrides_the_defaults(data):
    assert parse_catalogue(data).calibration == {}
    data["calibration"] = {
        "hold_s": 2,
        "swap_gap_s": 8,
        "maxima": [
            {"grasp": "AD", "object": "hardcover_book", "text": "Book\nAll fingertips"},
            {"grasp": "Power", "object": "ball", "text": "Ball\nPower grasp"},
        ],
    }
    cat = parse_catalogue(data)
    assert cat.calibration["maxima"][0]["highlight"] == list(cat.grasps["AD"].tips)
    assert cat.calibration["maxima"][1]["highlight"] == "hand"
    block = _grasp_block(calibration=cat.calibration)  # as the generator embeds it
    cal = calibration_for_block(block)
    assert [p.onset_s for p in cal.prompts] == [4.0, 4.0 + 2 + 3 + 8]
    assert {p.duration_s for p in cal.prompts} == {2.0}
    # embedded in a block file, the overrides survive the round trip
    assert calibration_for_block(TextBlock.from_dict(block.to_dict())) == cal


@pytest.mark.parametrize(
    "table, match",
    [
        ({"hold": 2}, "unknown key"),  # misspelt hold_s
        ({"hold_s": 0}, "hold_s"),
        ({"maxima": "TI"}, "maxima"),
        ({"maxima": [{"grasp": "Pinch", "object": "ball", "text": "x"}]}, "unknown grasp type"),
        ({"maxima": [{"grasp": "TI", "object": "a ball", "text": "x"}]}, "ids are"),
        ({"maxima": [{"grasp": "TI", "object": "ball"}]}, "missing text"),
        ({"maxima": [{"grasp": "TI", "object": "ball", "text": "x", "highlight": []}]}, "unknown"),
        ({"maxima": [{"grasp": ["TI"], "object": "ball", "text": "x"}]}, "grasp: ids are"),
        ({"maxima": [{"grasp": 3, "object": "ball", "text": "x"}]}, "grasp: ids are"),
        ({"maxima": ["TI"]}, "expected a table"),
        ({"maxima": []}, "maxima"),
        ({"gap_s": float("inf")}, "gap_s"),
    ],
)
def test_catalogue_calibration_table_errors(data, table, match):
    data["calibration"] = table
    with pytest.raises(CatalogueError, match=match):
        parse_catalogue(data)
