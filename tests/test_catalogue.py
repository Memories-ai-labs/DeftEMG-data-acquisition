"""The grasp and natural-task catalogue: the shipped file's combinations, and its checks."""

import copy
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from deftemg_acquisition.tasks.catalogue import (
    CatalogueError,
    catalogue_sha256,
    grasp_trials,
    load_catalogue,
    natural_units,
    parse_catalogue,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "protocols" / "grasp_natural" / "catalogue.toml"


@pytest.fixture(scope="module")
def cat():
    return load_catalogue(SHIPPED)


@pytest.fixture
def data():
    """The shipped file's contents, to edit into broken catalogues."""
    return copy.deepcopy(tomllib.loads(SHIPPED.read_text()))


def _obj(data, oid):
    return next(o for o in data["objects"] if o["id"] == oid)


def test_shipped_catalogue_counts(cat):
    trials = grasp_trials(cat)
    assert len(trials) == 456
    assert len(set((t.object, t.approach, t.grasp, t.variation) for t in trials)) == 456
    per_object = Counter(t.object for t in trials)
    expected = {
        "water_bottle_rubber": 32,
        "water_bottle_metal": 24,
        "tuna_can": 24,
        "hammer": 8,
        "potato_chip": 16,
        "chip_bag": 16,
        "book": 32,
        "mug_small": 16,
        "mug_large": 16,
    }
    expected |= dict.fromkeys(["baseball", "tennis_ball", "lacrosse_ball"], 16)
    expected |= dict.fromkeys(
        ["dish_soap", "windex", "soup_can", "peanut_butter_jar", "apple", "orange", "cracker_box"],
        32,
    )
    assert per_object == expected
    # every variation for every object
    assert all(n == 4 for n in Counter(t[:3] for t in _keys(trials)).values())

    steps = [s for u in natural_units(cat) for s in u.steps]
    assert len(steps) == 31
    assert Counter(s.group for s in steps) == {
        "pages": 1,
        "containers": 5,
        "screw_panel": 16,
        "usb": 6,
        "grip": 3,
    }
    assert Counter(s.task for s in steps) == {
        "pages": 1,
        "aquaphor": 1,
        "cream_jar": 2,
        "mug_drink": 2,
        "screw_panel": 16,
        "usb": 6,
        "grip": 3,
    }
    assert cat.sha256 == catalogue_sha256(SHIPPED) and len(cat.sha256) == 64


def _keys(trials):
    return [(t.object, t.approach, t.grasp, t.variation) for t in trials]


def _grasps(cat, oid, approach=None):
    return {t.grasp for t in grasp_trials(cat) if t.object == oid and t.approach == approach}


def test_shipped_catalogue_exclusions(cat):
    assert _grasps(cat, "tuna_can", "side") == {"TI", "TIM"}
    assert _grasps(cat, "tuna_can", "top") == {"TI", "TIM", "AD", "Power"}
    assert _grasps(cat, "water_bottle_metal", "side") == {"TIM", "AD", "Power"}
    assert _grasps(cat, "water_bottle_metal", "top") == {"TIM", "AD", "Power"}
    assert (
        _grasps(cat, "potato_chip", "side") == _grasps(cat, "potato_chip", "top") == {"TI", "TIM"}
    )
    assert _grasps(cat, "hammer") == {"AD", "Power"}
    assert {t.approach for t in grasp_trials(cat) if t.object == "book"} == {"edge", "flat"}
    assert {t.approach for t in grasp_trials(cat) if t.object == "baseball"} == {None}
    # each bottle carries its own condition, and nothing else has one
    conditions = {t.object: t.condition for t in grasp_trials(cat) if t.condition}
    assert conditions == {
        "water_bottle_rubber": "bottle_fill_rubber",
        "water_bottle_metal": "bottle_fill_metal",
    }
    assert set(cat.conditions) == {"bottle_fill_rubber", "bottle_fill_metal"}


def test_shipped_catalogue_text(cat):
    trials = {(t.object, t.approach, t.grasp, t.variation): t for t in grasp_trials(cat)}
    mug = trials["mug_small", None, "TI", "light"]
    assert mug.note == "on the lip" and mug.est_s == 4
    assert mug.text == "Small coffee mug\nThumb-index grasp, on the lip\nLight grasp"
    assert trials["mug_large", None, "TIM", "squeeze"].note == "on the lip"
    assert trials["mug_large", None, "AD", "squeeze"].note == ""
    assert trials["windex", "top", "Power", "pnp"].note == "near the top of the stem"
    book = trials["book", "edge", "Power", "pnp_sq"]
    assert book.text.splitlines() == [
        "Book, off the edge of the table",
        "Power grasp",
        "Pick up, squeeze while moving, place about 30 cm away",
    ]
    assert book.est_s == 9
    assert cat.grasps["Power"].highlight == "hand"
    assert cat.grasps["TIM"].highlight == ("thumb", "index", "middle")


def test_natural_pairs_stay_in_order(cat):
    units = {(u.task, u.variant): u for u in natural_units(cat)}
    assert len(units) == 1 + 1 + 1 + 2 + 8 + 3 + 3
    for (task, _), u in units.items():
        expected = {
            "screw_panel": ["loosen", "tighten"],
            "usb": ["insert", "remove"],
            "cream_jar": ["open", "close"],
        }.get(task, [None])
        assert [s.step for s in u.steps] == expected
    wing = units["screw_panel", "wing_nut"]
    assert [s.text for s in wing.steps] == [
        "Loosen the wing nut all the way off",
        "Tighten the wing nut",
    ]
    ratchet = units["screw_panel", "hex_ratchet"]
    assert ratchet.steps[0].text == "Loosen the hex bolt all the way off with the ratchet"
    assert units["mug_drink", "large"].steps[0].text == "Use the handle to drink from the large mug"
    pages = units["pages", None].steps[0]
    assert pages.duration_s == 20 and pages.group == "pages"
    assert all(s.duration_s is None for u in units.values() for s in u.steps if s.task != "pages")


def test_load_errors_name_the_file(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("[catalogue\n")
    with pytest.raises(CatalogueError, match="bad.toml"):
        load_catalogue(path)
    path.write_text(SHIPPED.read_text().replace('id = "hammer"', 'id = "baseball"'))
    with pytest.raises(CatalogueError, match=r"bad\.toml: objects: duplicate id 'baseball'"):
        load_catalogue(path)


def test_unknown_grasp(data):
    _obj(data, "hammer")["grasps"] = ["AD", "Fist"]
    with pytest.raises(CatalogueError, match=r"objects 'hammer'.grasps: unknown grasp type 'Fist'"):
        parse_catalogue(data)


def test_unknown_grasp_in_approach_grasps_and_notes(data):
    bad = copy.deepcopy(data)
    _obj(bad, "tuna_can")["approach_grasps"]["side"] = ["TI", "XX"]
    with pytest.raises(CatalogueError, match=r"'tuna_can'.approach_grasps.side: unknown grasp"):
        parse_catalogue(bad)
    _obj(data, "hammer")["notes"] = {"TI": "on the handle"}  # a grasp the hammer doesn't use
    with pytest.raises(CatalogueError, match=r"'hammer'.notes: unknown grasp type of this object"):
        parse_catalogue(data)


def test_unknown_condition(data):
    _obj(data, "water_bottle_metal")["condition"] = "bottle_fill"
    with pytest.raises(CatalogueError, match=r"'water_bottle_metal'.condition: unknown condition"):
        parse_catalogue(data)


def test_unknown_approach_and_key(data):
    bad = copy.deepcopy(data)
    _obj(bad, "apple")["approaches"] = ["side", "below"]
    with pytest.raises(CatalogueError, match=r"'apple'.approaches: unknown approach 'below'"):
        parse_catalogue(bad)
    _obj(data, "apple")["exclude_grasp"] = ["TI"]  # a misspelt key is not ignored
    with pytest.raises(CatalogueError, match=r"objects\[\d+\]: unknown key\(s\) exclude_grasp"):
        parse_catalogue(data)


@pytest.mark.parametrize("section", ["objects", "variations", "natural"])
def test_duplicate_ids(data, section):
    data[section].append(copy.deepcopy(data[section][0]))
    with pytest.raises(CatalogueError, match=f"{section}: duplicate id"):
        parse_catalogue(data)


def test_duplicate_variant(data):
    task = next(t for t in data["natural"] if t["id"] == "usb")
    task["variants"].append({"id": "usb_a", "name": "USB-A plug again"})
    with pytest.raises(CatalogueError, match=r"natural 'usb'.variants: duplicate id 'usb_a'"):
        parse_catalogue(data)


def test_empty_object(data):
    bad = copy.deepcopy(data)
    _obj(bad, "potato_chip")["exclude_grasps"] = ["TI", "TIM"]
    with pytest.raises(CatalogueError, match=r"objects 'potato_chip': no combinations"):
        parse_catalogue(bad)
    _obj(data, "tuna_can")["approach_grasps"]["side"] = []
    with pytest.raises(CatalogueError, match=r"'tuna_can': no combinations from approach 'side'"):
        parse_catalogue(data)


def test_natural_errors(data):
    bad = copy.deepcopy(data)
    grip = next(t for t in bad["natural"] if t["id"] == "grip")
    grip["text"] = "Squeeze the grip-strength tool once"  # every variant would read the same
    with pytest.raises(
        CatalogueError, match=r"natural 'grip': variants must differ in their prompt text"
    ):
        parse_catalogue(bad)
    bad = copy.deepcopy(data)
    next(t for t in bad["natural"] if t["id"] == "aquaphor")["text"] = "Open the {variant} cap"
    with pytest.raises(CatalogueError, match=r"'aquaphor': a text uses \{variant\}"):
        parse_catalogue(bad)
    screws = next(t for t in data["natural"] if t["id"] == "screw_panel")
    screws["variants"][0]["text"] = {"unscrew": "Unscrew it"}
    with pytest.raises(CatalogueError, match=r"text: unknown step 'unscrew'"):
        parse_catalogue(data)


def test_grasp_type_needs_tips_or_whole_hand(data):
    bad = copy.deepcopy(data)
    bad["grasps"]["TI"]["tips"] = ["thumb", "indx"]
    with pytest.raises(CatalogueError, match=r"grasps.TI.tips: unknown digit 'indx'"):
        parse_catalogue(bad)
    data["grasps"]["Power"]["tips"] = ["thumb"]
    with pytest.raises(CatalogueError, match=r"grasps.Power: give either tips or whole_hand"):
        parse_catalogue(data)


def test_timing_defaults_and_overrides(data):
    del data["timing"]
    assert parse_catalogue(data).timing.session_limit_s == 1800
    data["timing"] = {"reset_s": 7}
    timing = parse_catalogue(data).timing
    assert timing.reset_s == 7 and timing.countdown_s == 3
    data["timing"] = {"reset_s": -1}
    with pytest.raises(CatalogueError, match="timing.reset_s: expected a positive number"):
        parse_catalogue(data)


def test_one_step_variant_wording(data):
    grip = next(t for t in data["natural"] if t["id"] == "grip")
    grip["variants"][2]["text"] = "Squeeze the heavy grip-strength tool once, with both hands"
    units = {(u.task, u.variant): u for u in natural_units(parse_catalogue(data))}
    assert units["grip", "heavy"].steps[0].text.endswith("with both hands")
    assert units["grip", "light"].steps[0].text == "Squeeze the light grip-strength tool once"
    grip["variants"][2]["text"] = {"squeeze": "Squeeze it"}  # no step ids in a one-step task
    with pytest.raises(CatalogueError, match=r"'grip'.variants 'heavy'.text: a one-step task"):
        parse_catalogue(data)


def test_condition_must_be_a_name(data):
    _obj(data, "water_bottle_metal")["condition"] = ["bottle_fill_metal"]
    with pytest.raises(CatalogueError, match=r"'water_bottle_metal'.condition: ids are"):
        parse_catalogue(data)


def test_variants_must_be_a_list(data):
    usb = next(t for t in data["natural"] if t["id"] == "usb")
    usb["variants"] = {"id": "usb_c", "name": "USB-C plug"}
    with pytest.raises(CatalogueError, match=r"natural 'usb'.variants: expected a list"):
        parse_catalogue(data)


def test_fixed_duration_is_the_default_estimate(cat, data):
    pages = next(u for u in natural_units(cat) if u.task == "pages").steps[0]
    assert pages.est_s == pages.duration_s == 20
    task = next(t for t in data["natural"] if t["id"] == "pages")
    task["est_s"] = 25
    pages = next(u for u in natural_units(parse_catalogue(data)) if u.task == "pages").steps[0]
    assert pages.est_s == 25 and pages.duration_s == 20
    del task["est_s"], task["fixed_s"]
    with pytest.raises(CatalogueError, match=r"natural 'pages': missing est_s"):
        parse_catalogue(data)


def test_file_that_is_not_utf8(tmp_path):
    path = tmp_path / "latin1.toml"
    path.write_bytes(SHIPPED.read_bytes().replace(b"Baseball", b"Baseb\xe4ll"))
    with pytest.raises(CatalogueError, match=r"latin1\.toml: .*utf-8"):
        load_catalogue(path)
