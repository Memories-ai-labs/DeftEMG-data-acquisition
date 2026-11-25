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
    assert len(trials) == 590
    assert len(set((t.object, t.approach, t.grasp, t.variation) for t in trials)) == 590
    per_object = Counter(t.object for t in trials)
    # the four variations on every (approach, grasp type), plus 2 rotates per grasp type
    expected = {
        "water_bottle_rubber": 32 + 8,
        "water_bottle_metal": 24 + 6,
        "tuna_can": 24 + 8,
        "hammer": 8,  # no rotate
        "potato_chip": 16,  # no rotate
        "chip_bag": 16 + 8,
        "book": 32 + 8,
        "mug_small": 16 + 8,
        "mug_large": 16 + 8,
    }
    expected |= dict.fromkeys(["baseball", "tennis_ball", "lacrosse_ball"], 16 + 8)
    expected |= dict.fromkeys(
        ["dish_soap", "windex", "soup_can", "peanut_butter_jar", "apple", "orange", "cracker_box"],
        32 + 8,
    )
    assert per_object == expected
    # the four other variations on every (object, approach, grasp type)
    others = [k for k in _keys(trials) if not k[3].startswith("rotate")]
    assert all(n == 4 for n in Counter(k[:3] for k in others).values())

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
    mug = trials["mug_small", "top", "TI", "light"]
    assert mug.note == "on the lip" and mug.est_s == 4
    assert mug.text == (
        "Small coffee mug, from the top\nThumb-index grasp, on the lip\nSTAY ON TABLE\nLight grasp"
    )
    assert trials["mug_large", "top", "TIM", "squeeze"].note == "on the lip"
    assert trials["mug_large", "side", "AD", "squeeze"].note == "around the body"
    assert trials["windex", "top", "Power", "pnp"].note == "near the top of the stem"
    book = trials["book", "edge", "Power", "pnp_sq"]
    assert book.text.splitlines() == [
        "Book, off the edge of the table",
        "Power grasp",
        "PICK UP",
        "Pick up, squeeze while moving, place about 30 cm away",
    ]
    assert book.est_s == 9 and book.bold_line == 2
    assert cat.grasps["Power"].highlight == "hand"
    assert cat.grasps["TIM"].highlight == ("thumb", "index", "middle")


def test_every_grasp_prompt_has_its_lead_at_line_2(cat):
    leads = {
        "light": "STAY ON TABLE",
        "squeeze": "STAY ON TABLE",
        "pnp": "PICK UP",
        "pnp_sq": "PICK UP",
        "rotate_slow": "PICK UP",
        "rotate_fast": "PICK UP",
    }
    assert {v.id: v.lead for v in cat.variations.values()} == leads
    for t in grasp_trials(cat):
        lines = t.text.splitlines()
        assert t.bold_line == 2 and lines[2] == leads[t.variation], t
        assert lines[3] == cat.variations[t.variation].text


def test_rotate_on_the_first_approach_offering_each_grasp_type(cat):
    trials = grasp_trials(cat)
    for v in ["rotate_slow", "rotate_fast"]:
        rotate = [t for t in trials if t.variation == v]
        pairs = {(t.object, t.grasp) for t in rotate}
        assert len(rotate) == len(pairs) == 67  # once per (object, grasp type)
        every = {(t.object, t.grasp) for t in trials}
        assert pairs == {p for p in every if p[0] not in ("hammer", "potato_chip")}
        assert len({o for o, _ in pairs}) == 17
        at = {(t.object, t.grasp): t.approach for t in rotate}
        # the tuna can's side takes only TI and TIM, so AD and Power rotate from the top
        assert {g: at["tuna_can", g] for g in cat.grasps} == {
            "TI": "side", "TIM": "side", "AD": "top", "Power": "top"}  # fmt: skip
        assert {at["dish_soap", g] for g in cat.grasps} == {"side"}  # first listed approach
        assert {at["book", g] for g in cat.grasps} == {"edge"}
        assert {at["baseball", g] for g in cat.grasps} == {None}
    assert cat.variations["rotate_slow"].est_s == 12 and cat.variations["rotate_fast"].est_s == 8
    assert not cat.variations["pnp"].first_approach_only


def test_mugs_lip_from_the_top_body_from_the_side(cat):
    for mug in ["mug_small", "mug_large"]:
        ts = [t for t in grasp_trials(cat) if t.object == mug]
        assert {t.approach for t in ts} == {"top", "side"}
        for t in ts:
            if t.approach == "top":
                assert t.grasp in ("TI", "TIM") and t.note == "on the lip"
            else:
                assert t.grasp in ("AD", "Power") and t.note == "around the body"
            assert t.text.splitlines()[1].endswith(", " + t.note)


def test_shipped_terms_and_items(cat):
    terms = {t.term: t.text for t in cat.terms}
    assert terms["Light grasp"] == "touch the object lightly"
    assert terms["Grasp and squeeze"] == "touch the object and squeeze at a sub-maximum force"
    assert [t.term for t in cat.terms][:2] == ["Light grasp", "Grasp and squeeze"]  # file order
    items = {t.id: t.items for t in cat.natural}
    assert items["mug_drink"] == ("Small coffee mug", "Large coffee mug")
    assert all(items.values())


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


def _variation(data, vid):
    return next(v for v in data["variations"] if v["id"] == vid)


def test_first_approach_only_keeps_every_approach_with_a_prompt(data):
    """Variations only on the first approach would leave a later approach empty: an error."""
    for v in data["variations"]:
        v["first_approach_only"] = True
    with pytest.raises(CatalogueError, match=r"'water_bottle_rubber': no combinations from "):
        parse_catalogue(data)


def test_terms_are_optional(data):
    del data["terms"]
    assert parse_catalogue(data).terms == []


def _no_items(data):
    del data["natural"][0]["items"]


def _empty_items(data):
    data["natural"][0]["items"] = []


def _empty_item(data):
    data["natural"][0]["items"] = ["Book", " "]


def _items_not_a_list(data):
    data["natural"][0]["items"] = "Book"


def _repeated_term(data):
    data["terms"].append({"term": "Light grasp", "text": "again"})


@pytest.mark.parametrize(
    "edit, match",
    [
        (
            lambda d: _variation(d, "rotate_fast").update(first_approach_only="yes"),
            r"variations 'rotate_fast'.first_approach_only: expected true or false",
        ),
        (
            lambda d: _variation(d, "rotate_fast").update(first_approach_only=1),
            r"variations 'rotate_fast'.first_approach_only: expected true or false",
        ),
        (lambda d: _variation(d, "pnp").update(lead=""), r"variations 'pnp'.lead: expected a non"),
        (lambda d: _variation(d, "pnp").update(lead=["PICK UP"]), r"variations 'pnp'.lead"),
        (lambda d: _variation(d, "pnp").update(bold="PICK UP"), r"variations\[2\]: unknown key"),
        (_no_items, r"natural\[0\]: missing items"),
        (_empty_items, r"natural 'pages'.items: expected a non-empty list"),
        (_empty_item, r"natural 'pages'.items: expected a non-empty string"),
        (_items_not_a_list, r"natural 'pages'.items: expected a list of strings"),
        (_repeated_term, r"terms: 'Light grasp' is given twice"),
        (lambda d: d["terms"][0].update(term=""), r"terms\[0\].term: expected a non-empty"),
        (lambda d: d["terms"][0].update(text=" "), r"terms 'Light grasp'.text: expected a non"),
        (lambda d: d["terms"][0].pop("text"), r"terms\[0\]: missing text"),
        (lambda d: d["terms"][0].update(note="x"), r"terms\[0\]: unknown key\(s\) note"),
        (lambda d: d.update(terms={"term": "x", "text": "y"}), r"terms: expected a list"),
    ],
)
def test_new_key_errors(data, edit, match):
    edit(data)
    with pytest.raises(CatalogueError, match=match):
        parse_catalogue(data)
