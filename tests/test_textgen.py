"""Generating grasp and natural-task blocks: every combination once, blocks whole, sessions under
the limit, seeded, and the command's files and overwrite guard."""

import copy
import json
import shutil
import tomllib
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from deftemg_acquisition.cli import make_task_sequences as cli
from deftemg_acquisition.tasks.catalogue import (
    grasp_trials,
    load_catalogue,
    natural_units,
    parse_catalogue,
)
from deftemg_acquisition.tasks.textblock import TextBlock, load_block
from deftemg_acquisition.tasks.textgen import (
    block_estimate_s,
    calibrations_due,
    generate_task_blocks,
    session_capacity,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "protocols" / "grasp_natural" / "catalogue.toml"


@pytest.fixture(scope="module")
def cat():
    return load_catalogue(SHIPPED)


@pytest.fixture(scope="module")
def ts(cat):
    return generate_task_blocks(cat, seed=7)


def _grasp_key(p):
    return (p.object, p.approach, p.grasp, p.variation)


def _natural_key(p):
    return (p.object, p.variation, p.step)


def _of(ts, task_set):
    return [b for b in ts.blocks if b.task_set == task_set]


def test_every_combination_appears_once(cat, ts):
    grasp = Counter(_grasp_key(p) for b in _of(ts, "grasp") for p in b.prompts)
    expected = Counter((t.object, t.approach, t.grasp, t.variation) for t in grasp_trials(cat))
    assert sum(grasp.values()) == 456 and grasp == expected and max(grasp.values()) == 1
    natural = Counter(_natural_key(p) for b in _of(ts, "natural") for p in b.prompts)
    steps = [s for u in natural_units(cat) for s in u.steps]
    assert sum(natural.values()) == 31 and max(natural.values()) == 1
    assert natural == Counter((s.task, s.variant, s.step) for s in steps)


def test_prompt_fields_and_highlights(cat, ts):
    texts = {(t.object, t.approach, t.grasp, t.variation): t for t in grasp_trials(cat)}
    for b in _of(ts, "grasp"):
        for p in b.prompts:
            t = texts[_grasp_key(p)]
            assert p.text == t.text and p.est_s == t.est_s
            assert p.highlight == cat.grasps[p.grasp].highlight
    assert {p.highlight for b in _of(ts, "grasp") for p in b.prompts if p.grasp == "Power"} == {
        "hand"
    }
    tasks = {t.id for t in cat.natural}
    for b in _of(ts, "natural"):
        for p in b.prompts:
            assert p.object in tasks and p.highlight == () and p.grasp is None
            assert p.group == b.params["group"]
    pages = next(p for b in ts.blocks for p in b.prompts if p.object == "pages")
    assert pages.duration_s == 20


def test_one_block_per_object_and_group(cat, ts):
    grasp = _of(ts, "grasp")
    assert [len({p.object for p in b.prompts}) for b in grasp] == [1] * len(grasp)
    assert sorted(b.params["object"] for b in grasp) == sorted(o.id for o in cat.objects)
    sizes = Counter(t.object for t in grasp_trials(cat))
    for b in grasp:  # whole objects, 32-combination ones too
        assert len(b.prompts) == sizes[b.params["object"]]
    assert max(len(b.prompts) for b in grasp) == 32
    natural = _of(ts, "natural")
    assert sorted(b.params["group"] for b in natural) == sorted({t.group for t in cat.natural})
    for b in natural:
        assert {p.group for p in b.prompts} == {b.params["group"]}


def test_pairs_stay_consecutive_and_in_order(cat, ts):
    units = {(u.task, u.variant): [s.step for s in u.steps] for u in natural_units(cat)}
    for b in _of(ts, "natural"):
        i = 0
        while i < len(b.prompts):
            p = b.prompts[i]
            steps = units[(p.object, p.variation)]
            got = [(q.object, q.variation, q.step) for q in b.prompts[i : i + len(steps)]]
            assert got == [(p.object, p.variation, s) for s in steps]
            i += len(steps)


def test_natural_blocks_spread_over_sessions(ts):
    with_natural = [s for s in ts.sessions if any(b.task_set == "natural" for b in s.blocks)]
    assert len(with_natural) > 1
    assert all(sum(b.task_set == "natural" for b in s.blocks) == 1 for s in with_natural)


def test_natural_blocks_reach_the_last_session(cat):
    def last_has_natural(seed):
        last = generate_task_blocks(cat, seed=seed).sessions[-1]
        return any(b.task_set == "natural" for b in last.blocks)

    assert any(last_has_natural(seed) for seed in range(20))


def test_empty_catalogue_is_an_error():
    data = copy.deepcopy(tomllib.loads(SHIPPED.read_text()))
    data["objects"], data["natural"] = [], []
    with pytest.raises(ValueError, match="no objects or natural tasks"):
        generate_task_blocks(parse_catalogue(data), seed=1)


def test_sessions_fit_the_limit(cat, ts):
    limit = cat.timing.session_limit_s
    assert 4 <= len(ts.sessions) <= 8
    for s in ts.sessions:
        assert s.est_s <= limit
        work = sum(block_estimate_s(b, ts.timing) for b in s.blocks)
        assert work <= ts.capacity_s
        expect = cat.timing.session_setup_s + s.calibrations * cat.timing.calibration_s + work
        assert s.est_s == pytest.approx(expect)
    tight = generate_task_blocks(cat, replace(cat.timing, session_limit_s=1200), seed=7)
    assert len(tight.sessions) > len(ts.sessions)
    assert all(s.est_s <= 1200 for s in tight.sessions)


def test_estimate_rule(cat, ts):
    t = cat.timing
    b = next(b for b in ts.blocks if b.params.get("object") == "hammer")
    # 8 prompts: 2 grasp types x 4 variations (4 + 6 + 8 + 9 s), each with countdown and reset
    expect = t.object_swap_s + 2 * (4 + 6 + 8 + 9) + 8 * (t.countdown_s + t.reset_s)
    assert block_estimate_s(b, t) == pytest.approx(expect) == b.params["plan_est_s"]


def test_calibrations_counted_as_the_prompter_schedules_them(cat):
    t = cat.timing  # one at the start, then due 10 min after it, then every 20 min
    assert session_capacity(t) == t.session_limit_s - t.session_setup_s - 2 * t.calibration_s
    assert calibrations_due(300, t) == 1
    assert calibrations_due(600 - t.calibration_s + 1, t) == 2
    assert calibrations_due(1440, t) == 2
    assert calibrations_due(1800 - 2 * t.calibration_s + 1, t) == 3


def test_a_block_longer_than_a_session_is_an_error(cat):
    with pytest.raises(ValueError, match="screw_panel.*more than"):
        generate_task_blocks(cat, replace(cat.timing, session_limit_s=900), seed=1)
    with pytest.raises(ValueError, match="no time left"):
        generate_task_blocks(cat, replace(cat.timing, session_limit_s=250), seed=1)


def test_seeded(cat, ts):
    again = generate_task_blocks(cat, seed=7)
    assert [b.to_dict() for b in again.blocks] == [b.to_dict() for b in ts.blocks]
    other = generate_task_blocks(cat, seed=8)
    order = {b.params["object"]: [_grasp_key(p) for p in b.prompts] for b in _of(ts, "grasp")}
    other_order = {
        b.params["object"]: [_grasp_key(p) for p in b.prompts] for b in _of(other, "grasp")
    }
    assert order != other_order
    assert sum(order[k] != other_order[k] for k in order) > len(order) // 2
    a = generate_task_blocks(cat).seed  # random seed when not given
    assert isinstance(a, int)


def test_object_to_session_assignment_varies_with_seed(cat):
    def assignment(seed):
        return {
            b.params.get("object", b.params.get("group")): s.name
            for s in generate_task_blocks(cat, seed=seed).sessions
            for b in s.blocks
        }

    assert len({tuple(sorted(assignment(s).items())) for s in range(5)}) > 1


def test_block_params(cat, ts):
    for s in ts.sessions:
        for b in s.blocks:
            p = b.params
            assert p["seed"] == 7 and p["session"] == s.name
            assert p["catalogue_sha256"] == cat.sha256 and p["instructions"]
            assert p["timing"]["countdown_s"] == cat.timing.countdown_s
            assert p["timing"]["reset_s"] == cat.timing.reset_s
            assert p["plan_est_s"] == pytest.approx(block_estimate_s(b, ts.timing), abs=0.05)
            assert p["plan_est_s"] == pytest.approx(
                b.estimate_s() + ts.timing.object_swap_s, abs=0.05
            )
            assert "calibration" not in p  # the shipped file has no [calibration]
            assert b.name.startswith(f"{s.name}_b")
    conditions = {b.params.get("object"): b.params["conditions"] for b in ts.blocks}
    assert conditions["water_bottle_rubber"] == ["bottle_fill_rubber"]
    assert conditions["water_bottle_metal"] == ["bottle_fill_metal"]
    assert conditions["tuna_can"] == [] and conditions[None] == []
    questions = {b.params.get("object"): b.params.get("condition_questions") for b in ts.blocks}
    assert questions["water_bottle_rubber"] == {
        "bottle_fill_rubber": {
            "ask": cat.conditions["bottle_fill_rubber"].ask,
            "choices": ["filled", "empty"],
        }
    }
    assert questions["tuna_can"] is None and questions[None] is None  # only where one is used


def test_calibration_table_embedded():
    data = copy.deepcopy(tomllib.loads(SHIPPED.read_text()))
    data["calibration"] = {"hold_s": 2.0}
    cat = parse_catalogue(data, sha256="x")
    for b in generate_task_blocks(cat, seed=1).blocks:
        assert b.params["calibration"] == {"hold_s": 2.0}


def _run(*args):
    cli.main(["--catalogue", str(SHIPPED), *map(str, args)])


def test_command_writes_sessions_and_plan(tmp_path, ts):
    out = tmp_path / "G01"
    _run("--out", out, "--seed", 7)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["seed"] == 7 and plan["counts"] == {"grasp": 456, "natural": 31}
    assert plan["catalogue_path"] == "protocols/grasp_natural/catalogue.toml"  # repo-relative
    files = sorted(str(p.relative_to(out)) for p in out.glob("S??/*.json"))
    assert files == sorted(plan["files"])
    assert [s["session"] for s in plan["sessions"]] == [s.name for s in ts.sessions]
    for s, session in zip(plan["sessions"], ts.sessions, strict=True):
        assert s["est_s"] == pytest.approx(session.est_s, abs=0.05)
        names = [Path(b["file"]).name for b in s["blocks"]]
        assert names == sorted(names)  # the prompter runs a folder in name order
        for b, block in zip(s["blocks"], session.blocks, strict=True):
            loaded = load_block(out / b["file"])  # passes TextBlock's checks
            assert isinstance(loaded, TextBlock) and loaded.to_dict() == block.to_dict()
            assert loaded.params["seed"] == 7
    assert Path(plan["sessions"][0]["blocks"][0]["file"]).name.startswith("b01_")


def test_guard_refuses_without_force_and_force_removes_only_listed_files(tmp_path):
    out = tmp_path / "G01"
    _run("--out", out, "--seed", 1)
    before = json.loads((out / "plan.json").read_text())["files"]
    with pytest.raises(SystemExit, match="already holds a set"):
        _run("--out", out, "--seed", 2)
    keep = out / "notes.txt"
    keep.write_text("mine")
    (out / "S01" / "notes.txt").write_text("mine")
    _run("--out", out, "--seed", 2, "--force")
    after = json.loads((out / "plan.json").read_text())
    assert after["seed"] == 2 and keep.exists() and (out / "S01" / "notes.txt").exists()
    present = sorted(str(p.relative_to(out)) for p in out.glob("S??/*.json"))
    assert present == sorted(after["files"])  # no file of the old set is left over
    assert set(before) != set(after["files"])


def test_force_removes_session_folders_left_empty(tmp_path):
    out = tmp_path / "G01"
    _run("--out", out, "--seed", 1, "--session-limit-s", 1200)  # more, shorter sessions
    n_old = len(json.loads((out / "plan.json").read_text())["sessions"])
    _run("--out", out, "--seed", 1, "--force")
    n_new = len(json.loads((out / "plan.json").read_text())["sessions"])
    assert n_old > n_new
    assert sorted(p.name for p in out.glob("S??")) == [f"S{k:02d}" for k in range(1, n_new + 1)]


def test_guard_never_deletes_unlisted_blocks(tmp_path):
    out = tmp_path / "G01"
    _run("--out", out, "--seed", 1)
    stray = out / "S01" / "b99_extra.json"
    stray.write_text("{}")
    with pytest.raises(SystemExit, match="not listed"):
        _run("--out", out, "--seed", 2, "--force")
    assert stray.exists() and json.loads((out / "plan.json").read_text())["seed"] == 1
    (out / "plan.json").unlink()
    with pytest.raises(SystemExit, match="already holds a set"):
        _run("--out", out, "--seed", 2)
    with pytest.raises(SystemExit, match="no plan.json"):
        _run("--out", out, "--seed", 2, "--force")


def test_guard_rejects_a_plan_listing_other_paths(tmp_path):
    out = tmp_path / "G01"
    _run("--out", out, "--seed", 1)
    victim = tmp_path / "victim.json"
    victim.write_text("{}")
    plan = json.loads((out / "plan.json").read_text())
    plan["files"].append("../victim.json")
    (out / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(SystemExit, match="must list session block files"):
        _run("--out", out, "--seed", 2, "--force")
    assert victim.exists()


def test_plan_names_a_catalogue_outside_the_repo_by_file_name(tmp_path):
    shutil.copy(SHIPPED, tmp_path / "mine.toml")
    cli.main(["--catalogue", str(tmp_path / "mine.toml"), "--out", str(tmp_path / "G01")])
    assert json.loads((tmp_path / "G01" / "plan.json").read_text())["catalogue_path"] == "mine.toml"


def test_summary_writes_nothing(tmp_path, capsys):
    out = tmp_path / "G01"
    _run("--summary", "--out", out, "--seed", 7)
    assert not out.exists()
    text = capsys.readouterr().out
    assert "grasp combinations: 456" in text and "natural trials: 31" in text
    assert "S01" in text and "est" in text and "seed 7" in text
    _run("--summary")  # no --out needed
    assert " calibrations " in text and " 1 calibrations" not in text and "1 sessions" not in text


def test_command_options(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _run()  # no --out and no --summary
    with pytest.raises(SystemExit, match="more than"):
        _run("--summary", "--session-limit-s", 900)
    _run("--summary", "--session-limit-s", 1200, "--seed", 7)
    assert "S08" in capsys.readouterr().out


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def old_set(tmp_path):
    """A generated set in tmp_path/G01, and a folder outside it."""
    out, external = tmp_path / "G01", tmp_path / "external"
    _run("--out", out, "--seed", 1)
    external.mkdir()
    (external / "b01_keep.json").write_text("{}")
    return out, external


def _refused(out, external, match):
    before, outside = _snapshot(out), _snapshot(external)
    with pytest.raises(SystemExit, match=match):
        _run("--out", out, "--seed", 2, "--force")
    assert _snapshot(out) == before and _snapshot(external) == outside  # nothing deleted


def test_force_refuses_a_linked_session_folder(old_set):
    out, external = old_set
    shutil.rmtree(out / "S07")
    (out / "S07").symlink_to(external, target_is_directory=True)
    plan = json.loads((out / "plan.json").read_text())
    plan["files"].append("S07/b01_keep.json")
    (out / "plan.json").write_text(json.dumps(plan))
    _refused(out, external, "S07 is not a plain folder")


def test_force_refuses_a_linked_block_file(old_set):
    out, external = old_set
    plan = json.loads((out / "plan.json").read_text())
    block = out / plan["files"][0]
    block.unlink()
    block.symlink_to(external / "b01_keep.json")
    _refused(out, external, "is not a plain file")


def test_a_linked_out_folder_is_refused(old_set, tmp_path):
    out, external = old_set
    link = tmp_path / "link"
    link.symlink_to(out, target_is_directory=True)
    before = _snapshot(out)
    with pytest.raises(SystemExit, match="symbolic link"):
        _run("--out", link, "--seed", 2, "--force")
    assert _snapshot(out) == before


def test_a_failed_write_leaves_the_old_set(old_set, monkeypatch):
    out, _ = old_set
    before = _snapshot(out)
    calls = []

    def save(self, path):
        calls.append(path)
        if len(calls) > 3:
            raise OSError("disk full")
        real_save(self, path)

    real_save = TextBlock.save
    monkeypatch.setattr(TextBlock, "save", save)
    with pytest.raises(OSError, match="disk full"):
        _run("--out", out, "--seed", 2, "--force")
    assert _snapshot(out) == before
    assert sorted(p.name for p in out.iterdir() if p.name.startswith(".")) == []
