import dataclasses

import numpy as np
import pytest

from deftemg_acquisition.data.labels import (
    IGNORE,
    LabelSpec,
    digit_targets,
    label_span_ns,
    prompt_counts,
    rest_span_ns,
)
from deftemg_acquisition.data.trials import (
    Prompts,
    find_sessions,
    load_trial,
    scan_trials,
    select_trials,
)

S = 1_000_000_000  # ns per second
DIGITS = ("thumb", "index", "middle", "ring", "pinky")


def test_synthetic_sessions_load_like_recordings(synthetic_sessions, prompter_trial):
    infos = scan_trials(find_sessions([str(p) for p in synthetic_sessions]))
    assert len(infos) == 6 and all(i.is_prompter and not i.simulated for i in infos)
    trial = load_trial(infos[0].npz_path)
    assert trial.emg_uv.shape[1] == 18 and np.all(np.diff(trial.emg_t_ns) > 0)
    assert trial.prompts.reference == "paint" and trial.prompts.level_names == ("low", "high")
    assert trial.info.subject == "SYN" and trial.info.arm == "right"
    # The stall in session B splits its trials into two continuous segments.
    stalled = load_trial(scan_trials([synthetic_sessions[1]])[0].npz_path)
    assert len(stalled.segment_bounds()) == 2
    # A trial recorded by the real prompter (simulated armband) loads through the same code.
    rec = load_trial(sorted(prompter_trial.glob("trial_*.npz"))[0])
    assert rec.prompts.level_names == ("low", "high") and rec.info.simulated
    with np.load(rec.info.npz_path) as d_rec, np.load(trial.info.npz_path) as d_syn:
        needed = {k for k in d_rec.files if k.startswith(("emg_", "imu_", "prompt_"))}
        needed -= {"emg_offset", "imu_offset"}  # byte offsets into the .bin: not written
        assert needed <= set(d_syn.files)
        for k in needed:
            assert d_rec[k].dtype.kind == d_syn[k].dtype.kind, k
            assert d_rec[k].ndim == d_syn[k].ndim, k
        # IMU and EMG packets share one sequence counter without collisions.
        assert not set(d_syn["imu_seq"]) & set(d_syn["emg_seq"])
    assert np.all(np.diff(stalled.emg_t_ns) >= 0)
    assert stalled.info.record["n_dropped"] > 0


def test_select_trials_reports_reasons(
    synthetic_sessions, prompter_trial, keyboard_trial, legacy_trial
):
    sessions = [*synthetic_sessions, prompter_trial, keyboard_trial.parent, legacy_trial.parent]
    infos = scan_trials(sessions)
    usable, skipped = select_trials(infos)
    reasons = dict(skipped)
    assert len(usable) == 6 + 2  # the legacy copy keeps its two unmodified trials
    assert all(i.synthetic for i in usable)
    assert reasons[f"{prompter_trial.name}/trial_001_rep"] == "simulated armband"
    assert reasons[f"k/{keyboard_trial.stem}"] == "simulated armband"
    kb_reasons = dict(select_trials(infos, exclude_simulated=False, allow_mixed=True)[1])
    assert kb_reasons[f"k/{keyboard_trial.stem}"] == "not a prompter trial"
    assert "legacy" in reasons[f"L/{legacy_trial.stem}"]
    incomplete = dataclasses.replace(usable[0], completed=False)
    assert select_trials([incomplete])[1][0][1] == "incomplete block"
    assert select_trials([incomplete], include_incomplete=True)[0] == [incomplete]
    other = dataclasses.replace(usable[1], session={**usable[1].session, "filter_status": {}})
    with pytest.raises(ValueError, match="mix filter settings"):
        select_trials([usable[0], other])
    assert len(select_trials([usable[0], other], allow_mixed=True)[0]) == 2
    meta = {**usable[1].session, "metadata": {"arm": "left"}}
    with pytest.raises(ValueError, match="arms"):
        select_trials([usable[0], dataclasses.replace(usable[1], session=meta)])
    with pytest.raises(ValueError, match="recorded and synthetic"):
        select_trials([usable[0], dataclasses.replace(usable[1], synthetic=False)])
    three = dataclasses.replace(usable[0], level_names=("low", "medium", "high"))
    assert "expected ['low', 'high']" in select_trials([three])[1][0][1]


def test_replay_loads_keyboard_and_legacy_trials(keyboard_trial, legacy_trial):
    from deftemg_acquisition.viz.replay import load_trial as replay_load

    kb = replay_load(keyboard_trial)
    assert kb.prompts == [] and kb.sequence is None and kb.duration_s > 0.1
    legacy = replay_load(legacy_trial)
    raw = load_trial(legacy_trial)
    assert raw.prompts.reference == "schedule" and raw.prompts.legacy_levels
    assert raw.prompts.digit_names == DIGITS
    assert raw.prompts.level_names == ("low", "medium", "high")
    levels = [p.label.split()[-1] for p in legacy.prompts]
    assert levels and all(set(lv) <= set("LMH/") for lv in levels) and "H" in "".join(levels)


def make_prompts(rows, level_names=("low", "high")):
    """rows: (onset_s, offset_s, {digit: level_index})."""
    n = len(rows)
    digits = np.zeros((n, 5), bool)
    level = np.full((n, 5), -1)
    for i, (_, _, d) in enumerate(rows):
        for name, lv in d.items():
            digits[i, DIGITS.index(name)] = True
            level[i, DIGITS.index(name)] = lv
    on = np.array([r[0] for r in rows])
    off = np.array([r[1] for r in rows])
    return Prompts(
        onset_ns=(on * S).astype(np.int64),
        offset_ns=(off * S).astype(np.int64),
        digits=digits,
        level=level,
        digit_names=DIGITS,
        level_names=level_names,
        onset_s=on,
        offset_s=off,
        reference="paint",
        legacy_levels=False,
    )


def test_digit_targets_shift_delay_and_margin():
    prompts = make_prompts([(2.0, 3.0, {"index": 1}), (4.0, 4.5, {"thumb": 0, "pinky": 1})])
    spec = LabelSpec(onset_shift_s=-0.15, offset_shift_s=0.20, delay_s=0.10, margin_s=0.10)
    t = np.arange(0, 6 * S, S // 1000)  # 1 kHz
    y = digit_targets(t, prompts, spec)
    ts = t / S
    index = ts[y[:, 1] == 2]
    assert index.min() == pytest.approx(1.95) and index.max() == pytest.approx(3.299)
    assert set(np.unique(y[:, 0])) == {0, 1} and set(np.unique(y[:, 4])) == {0, 2}
    assert np.all(y[:, 2] == 0)  # middle never prompted
    ym = digit_targets(t, prompts, spec, margin=True)
    ignored = ts[ym[:, 1] == IGNORE]
    assert ignored.min() == pytest.approx(1.85) and ignored.max() == pytest.approx(3.399)
    assert np.all(ym[:, 2] == 0)  # margins only around a digit's own edges
    np.testing.assert_array_equal(ym[(ym != IGNORE)], y[(ym != IGNORE)])
    counts = prompt_counts(prompts, ("none", "low", "high"), DIGITS)
    assert counts.tolist() == [[1, 1, 0], [1, 0, 1], [2, 0, 0], [2, 0, 0], [1, 0, 1]]
    with pytest.raises(ValueError, match="not one of the classes"):
        digit_targets(t, make_prompts([(1, 2, {"ring": 1})], ("low", "medium")), spec)
    # Where two prompts for the same digit touch, the later one wins.
    touching = make_prompts([(1.0, 2.0, {"ring": 0}), (2.2, 3.0, {"ring": 1})])
    y2 = digit_targets(t, touching, spec)
    assert y2[np.searchsorted(t, int(2.2 * S)), 3] == 2


def test_rest_and_label_spans(synthetic_sessions):
    trial = load_trial(scan_trials([synthetic_sessions[0]])[0].npz_path)
    spec = LabelSpec()
    t0 = trial.emg_t_ns[0]
    first_label = trial.prompts.onset_ns.min() - round(0.05 * S)
    rest = rest_span_ns(trial, spec, skip_start_s=1.0, guard_s=0.3)
    assert rest == (t0 + S, first_label - round(0.3 * S))
    lo, hi = label_span_ns(trial, spec, skip_start_s=1.0, tail_s=1.0)
    assert lo == t0 + S and hi == trial.prompts.offset_ns.max() + round(1.3 * S)
    # Without a lead-in the rest span is empty (start >= end), not an error.
    no_lead = rest_span_ns(trial, spec, skip_start_s=10.0, guard_s=0.3)
    assert no_lead[0] >= no_lead[1]


def test_null_synthetic_sessions_carry_no_label_information(tmp_path):
    from deftemg_acquisition.data.labels import rest_samples
    from deftemg_acquisition.data.press_times import zscore_envelope
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    ratios = {}
    for null in (False, True):
        cfg = SyntheticConfig(n_blocks=6, per_block=8, null=null, seed=3, compress=False)
        d = write_session(tmp_path / str(null), cfg)
        r = []
        for info in scan_trials([d]):
            trial = load_trial(info.npz_path)
            y = digit_targets(trial.emg_t_ns, trial.prompts, LabelSpec())
            rest = rest_samples(trial, rest_span_ns(trial, LabelSpec(), 1.0, 0.3))
            env = zscore_envelope(trial.emg_uv, rest, 2000.0, 0.025)
            span = trial.emg_t_ns >= trial.prompts.onset_ns.min()  # after the rest lead-in
            active = (y > 0).any(axis=1)
            r.append((env[span & active].mean(), env[span & ~active].mean()))
        on, off = np.mean(r, axis=0)  # pooled: a mean of per-trial ratios is biased upwards
        ratios[null] = float(on / off)
    # Null: ~1 (over many seeds the same as for randomly shifted labels; trial-to-trial SD of
    # the ratio ~0.2, so 6 trials give ~0.08).
    assert ratios[False] > 1.4 and abs(ratios[True] - 1) < 0.2


def test_alignment_modes_and_nulls(tmp_path):
    from deftemg_acquisition.data.labels import Alignment, aligned_prompts
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    # EMG starts 250 ms before and ends 300 ms after each drawn prompt (+16 ms paint delay).
    cfg = SyntheticConfig(n_blocks=1, per_block=10, lead_s=(0.25, 0.25), lag_s=(0.3, 0.3),
                          strength=2.0, seed=7, compress=False)  # fmt: skip
    trial = load_trial(scan_trials([write_session(tmp_path / "a", cfg)])[0].npz_path)
    spec = LabelSpec()
    p_med, med, _ = aligned_prompts(trial, spec, Alignment("emg_trial_median"))
    assert med.onset_shift_s == 0 and med.delay_s == spec.delay_s  # times now in the prompts
    on_lag = (p_med.onset_ns - trial.prompts.onset_ns) / 1e9
    off_lag = (p_med.offset_ns - trial.prompts.offset_ns) / 1e9
    assert np.all((on_lag > -0.27) & (on_lag < -0.18)) and np.all(
        (off_lag > 0.22) & (off_lag < 0.34)
    )
    p_emg, s_emg, _ = aligned_prompts(trial, spec, Alignment("emg"))
    assert s_emg.onset_shift_s == 0 and s_emg.delay_s == spec.delay_s
    lag = (p_emg.onset_ns - trial.prompts.onset_ns) / 1e9
    assert np.all((lag > -0.3) & (lag < -0.15))
    _, _, ignore = aligned_prompts(trial, spec, Alignment(mask_undetected=True))
    assert ignore.sum() <= 1  # strong presses are detected
    # Nulls keep what is prompted but break its link to the EMG.
    perm, _, _ = aligned_prompts(trial, spec, Alignment(null="permute", seed=1))
    np.testing.assert_array_equal(perm.onset_ns, trial.prompts.onset_ns)
    assert sorted(map(tuple, perm.level.tolist())) == sorted(
        map(tuple, trial.prompts.level.tolist())
    )
    shift, _, _ = aligned_prompts(trial, spec, Alignment(null="shift", seed=1))
    moved = (shift.onset_ns - trial.prompts.onset_ns) % int(trial.emg_t_ns[-1] - trial.emg_t_ns[0])
    assert np.all(np.abs(np.diff(moved)) < 1e6) or len(set(np.round(moved / 1e9, 1))) <= 2
    assert not np.array_equal(shift.onset_ns, trial.prompts.onset_ns)


def test_measured_edges_fall_back_and_respect_censoring():
    from deftemg_acquisition.data.labels import Alignment, _measured_edges
    from deftemg_acquisition.data.press_times import PressTimes

    p = make_prompts([(2.0, 2.5, {"index": 0}), (4.0, 4.5, {"index": 0}), (6.0, 6.5, {"index": 0}),
                      (8.0, 8.5, {"index": 0})])  # fmt: skip
    ms = 1_000_000
    times = PressTimes(
        onset_ns=p.onset_ns - 150 * ms,
        offset_ns=p.offset_ns + np.array([200, 200, -800, 500]) * ms,  # #2 implausible
        detected=np.array([True, True, True, True]),
        onset_censored=np.zeros(4, bool),
        offset_censored=np.array([False, False, False, True]),  # #3: true offset even later
        peak=np.ones(4),
    )
    on, off = _measured_edges(p, times, LabelSpec(), Alignment("emg", min_detected=2))
    lag_off = (off - p.offset_ns) / ms
    assert lag_off[0] == 200 and lag_off[1] == 200
    assert lag_off[2] == pytest.approx(200)  # implausible -> the median of plausible ones
    assert lag_off[3] >= 500  # censored: never short of its bound


def test_short_gaps_bound_offsets_by_the_next_onset(tmp_path):
    from deftemg_acquisition.data.press_times import estimate_press_times
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    cfg = SyntheticConfig(n_blocks=1, per_block=12, gap_s=(0.4, 0.6), lag_s=(0.3, 0.3),
                          strength=2.0, seed=2, compress=False)  # fmt: skip
    trial = load_trial(scan_trials([write_session(tmp_path / "g", cfg)])[0].npz_path)
    times = estimate_press_times(trial, rest_span_ns(trial, LabelSpec(), 1.0, 0.3))
    order = np.argsort(trial.prompts.onset_ns)
    det = times.detected[order]
    for a, b in zip(order[:-1], order[1:], strict=True):
        if times.detected[a] and times.detected[b]:
            assert times.offset_ns[a] <= times.onset_ns[b]
    assert det.mean() > 0.8


def test_no_rest_falls_back_and_no_prompts_is_skipped(tmp_path, caplog):
    from deftemg_acquisition.data.labels import Alignment, aligned_prompts
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    cfg = SyntheticConfig(n_blocks=1, per_block=4, lead_in_s=0.8, seed=4, compress=False)
    info = scan_trials([write_session(tmp_path / "n", cfg)])[0]
    trial = load_trial(info.npz_path)
    with caplog.at_level("WARNING"):
        p, spec, _ = aligned_prompts(trial, LabelSpec(), Alignment("emg"))
    assert "prompt-aligned labels" in caplog.text
    np.testing.assert_array_equal(p.onset_ns, trial.prompts.onset_ns)
    assert spec == LabelSpec()
    empty = dataclasses.replace(info, n_prompts=0)
    assert select_trials([empty])[1] == [(info.id, "no prompts")]


def test_calibration_block_and_gains(tmp_path):
    from deftemg_acquisition.data.calibration import (
        press_fraction,
        session_calibration,
        session_gain,
    )
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    cfg = SyntheticConfig(n_blocks=2, per_block=8, calibration=True, calibration_reps=2, seed=8,
                          compress=False)  # fmt: skip
    session = write_session(tmp_path / "C", cfg)
    infos = scan_trials([session])
    assert infos[0].is_calibration and infos[0].level_names == ("max",)
    usable, skipped = select_trials(infos)
    assert len(usable) == 2 and skipped == [(infos[0].id, "calibration block")]
    cal = session_calibration(session)
    assert cal is not None and set(cal.max_rms_uv) == set(DIGITS)
    noise = np.median(cal.rest_sd_uv)
    assert np.all(cal.rest_sd_uv > 0) and all(cal.max_amplitude_uv(d) > 3 * noise for d in DIGITS)
    frac = press_fraction(load_trial(usable[0].npz_path), cal)
    low = [r["fraction_of_max"] for r in frac if r["level"] == "low"]
    high = [r["fraction_of_max"] for r in frac if r["level"] == "high"]
    assert low and high and np.median(low) < np.median(high) and max(high) < 1
    single = load_trial(usable[0].npz_path).prompts.digits.sum(axis=1) == 1
    assert {r["prompt"] for r in frac} <= set(np.flatnonzero(single))
    assert set(cal.rep_ratio) == set(DIGITS) and not [w for w in cal.warnings if "noise" in w]
    assert session_gain(cal, "rest") == pytest.approx(1 / np.median(cal.rest_sd_uv))
    assert session_gain(cal, "max") < session_gain(cal, "rest")
    assert session_calibration(tmp_path) is None  # no trials there
    one = write_session(tmp_path / "D", SyntheticConfig(n_blocks=1, per_block=8, calibration=True,
                                                         seed=9, compress=False))  # fmt: skip
    cal = session_calibration(one)  # one press per digit (the default): no fatigue ratio
    assert set(cal.max_rms_uv) == set(DIGITS) and not cal.rep_ratio


def test_pinch_calibration_and_press_fractions(tmp_path):
    """A pinch calibration: each digit's maximum is the best pinch it takes part in (the thumb:
    all four), fatigue compares repeats of the same pinch, and a task pinch is measured against
    the same pinch's maximum. A press calibration gives pinches no reference (other posture)."""
    from deftemg_acquisition.data.calibration import press_fraction, session_calibration
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    base = {"n_blocks": 1, "per_block": 8, "task_sets": ("pinch",), "calibration": True,
            "seed": 4, "compress": False}  # fmt: skip
    cfg = SyntheticConfig(**base, calibration_task="pinch", calibration_reps=2)
    session = write_session(tmp_path / "P", cfg)
    cal = session_calibration(session)
    assert cal.task == "pinch" and set(cal.max_rms_uv) == set(DIGITS)
    assert set(cal.chord_max_rms_uv) == {f"thumb+{d}" for d in DIGITS[1:]}
    assert cal.rep_ratio["thumb"] == pytest.approx(cal.rep_ratio["index"])  # thumb + index twice
    best = max(cal.chord_max_rms_uv.values(), key=lambda a: float(np.sqrt(np.mean(a**2))))
    assert np.array_equal(cal.max_rms_uv["thumb"], best)  # the thumb: its strongest pinch
    task = load_trial(select_trials(scan_trials([session]))[0][0].npz_path)
    frac = press_fraction(task, cal)
    assert frac and {r["digit"] for r in frac} <= set(cal.chord_max_rms_uv)
    assert all(0 < r["fraction_of_max"] < 1.5 for r in frac)
    pressed = write_session(tmp_path / "Q", SyntheticConfig(**base))  # a press calibration
    assert press_fraction(load_trial(select_trials(scan_trials([pressed]))[0][0].npz_path),
                          session_calibration(pressed)) == []  # fmt: skip


def test_calibration_per_trial(tmp_path):
    import json
    import shutil

    from deftemg_acquisition.data.calibration import calibration_status, session_calibration
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    session = write_session(tmp_path / "C", SyntheticConfig(n_blocks=2, per_block=6,
                            calibration=True, seed=9, compress=False))  # fmt: skip
    log_path = session / "trials.jsonl"
    recs = [json.loads(line) for line in log_path.read_text().splitlines()]
    late_npz = session / "trial_099_calibration_late.npz"
    shutil.copy(recs[0]["npz_path"], late_npz)
    late = {**recs[0], "index": 99, "label": "calibration_late", "npz_path": str(late_npz),
            "host_start_ns": recs[1]["host_end_ns"] + 1}  # fmt: skip

    def write(records):
        log_path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return {i.label: i for i in scan_trials([session])}

    infos = write([*recs, late])
    first, second = (load_trial(infos[r["label"]].npz_path) for r in recs[1:])
    cal_first = infos[recs[0]["label"]].id
    assert session_calibration(session).trial_id == infos["calibration_late"].id  # the latest
    start = lambda t: t.info.record["host_start_ns"]  # noqa: E731
    assert session_calibration(session, start(first)).trial_id == cal_first
    assert session_calibration(session, start(second)).trial_id == infos["calibration_late"].id

    # The early block stopped part-way: an earlier trial uses the later block.
    infos = write([{**recs[0], "metadata": {**recs[0]["metadata"], "completed": False}},
                   *recs[1:], late])  # fmt: skip
    first = load_trial(infos[recs[1]["label"]].npz_path)
    assert session_calibration(session, start(first)).trial_id == infos["calibration_late"].id
    # Only a block stopped part-way: no calibration.
    write([{**recs[0], "metadata": {**recs[0]["metadata"], "completed": False}}, *recs[1:]])
    assert session_calibration(session, start(first)) is None
    assert calibration_status(session) == "incomplete"


def test_unusable_calibration_blocks_and_weak_digits(tmp_path):
    import json

    from deftemg_acquisition.data import calibration as C
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    session = write_session(tmp_path / "B", SyntheticConfig(n_blocks=1, per_block=6,
                            calibration=True, seed=1, compress=False))  # fmt: skip
    recs = [json.loads(line) for line in (session / "trials.jsonl").read_text().splitlines()]
    first = recs[0]["npz_path"]
    with np.load(first) as d:
        a = {k: d[k] for k in d.files}
    good = dict(a)
    # The good block (written as a later one): the subject didn't press index in it.
    emg, t = a["emg_uv"].astype(float), a["emg_host_time_ns"]
    for i in np.flatnonzero(a["prompt_digits"][:, 1]):
        lo, hi = np.searchsorted(t, [a["prompt_onset_host_ns"][i] - int(0.4e9),
                                     a["prompt_offset_host_ns"][i] + int(0.6e9)])  # fmt: skip
        emg[lo:hi] = emg[1000 : 1000 + hi - lo]
    good["emg_uv"] = np.round(emg * 7.5) / 7.5
    late_npz = session / "trial_090_calibration_late.npz"
    np.savez(late_npz, **good)
    # The first block: almost no rest before its first press, so it can't be used at all.
    keep = t >= a["prompt_onset_host_ns"][0] - int(1.2e9)
    for k in ("emg_uv", "emg_counts", "emg_seq", "emg_host_time_ns"):
        a[k] = a[k][keep]
    np.savez(first, **a)
    late = {**recs[0], "index": 90, "label": "calibration_late", "npz_path": str(late_npz),
            "host_start_ns": recs[-1]["host_end_ns"] + 1}  # fmt: skip
    (session / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [*recs, late]))

    blocks = C.session_blocks(session)
    assert isinstance(blocks[0][2], str) and "rest too short" in blocks[0][2]
    task = load_trial(recs[1]["npz_path"])
    cal = C.session_calibration(session, task.info.record["host_start_ns"])
    assert cal.trial_id.endswith("calibration_late")  # the broken block doesn't hide it
    assert "index" not in cal.max_rms_uv and any("index" in w and "left out" in w
                                                 for w in cal.warnings)  # fmt: skip
    assert {r["digit"] for r in C.press_fraction(task, cal)} <= set(cal.max_rms_uv)
    assert C.calibration_status(session) == "ok"


def test_calibration_is_measured_on_filtered_emg(tmp_path):
    """Calibration blocks of raw recordings are measured on the EMG filtered as ``filter-emg
    --harmonics all`` filters it (zero-phase 20 Hz high-pass and 60 Hz notches at every
    harmonic, per continuous segment), so an offset, drift and mains hum don't swamp the rest
    noise or the maximum presses."""
    import json

    from deftemg_acquisition.analysis.filtering import EmgFilter, FilterConfig, filter_trial
    from deftemg_acquisition.data import calibration as C
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    session = write_session(tmp_path / "H", SyntheticConfig(n_blocks=1, per_block=6,
                            calibration=True, seed=3, compress=False))  # fmt: skip
    recs = [json.loads(line) for line in (session / "trials.jsonl").read_text().splitlines()]
    filt = EmgFilter()  # the default: high-pass 20 Hz + comb60all, zero-phase
    clean = C.calibration_from_trial(load_trial(recs[0]["npz_path"]), filt)
    for rec in recs:  # what a raw recording adds: an offset, drift, mains hum and harmonics
        with np.load(rec["npz_path"]) as d:
            a = {k: d[k] for k in d.files}
        t_s = np.arange(len(a["emg_uv"])) / 2000.0  # sample times (host times jitter)
        hum = sum(20.0 / k * np.sin(2 * np.pi * 60 * k * t_s + k) for k in (1, 2, 3, 5))
        extra = 3000.0 + 400.0 * np.sin(2 * np.pi * 0.7 * t_s) + hum
        a["emg_uv"] = (a["emg_uv"] + extra[:, None]).astype(a["emg_uv"].dtype)
        np.savez(rec["npz_path"], **a)
    infos = scan_trials([session])
    cal_info = next(i for i in infos if i.is_calibration)
    task_info = next(i for i in infos if not i.is_calibration)

    raw = C.calibration_from_trial(load_trial(cal_info.npz_path))
    filtered = C.calibration_from_trial(load_trial(cal_info.npz_path), filt)
    assert raw.emg_filter is None and filtered.emg_filter == filt
    assert np.median(raw.rest_sd_uv) > 10 * np.median(filtered.rest_sd_uv)
    np.testing.assert_allclose(filtered.rest_sd_uv, clean.rest_sd_uv, rtol=0.02)
    for d in DIGITS:
        assert filtered.max_amplitude_uv(d) == pytest.approx(clean.max_amplitude_uv(d), rel=0.02)
    cal = C.session_calibration(session, emg_filter=filt)
    assert cal.max_amplitude_uv("index") == pytest.approx(filtered.max_amplitude_uv("index"))
    offline = filter_trial(load_trial(task_info.npz_path), FilterConfig(harmonics="all"))
    assert C.press_fraction(offline, cal)
    unfiltered = C.session_calibration(session)  # the drift hides every press
    assert unfiltered.emg_filter is None and not unfiltered.max_rms_uv


def test_press_detector_recovers_synthetic_timing(tmp_path):
    from deftemg_acquisition.data.press_times import estimate_press_times
    from deftemg_acquisition.data.synthetic import SyntheticConfig, write_session

    cfg = SyntheticConfig(n_blocks=4, lead_s=(0.2, 0.2), lag_s=(0.3, 0.3), strength=2.0,
                          seed=11, compress=False)  # fmt: skip
    on_err, off_err = [], []
    for info in scan_trials([write_session(tmp_path / "s", cfg)]):
        trial = load_trial(info.npz_path)
        times = estimate_press_times(trial, rest_span_ns(trial, LabelSpec(), 1.0, 0.3))
        on, off = times.lags_ms(trial.prompts)
        ok = times.detected & ~times.onset_censored & ~times.offset_censored
        on_err += list(on[ok] - (-200 + 16))  # true lead 200 ms; drawn 16 ms after schedule
        off_err += list(off[ok] - (300 + 16))
    assert abs(np.median(on_err)) < 30 and abs(np.median(off_err)) < 60
    assert len(on_err) >= 30
