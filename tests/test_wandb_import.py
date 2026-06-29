"""
Tests for the W&B import transform (scripts/wandb_import.py).

This guards the *pure* JSON -> run-payload transform only (no network, no wandb).
Expected numbers are hand-derived from the raw run_stage_c.json files and
cross-checked against experiments/exp3-lambda-sweep/report.md, so the test pins
the faithful-import contract independently of the implementation.
"""

import math

import pytest

from scripts.wandb_import import RUN_SPECS, iter_all_payloads, build_seed_payloads, load_run_json


def _payloads_by_name():
    return {p["name"]: p for p in iter_all_payloads()}


def test_total_run_count():
    # exp1: 2 seeds, exp2: 2 seeds, exp3: 3 lambda x 2 seeds = 6  ->  10 runs.
    payloads = list(iter_all_payloads())
    assert len(payloads) == 10
    # Every payload has a unique deterministic id and a group.
    ids = {p["id"] for p in payloads}
    assert len(ids) == 10
    groups = {p["group"] for p in payloads}
    assert groups == {"exp1", "exp2", "exp3"}


def test_specs_cover_three_experiments():
    shorts = {s["exp_short"] for s in RUN_SPECS}
    assert shorts == {"exp1", "exp2", "exp3"}
    # exp3 sweeps three lambdas.
    exp3_lams = sorted(s["lam"] for s in RUN_SPECS if s["exp_short"] == "exp3")
    assert exp3_lams == [0.1, 0.3, 0.5]


def test_exp1_calibration_fail_has_no_stage_c():
    p = _payloads_by_name()["exp1-lam1.0-seed1"]
    assert p["config"]["run_outcome"] == "Q-CALIBRATION-FAIL"
    assert p["config"]["seed_token"] == "Q-CALIBRATION-FAIL"
    assert p["config"]["stage_c_reached"] is False
    assert p["config"]["cal_stable"] is False
    # Infinity BCE/FID must be sanitized to None for the summary (W&B can't take inf).
    assert p["summary"]["bce_joint"] is None
    assert p["summary"]["fid_joint"] is None
    # The deterministic reconfirm tau is still present.
    assert p["summary"]["tau_hat_worst"] == pytest.approx(5.07148915170648)
    # No per-layer / reject content for a cal-fail run.
    assert p["per_layer_rows"] == []
    assert p["reject_rows"] == []
    # ...but the per-layer reconfirm tau must NOT be collapsed to its max (audit fix):
    assert p["config"]["tau_hat_layers"] == pytest.approx(
        [2.060217011567023, 2.4588426584053873, 2.279833444807127, 5.07148915170648])
    assert [r[0] for r in p["reconfirm_rows"]] == [0, 1, 2, 3]
    assert p["reconfirm_rows"][3][1] == pytest.approx(5.07148915170648)
    assert p["config"]["reconfirm_failed_layer"] == 0
    # Infinity joint metrics must keep an explicit divergence signal, not just None.
    assert p["summary"]["bce_joint"] is None
    assert p["summary"]["bce_joint_diverged"] is True
    assert "inf" in p["summary"]["bce_joint_raw"].lower()


def test_exp2_seed2_failed_axes_preserved():
    """seed 2 of exp2/exp3 has a marginal cal axis on layer 2 -> must survive."""
    p = _payloads_by_name()["exp2-lam1.0-seed2"]
    assert p["config"]["reconfirm_failed_axes"] == [[], [], ["tau_hat"], []]
    # surfaced in the reconfirm table too
    assert p["reconfirm_rows"][2][2] == "tau_hat"


def test_exp2_seed1_gate_reductions_match_report():
    p = _payloads_by_name()["exp2-lam1.0-seed1"]
    s = p["summary"]
    assert p["config"]["stage_c_reached"] is True
    # lower-quartile Q ratio -> report: 0.93x
    assert s["lq_Q_ratio"] == pytest.approx(0.925, abs=3e-3)
    # worst-layer tau ratio -> report: 1.04x
    assert s["worst_tau_ratio"] == pytest.approx(1.037, abs=3e-3)
    # worst (min) ESS joint/control -> report: 20.1 / 20.9
    assert s["worst_ESS_joint"] == pytest.approx(20.123, abs=1e-2)
    assert s["worst_ESS_control"] == pytest.approx(20.872, abs=1e-2)
    # quality held, improvement not met -> NEGATIVE
    assert s["quality_held"] is True
    assert s["improvement_met"] is False
    assert p["config"]["seed_token"] == "HTDML-MARGIN-NEGATIVE"
    # 10 blocks, 2 rejects in the reject log.
    assert s["n_blocks"] == 10
    assert s["n_rejects"] == 2
    assert len(p["per_layer_rows"]) == 4


def test_exp3_lam01_seed1_positive_via_tau_leg():
    p = _payloads_by_name()["exp3-lam0.1-seed1"]
    s = p["summary"]
    assert p["config"]["seed_token"] == "HTDML-MARGIN-POSITIVE"
    # report: lq-Q 0.91x (fails), tau 0.51x (passes) -> improvement via tau leg
    assert s["lq_Q_ratio"] == pytest.approx(0.907, abs=3e-3)
    assert s["worst_tau_ratio"] == pytest.approx(0.514, abs=3e-3)
    assert s["improvement_met"] is True
    assert s["ess_nondeg"] is True


def test_gate_legs_consistent_with_driver_token_for_all_stage_c_runs():
    """Strong invariant: our recomputed pass/fail must agree with the token the
    driver emitted, for every run that reached Stage C."""
    for p in iter_all_payloads():
        if not p["config"]["stage_c_reached"]:
            continue
        assert p["summary"]["gate_consistent"] is True, p["name"]


def test_run_level_negative_everywhere_except_per_seed_positives():
    by = _payloads_by_name()
    # Run-level outcome is NEGATIVE/CAL-FAIL for all; only two per-seed tokens are POSITIVE.
    positives = [n for n, p in by.items() if p["config"]["seed_token"] == "HTDML-MARGIN-POSITIVE"]
    assert sorted(positives) == ["exp3-lam0.1-seed1", "exp3-lam0.3-seed1"]
    # No run is a run-level POSITIVE.
    assert all(p["config"]["run_outcome"] != "HTDML-MARGIN-POSITIVE" for p in by.values())


def test_summary_values_are_wandb_safe():
    """No inf/nan may leak into the summary dict (W&B silently drops/garbles them)."""
    for p in iter_all_payloads():
        for k, v in p["summary"].items():
            if isinstance(v, float):
                assert math.isfinite(v), f"{p['name']}.{k} = {v}"
