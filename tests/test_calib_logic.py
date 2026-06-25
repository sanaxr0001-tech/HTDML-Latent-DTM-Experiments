"""Task 12 — CPU unit tests for scripts/calib_logic.py (the NON-GPU calibration logic).

These verify the wall-time budget guard, the freeze-from-measurement rules
(L_traj / N_chains / N_R / C), and the a-priori ESS_min RULE — WITHOUT any dtm.train / GPU
(build-notes §"CPU vs GPU").  Run before the 4060 GPU pass.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import calib_logic as cl  # noqa: E402


# ============================================================================== wall-time guard
def test_wallclock_within_budget_then_over():
    wc = cl.WallClock(cap_seconds=0.05)
    assert wc.checkpoint("immediate") is True          # fresh → within budget
    assert wc.remaining() > 0
    time.sleep(0.06)
    assert wc.over_cap() is True
    assert wc.checkpoint("after-sleep") is False       # over → False, no raise


def test_wallclock_raises_on_over_when_asked():
    wc = cl.WallClock(cap_seconds=0.02)
    time.sleep(0.03)
    with pytest.raises(cl.BudgetWall):
        wc.checkpoint("stage", raise_on_over=True)


def test_wallclock_would_exceed_declines_a_too_big_stage():
    wc = cl.WallClock(cap_seconds=10.0)
    # an estimated 100s stage on a 10s cap (fresh clock) → decline
    assert wc.would_exceed(100.0) is True
    # a 1s stage with a 2x margin = 2s < 10s → allowed
    assert wc.would_exceed(1.0, margin=2.0) is False


# ============================================================================== ESS_min a-priori RULE
def test_ess_min_rule_is_fixed_floor_independent_of_measurement():
    r = cl.ess_min_rule()
    assert r["ESS_min"] == 10.0
    # equivalent τ ceiling: K/(2·ESS_min) = 50/20 = 2.5
    assert r["tau_ceiling"] == pytest.approx(2.5)
    assert "a-priori" in r["rule"]
    # the rule does NOT take any measured τ̂ / smoke result as input — invariance check:
    r2 = cl.ess_min_rule(k_window=50, ess_floor=10.0)
    assert r2["ESS_min"] == r["ESS_min"]


def test_ess_min_rule_floor_is_configurable_but_default_is_10():
    # the RULE value comes from the floor only (a-priori), not from any data
    assert cl.ess_min_rule(ess_floor=20.0)["ESS_min"] == 20.0
    assert cl.ess_min_rule()["ESS_min"] == cl.ESS_MIN_FLOOR == 10.0


# ============================================================================== freeze-from-measurement
def test_freeze_satisfies_trajectory_adequacy_gate():
    """L_traj ≥ C·τ̂ for the measured τ̂ (gate (i) self-consistency)."""
    out = cl.freeze_from_measurement(tau_hat=3.0)
    assert out["L_traj"] >= out["C"] * out["tau_hat"]
    assert out["L_traj"] > cl.K_WINDOW          # ρ_Y(50) defined
    assert out["C"] == 5.0
    assert out["N_R"] == 16


def test_freeze_L_traj_beats_white_noise_se():
    """L_traj ≥ 1/se² so the white-noise autocorr SE ≪ 0.05."""
    out = cl.freeze_from_measurement(tau_hat=1.0, se_target=0.05)
    assert out["L_traj"] >= 1.0 / (0.05 ** 2)   # ≥ 400
    # N_chains·L_traj also beats the SE target
    assert out["N_chains"] * out["L_traj"] >= 1.0 / (0.05 ** 2)


def test_freeze_scales_L_traj_with_large_tau():
    """A larger measured τ̂ forces a longer L_traj (adequacy binds)."""
    small = cl.freeze_from_measurement(tau_hat=2.0)
    large = cl.freeze_from_measurement(tau_hat=200.0)
    assert large["L_traj"] > small["L_traj"]
    assert large["L_traj"] >= large["C"] * large["tau_hat"]


def test_freeze_L_traj_is_multiple_of_K():
    out = cl.freeze_from_measurement(tau_hat=7.3)
    assert out["L_traj"] % cl.K_WINDOW == 0


def test_freeze_n_chains_non_degenerate():
    out = cl.freeze_from_measurement(tau_hat=1000.0)   # huge L_traj → tiny n_chains by SE
    assert out["N_chains"] >= 4


def test_freeze_rejects_bad_tau():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            cl.freeze_from_measurement(tau_hat=bad)


def test_freeze_respects_cap():
    out = cl.freeze_from_measurement(tau_hat=500.0, l_traj_cap=600)
    assert out["L_traj"] <= 600


def test_freeze_emits_justifications_for_all_four():
    out = cl.freeze_from_measurement(tau_hat=4.0)
    for key in ("L_traj", "N_chains", "N_R", "C", "tau_hat"):
        assert key in out["justification"]
        assert isinstance(out["justification"][key], str) and out["justification"][key]
