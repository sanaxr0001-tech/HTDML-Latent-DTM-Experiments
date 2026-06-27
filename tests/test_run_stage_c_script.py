# tests/test_run_stage_c_script.py
import json, os
import pytest
from scripts import run_stage_c as R

def test_parse_config_env_defaults_and_overrides():
    seeds, const, mode = R.parse_config({"MODE": "full", "BUDGET_H": "4.0", "SEEDS": "1,2"})
    assert mode == "full" and seeds == [1, 2] and const.GPU_H_CAP == 4.0 and const.ESS_min == 10.0
    seeds_s, _c, mode_s = R.parse_config({"MODE": "smoke"})
    assert mode_s == "smoke" and len(seeds_s) == 1            # smoke defaults to a single seed

def test_parse_config_full_requires_explicit_budget_h():
    """Footgun fix: MODE=full with NO BUDGET_H must FAIL LOUD pre-GPU (a silent 4.0 default would
    guillotine seed-2's 200-epoch Stage B at ~5.1 GPU-h/seed). Smoke keeps the cheap 4.0 default."""
    with pytest.raises(ValueError):
        R.parse_config({"MODE": "full"})                       # no BUDGET_H → loud, pre-GPU
    _s, const, mode = R.parse_config({"MODE": "full", "BUDGET_H": "16.0"})
    assert mode == "full" and const.GPU_H_CAP == 16.0
    _ss, const_s, mode_s = R.parse_config({"MODE": "smoke"})    # smoke default stays 4.0
    assert mode_s == "smoke" and const_s.GPU_H_CAP == 4.0

def test_parse_config_lambda_joint_env_override_and_default():
    """exp3 lower-λ sweep knob: LAMBDA_JOINT overrides the joint-arm steering strength
    (FrozenConstants.lambda_joint, a run param — NOT a frozen-five PIN); unset keeps the frozen 1.0.
    The override must not disturb the frozen-five PINS (L_traj/N_chains/...)."""
    _s, const, _m = R.parse_config({"MODE": "full", "BUDGET_H": "4.0"})
    assert const.lambda_joint == 1.0                                  # default: frozen 1.0 preserved
    _s2, const2, _m2 = R.parse_config({"MODE": "full", "BUDGET_H": "4.0", "LAMBDA_JOINT": "0.3"})
    assert const2.lambda_joint == 0.3                                 # override honored
    _s3, const3, _m3 = R.parse_config({"MODE": "smoke", "LAMBDA_JOINT": "0.5"})
    assert const3.lambda_joint == 0.5                                 # works in smoke too
    assert const3.L_traj == 400 and const3.N_chains == 4             # frozen-five PINS untouched

def test_write_outputs_json_before_report_and_exit_code(tmp_path):
    res = {"outcome": "HTDML-MARGIN-NEGATIVE", "constants": {}, "provenance": {}, "budget": {},
           "seeds": [], "two_seed": {"run_token": "HTDML-MARGIN-NEGATIVE"}}
    code = R.write_outputs(res, str(tmp_path), mode="full")
    assert code == 0
    assert os.path.exists(tmp_path / "run_stage_c.json")
    assert json.load(open(tmp_path / "run_stage_c.json"))["outcome"] == "HTDML-MARGIN-NEGATIVE"
    assert os.path.exists(tmp_path / "report.md")            # report written AFTER the json exists

def test_write_outputs_budget_wall_exit_2(tmp_path):
    res = {"outcome": "BUDGET-WALL", "constants": {}, "provenance": {}, "budget": {}, "seeds": [],
           "two_seed": {"run_token": "BUDGET-WALL"}}
    assert R.write_outputs(res, str(tmp_path), mode="full") == 2     # exit 2 per p0

def test_build_provenance_has_git_and_backend_keys():
    prov = R.build_provenance()
    for k in ("git_sha", "env_freeze", "jax_backend", "is_patch_live"):
        assert k in prov

def test_realops_stage_b_dir_and_resuming():
    """RealOps carries outdir + resume_from; the pure path/predicate helpers are CPU-safe (no jax)."""
    _s, const, _m = R.parse_config({"MODE": "smoke"})
    ops = R.RealOps(const, smoke=True, outdir="/o", resume_from=None)
    assert ops.resuming() is False
    assert ops._stage_b_dir("/o", 2) == os.path.join("/o", "checkpoints", "seed2", "stage_b")
    ops_r = R.RealOps(const, smoke=True, outdir="/o", resume_from="/r")
    assert ops_r.resuming() is True

def test_resolve_outdir_prefers_arg_then_env_then_default():
    """OUTDIR knob (exp2): explicit arg > OUTDIR env > default ../results, so run-2 can write to its own
    experiments/exp2-cal-gate-fix/artifacts/ without clobbering run-1, and the default stays backward-compat."""
    assert R.resolve_outdir({}, "/x/y") == "/x/y"                               # explicit arg wins
    assert R.resolve_outdir({"OUTDIR": "experiments/exp2-cal-gate-fix/artifacts"}) == \
        "experiments/exp2-cal-gate-fix/artifacts"                              # OUTDIR env
    assert R.resolve_outdir({"OUTDIR": "env/path"}, "/arg/path") == "/arg/path"  # arg overrides env
    assert R.resolve_outdir({}).replace("\\", "/").endswith("results")          # default (backward-compat)
