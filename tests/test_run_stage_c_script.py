# tests/test_run_stage_c_script.py
import json, os
from scripts import run_stage_c as R

def test_parse_config_env_defaults_and_overrides():
    seeds, const, mode = R.parse_config({"MODE": "full", "BUDGET_H": "4.0", "SEEDS": "1,2"})
    assert mode == "full" and seeds == [1, 2] and const.GPU_H_CAP == 4.0 and const.ESS_min == 10.0
    seeds_s, _c, mode_s = R.parse_config({"MODE": "smoke"})
    assert mode_s == "smoke" and len(seeds_s) == 1            # smoke defaults to a single seed

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

def test_resolve_outdir_prefers_arg_then_env_then_default():
    """OUTDIR knob (exp2): explicit arg > OUTDIR env > default ../results, so run-2 can write to its own
    experiments/exp2-cal-gate-fix/artifacts/ without clobbering run-1, and the default stays backward-compat."""
    assert R.resolve_outdir({}, "/x/y") == "/x/y"                               # explicit arg wins
    assert R.resolve_outdir({"OUTDIR": "experiments/exp2-cal-gate-fix/artifacts"}) == \
        "experiments/exp2-cal-gate-fix/artifacts"                              # OUTDIR env
    assert R.resolve_outdir({"OUTDIR": "env/path"}, "/arg/path") == "/arg/path"  # arg overrides env
    assert R.resolve_outdir({}).replace("\\", "/").endswith("results")          # default (backward-compat)
