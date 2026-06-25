# scripts/run_stage_c.py
"""Stage-C run entry: binds the real GPU ops into the pure orchestrator + writes JSON/report.
The ONLY file that touches the GPU.  MODE=smoke (tiny plumbing) | full (paid 2-seed run)."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _bp in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _bp not in sys.path:
        sys.path.insert(0, _bp)

import htdml.paths as _p  # noqa: E402
_p.bootstrap_paths()
from htdml import orchestrator as O  # noqa: E402
from htdml.driver import AcceptanceConstants  # noqa: E402
from scripts.calib_logic import WallClock  # noqa: E402

def parse_config(env):
    mode = env.get("MODE", "full").lower()
    budget_h = float(env.get("BUDGET_H", "4.0"))
    if "SEEDS" in env:
        seeds = [int(s) for s in env["SEEDS"].split(",") if s.strip()]
    else:
        seeds = [1] if mode == "smoke" else [1, 2]
    const = O.FrozenConstants(GPU_H_CAP=budget_h)        # ESS_min/C/L_traj/... frozen defaults (PINS)
    return seeds, const, mode

def build_provenance():
    def _git(*a):
        try: return subprocess.check_output(["git", *a], cwd=os.path.dirname(__file__) or ".").decode().strip()
        except Exception: return "unknown"
    try:
        import jax; backend = jax.default_backend()
    except Exception: backend = "unknown"
    try:
        from harness.reversible_scan import is_patch_live; patch = bool(is_patch_live())
    except Exception: patch = None
    return {"git_sha": _git("rev-parse", "HEAD"), "env_freeze": "env-h200-freeze.txt",
            "jax_backend": backend, "is_patch_live": patch}

_REPORT = "# Stage-C run report\n\nOutcome: **{outcome}**\n\nWritten after run_stage_c.json. MEASURE-ONLY — no wiki tag move.\n"

def write_outputs(result, outdir, *, mode):
    os.makedirs(outdir, exist_ok=True)
    name = "run_stage_c_smoke.json" if mode == "smoke" else "run_stage_c.json"
    with open(os.path.join(outdir, name), "w") as f:        # JSON FIRST
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(outdir, "report.md"), "w") as f:  # report AFTER the json exists
        f.write(_REPORT.format(outcome=result["outcome"]))
    return 2 if result["outcome"] == "BUDGET-WALL" else 0

class RealOps:
    """Binds the real GPU ops (driver.* + TrainabilityProbe + FID).  Smoke-only — no unit test.
    Each method wires the verbatim primitives; see the spec §3.1 seam contract for shapes."""
    def __init__(self, const, *, smoke): self.const = const; self.smoke = smoke
    # NOTE: method bodies call driver.stage_a_pretrain / stage_b_train_latent_dtm / fork_checkpoint /
    # joint_update_step + TrainabilityProbe.evaluate/.calibrate + FID-on-decoded-28x28.  They are
    # implemented during the on-box smoke pass (MODE=smoke) where a GPU is present; the structure mirrors
    # p0_decision.md §On-box RUNBOOK.  Kept out of the import path's CPU surface intentionally.
    def pretrain_encoder(self, seed, clock): raise NotImplementedError("smoke-pass wiring")
    def train_latent_dtm(self, enc, seed, clock): raise NotImplementedError("smoke-pass wiring")
    def calibrate_tau(self, dtm, clock): raise NotImplementedError("smoke-pass wiring")
    def estimate_probe_cost(self, L): raise NotImplementedError("smoke-pass wiring")
    def fork(self, dtm, workdir): raise NotImplementedError("smoke-pass wiring")
    def epoch_block_pair(self, j, c, lr, L, clock): raise NotImplementedError("smoke-pass wiring")
    def probe_committed_pair(self, j, c, L, clock): raise NotImplementedError("smoke-pass wiring")
    def commit_pair(self, j, c): raise NotImplementedError("smoke-pass wiring")
    def rollback_pair(self, j, c): raise NotImplementedError("smoke-pass wiring")

def main(env=None, outdir=None):
    env = os.environ if env is None else env
    seeds, const, mode = parse_config(env)
    outdir = outdir or os.path.join(os.path.dirname(__file__), "..", "results")
    clock = WallClock(cap_seconds=const.GPU_H_CAP * 3600.0)
    ops = RealOps(const, smoke=(mode == "smoke"))
    result = O.run_stage_c(ops, seeds=seeds, acc=AcceptanceConstants(
        ESS_min=const.ESS_min, C=const.C, L_traj=const.L_traj, N_chains=const.N_chains, N_R=const.N_R),
        const=const, workdir=os.path.join(outdir, "work"), provenance=build_provenance(), clock=clock)
    return write_outputs(result, outdir, mode=mode)

if __name__ == "__main__":
    sys.exit(main())
