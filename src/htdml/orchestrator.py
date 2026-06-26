# src/htdml/orchestrator.py
"""Pure Stage-C orchestration: sequence + gates + reject loop + routing + result assembly,
behind an injected StageCOps seam.  NO jax import here — CPU-testable with fakes."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

import htdml.paths as _p; _p.bootstrap_paths()
from htdml.driver import (SeedMetrics, AcceptanceConstants, RejectState, reject_gate,
                          route_seed, route_run, apply_rejection, apply_acceptance)
from scripts.calib_logic import WallClock, BudgetWall, freeze_from_measurement


@dataclass
class BlockResult:
    joint_layers: List[dict]
    control_layers: List[dict]
    bce_joint: float
    fid_joint: float
    bce_control: float
    fid_control: float
    gpu_h: float = 0.0


@dataclass(frozen=True)
class FrozenConstants:
    ESS_min: float = 10.0
    C: float = 5.0
    L_traj: int = 400
    N_chains: int = 4
    N_R: int = 16
    GPU_H_CAP: float = 4.0
    lambda_joint: float = 1.0     # Stage-C steering strength (control=0, joint=this); run param
    lr0: float = 1e-3             # encoder Stage-C LR (halved on reject)
    max_joint_epochs: int = 20
    epochs_per_block: int = 2
    max_consecutive: int = 2


@dataclass
class ReconfirmResult:
    status: str               # "proceed" | "cal_fail" | "budget_wall"
    L_adeq: Optional[int]
    record: dict


@dataclass
class SeedResult:
    seed: int
    token: str
    metrics: SeedMetrics
    reconfirm: dict
    reject_log: List[dict] = field(default_factory=list)


class StageCOps(Protocol):
    def pretrain_encoder(self, seed: int, clock: WallClock) -> Any: ...
    def train_latent_dtm(self, encoder: Any, seed: int, clock: WallClock) -> Any: ...
    def calibrate_tau(self, dtm: Any, clock: WallClock) -> dict: ...                 # {tau_hat_layers, cal_stable, failed_layer, cal_curves?, failed_axes?}
    def estimate_probe_cost(self, L_traj: int) -> float: ...                          # seconds
    def fork(self, dtm: Any, workdir: str) -> Tuple[Any, Any]: ...                    # (control, joint)
    def epoch_block_pair(self, joint, control, encoder_lr: float, L_traj: int, clock: WallClock) -> BlockResult: ...
    def probe_committed_pair(self, joint, control, L_traj: int, clock: WallClock) -> BlockResult: ...
    def commit_pair(self, joint, control) -> None: ...
    def rollback_pair(self, joint, control) -> None: ...


def build_seed_metrics(block: BlockResult, *, cal_all_stable: bool = True,
                       gpu_h: float = 0.0, budget_wall: bool = False) -> SeedMetrics:
    return SeedMetrics(
        joint_layers=block.joint_layers, control_layers=block.control_layers,
        bce=block.bce_joint, fid=block.fid_joint,
        control_bce=block.bce_control, control_fid=block.fid_control,
        gpu_h=gpu_h, budget_wall=budget_wall, cal_all_stable=cal_all_stable,
    )


def reconfirm_l_traj(ops, dtm, clock, const) -> ReconfirmResult:
    cal = ops.calibrate_tau(dtm, clock)
    tau_layers = [float(t) for t in cal["tau_hat_layers"]]
    base = {"tau_hat_layers": tau_layers, "L_traj_frozen": const.L_traj}
    # persist the per-layer doubling curves + failed axes so a Q-CALIBRATION-FAIL is diagnosable from the
    # run JSON WITHOUT a re-run (run 5b9cbbc discarded them).  Additive + guarded: fakes/old ops that omit
    # the keys are unaffected; threaded once here so ALL branches (cal_fail/budget_wall/proceed) carry them.
    if "cal_curves" in cal:
        base["cal_curves"] = cal["cal_curves"]
    if "failed_axes" in cal:
        base["failed_axes"] = cal["failed_axes"]
    if not cal.get("cal_stable", True):
        base.update(cal_stable=False, failed_layer=cal.get("failed_layer"),
                    tau_hat_worst=(max(tau_layers) if tau_layers else None),
                    L_traj_adequate=None, adjusted=False, affordable=None)
        return ReconfirmResult("cal_fail", None, base)
    tau_worst = max(tau_layers)
    freeze = freeze_from_measurement(tau_worst)        # UNCAPPED — never pass l_traj_cap
    L_adeq = int(freeze["L_traj"])
    base.update(cal_stable=True, failed_layer=None, tau_hat_worst=tau_worst,
                L_traj_adequate=L_adeq, adjusted=bool(L_adeq > const.L_traj))
    if clock.would_exceed(ops.estimate_probe_cost(L_adeq)):
        base["affordable"] = False
        return ReconfirmResult("budget_wall", L_adeq, base)
    base["affordable"] = True
    return ReconfirmResult("proceed", L_adeq, base)


def run_reject_loop(ops, joint, control, clock, acc, const, L_adeq) -> Tuple[BlockResult, List[dict]]:
    rstate = RejectState(encoder_lr=const.lr0)
    last_accepted: Optional[BlockResult] = None
    reject_log: List[dict] = []
    epochs_done = 0
    while (not rstate.stop) and epochs_done < const.max_joint_epochs and not clock.over_cap():
        block = ops.epoch_block_pair(joint, control, rstate.encoder_lr, L_adeq, clock)
        epochs_done += const.epochs_per_block
        dec = reject_gate(block.joint_layers, block.control_layers, acc)
        if dec.reject:
            ops.rollback_pair(joint, control)
            rstate = apply_rejection(rstate, max_consecutive=const.max_consecutive)
            reject_log.append({"epoch": epochs_done, "reject": True, "reason": dec.reason,
                               "encoder_lr": rstate.encoder_lr})
        else:
            ops.commit_pair(joint, control)
            last_accepted = block
            rstate = apply_acceptance(rstate)
            reject_log.append({"epoch": epochs_done, "reject": False, "encoder_lr": rstate.encoder_lr})
        clock.checkpoint(f"joint_epoch_{epochs_done}", raise_on_over=True)
    if last_accepted is None:
        last_accepted = ops.probe_committed_pair(joint, control, L_adeq, clock)
    return last_accepted, reject_log


def _budget_wall_metrics(clock) -> SeedMetrics:
    return SeedMetrics(joint_layers=[], control_layers=[], bce=float("inf"), fid=float("inf"),
                       control_bce=0.0, control_fid=0.0, gpu_h=clock.elapsed() / 3600.0,
                       budget_wall=True, cal_all_stable=True)


def run_one_seed(ops, seed, clock, acc, const, workdir) -> SeedResult:
    try:
        enc = ops.pretrain_encoder(seed, clock); clock.checkpoint(f"seed{seed}_stageA", raise_on_over=True)
        dtm = ops.train_latent_dtm(enc, seed, clock); clock.checkpoint(f"seed{seed}_stageB", raise_on_over=True)
        rc = reconfirm_l_traj(ops, dtm, clock, const)
        if rc.status == "cal_fail":
            m = build_seed_metrics(BlockResult([], [], float("inf"), float("inf"), 0.0, 0.0),
                                   cal_all_stable=False, gpu_h=clock.elapsed() / 3600.0)
            return SeedResult(seed, route_seed(m, acc), m, rc.record, [])
        if rc.status == "budget_wall":
            m = _budget_wall_metrics(clock)
            return SeedResult(seed, route_seed(m, acc), m, rc.record, [])
        control, joint = ops.fork(dtm, workdir)
        block, log = run_reject_loop(ops, joint, control, clock, acc, const, rc.L_adeq)
        m = build_seed_metrics(block, cal_all_stable=True, gpu_h=clock.elapsed() / 3600.0)
        return SeedResult(seed, route_seed(m, acc), m, rc.record, log)
    except BudgetWall:
        m = _budget_wall_metrics(clock)
        return SeedResult(seed, route_seed(m, acc), m, {"affordable": False}, [])


TOKENS_ALL = ("HTDML-MARGIN-POSITIVE", "HTDML-MARGIN-NEGATIVE", "QUALITY-LOSS",
              "PLATEAU-UNRESOLVED", "Q-CALIBRATION-FAIL", "BUDGET-WALL")

def _layers_json(layers):
    keys = ("Q_struct_perp", "tau_int_Y", "ESS_hat", "r_grad[50]", "gradient_norm", "L_traj", "tau_hat", "cal_stable")
    return [{k: ly.get(k) for k in keys} for ly in layers]

def assemble_result(run_token, seed_results, const, provenance, clock) -> dict:
    return {
        "outcome": run_token,
        "constants": {"ESS_min": const.ESS_min, "C": const.C, "L_traj": const.L_traj,
                      "N_chains": const.N_chains, "N_R": const.N_R, "GPU_H_CAP": const.GPU_H_CAP,
                      "lambda_joint": const.lambda_joint},
        "provenance": dict(provenance),
        "budget": {"gpu_h_total": float(clock.elapsed() / 3600.0),
                   "budget_wall": any(sr.token == "BUDGET-WALL" for sr in seed_results)},
        "seeds": [{"seed": sr.seed, "token": sr.token,
                   "bce_joint": sr.metrics.bce, "fid_joint": sr.metrics.fid,
                   "bce_control": sr.metrics.control_bce, "fid_control": sr.metrics.control_fid,
                   "joint_layers": _layers_json(sr.metrics.joint_layers),
                   "control_layers": _layers_json(sr.metrics.control_layers),
                   "l_traj_reconfirm": sr.reconfirm, "reject_log": sr.reject_log}
                  for sr in seed_results],
        "two_seed": {"both_pass": run_token == "HTDML-MARGIN-POSITIVE",
                     "run_token": run_token},
    }

def run_stage_c(ops, *, seeds, acc, const, workdir, provenance, clock=None) -> dict:
    clock = clock or WallClock(cap_seconds=const.GPU_H_CAP * 3600.0)
    seed_results: List[SeedResult] = []
    for seed in seeds:
        sr = run_one_seed(ops, seed, clock, acc, const, os.path.join(workdir, f"seed{seed}"))
        seed_results.append(sr)
        if sr.token == "BUDGET-WALL":            # C6: shared budget exhausted — do not start the next seed
            break
    if any(sr.token == "BUDGET-WALL" for sr in seed_results):
        run_token = "BUDGET-WALL"
    elif len(seeds) == 1:                         # smoke / single-seed: report the seed token, no aggregation
        run_token = seed_results[0].token
    elif len(seed_results) < 2:                   # defensive: broke early without a budget-wall token
        run_token = "BUDGET-WALL"
    else:
        run_token = route_run(seed_results[0].metrics, seed_results[1].metrics, acc)
    return assemble_result(run_token, seed_results, const, provenance, clock)
