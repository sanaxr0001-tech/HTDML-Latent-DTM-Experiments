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
    def calibrate_tau(self, dtm: Any, clock: WallClock) -> dict: ...                 # {tau_hat_layers, cal_stable, failed_layer}
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
