# tests/test_orchestrator.py
import htdml  # bootstrap vendored paths (conftest also does this)
from htdml import orchestrator as O
from htdml.driver import SeedMetrics

def _layer(q=1.0, tau=1.0, ess=20.0, r50=0.01, g=1.0):
    return {"Q_struct_perp": q, "tau_int_Y": tau, "ESS_hat": ess, "r_grad[50]": r50,
            "gradient_norm": g, "L_traj": 400, "tau_hat": tau, "cal_stable": True}

def _block(**kw):
    return O.BlockResult(joint_layers=[_layer() for _ in range(4)],
                         control_layers=[_layer() for _ in range(4)],
                         bce_joint=0.10, fid_joint=12.0, bce_control=0.10, fid_control=12.0, gpu_h=0.5)

def test_build_seed_metrics_maps_block_fields():
    m = O.build_seed_metrics(_block(), cal_all_stable=True, gpu_h=0.5, budget_wall=False)
    assert isinstance(m, SeedMetrics)
    assert m.bce == 0.10 and m.control_fid == 12.0
    assert len(m.joint_layers) == 4 and m.cal_all_stable is True and m.budget_wall is False


class _Clock:
    """Fake WallClock: would_exceed returns scripted; over_cap False; checkpoint no-op."""
    def __init__(self, exceed=False): self._exceed = exceed; self.calls = []
    def would_exceed(self, est, margin=1.0): self.calls.append(("would_exceed", est)); return self._exceed
    def over_cap(self): return False
    def elapsed(self): return 0.0
    def checkpoint(self, label="", *, raise_on_over=False): return True

class _Ops:
    """Minimal fake StageCOps; tests override individual methods."""
    def __init__(self, **over): self._o = over; self.seen = {}
    def __getattr__(self, name):
        def f(*a, **k): self.seen.setdefault(name, []).append((a, k)); return self._o[name](*a, **k) if name in self._o else None
        return f

def test_reconfirm_cal_fail_short_circuits():
    ops = _Ops(calibrate_tau=lambda dtm, clk: {"tau_hat_layers": [1.0, 1.0, 1.0, 99.0],
                                               "cal_stable": False, "failed_layer": 3})
    rc = O.reconfirm_l_traj(ops, dtm=object(), clock=_Clock(), const=O.FrozenConstants())
    assert rc.status == "cal_fail" and rc.L_adeq is None
    assert rc.record["failed_layer"] == 3 and rc.record["cal_stable"] is False
    assert "estimate_probe_cost" not in ops.seen   # freeze/estimate NEVER called on cal-fail

def test_reconfirm_within_frozen_no_adjust():
    ops = _Ops(calibrate_tau=lambda d, c: {"tau_hat_layers": [1.0, 1.4, 0.9, 1.1], "cal_stable": True, "failed_layer": None},
               estimate_probe_cost=lambda L: 10.0)
    rc = O.reconfirm_l_traj(ops, object(), _Clock(exceed=False), O.FrozenConstants())
    assert rc.status == "proceed" and rc.L_adeq == 400 and rc.record["adjusted"] is False
    assert rc.record["tau_hat_worst"] == 1.4   # scalar max passed downstream

def test_reconfirm_binds_affordable_adjusts():
    ops = _Ops(calibrate_tau=lambda d, c: {"tau_hat_layers": [1, 1, 1, 120.0], "cal_stable": True, "failed_layer": None},
               estimate_probe_cost=lambda L: 10.0)
    rc = O.reconfirm_l_traj(ops, object(), _Clock(exceed=False), O.FrozenConstants())
    assert rc.status == "proceed" and rc.L_adeq > 400 and rc.record["adjusted"] is True

def test_reconfirm_binds_unaffordable_budget_wall_uncapped():
    seen = {}
    def est(L): seen["L"] = L; return 99999.0
    ops = _Ops(calibrate_tau=lambda d, c: {"tau_hat_layers": [1, 1, 1, 120.0], "cal_stable": True, "failed_layer": None},
               estimate_probe_cost=est)
    rc = O.reconfirm_l_traj(ops, object(), _Clock(exceed=True), O.FrozenConstants())
    assert rc.status == "budget_wall" and rc.record["affordable"] is False
    assert seen["L"] == rc.L_adeq and rc.L_adeq > 400   # cost estimated at the UNCAPPED adequate L


# ---------------------------------------------------------------------------
# Task 3: run_reject_loop
# ---------------------------------------------------------------------------
from htdml.driver import AcceptanceConstants

def _acc(**kw):
    base = dict(ESS_min=10.0, C=5.0, L_traj=400, N_chains=4, N_R=16)
    base.update(kw); return AcceptanceConstants(**base)

def _good_block():  # passes reject_gate (Q not dropped, r50 ok, ESS ok)
    return O.BlockResult([_layer(q=2.0) for _ in range(4)], [_layer(q=1.0) for _ in range(4)],
                         0.1, 12.0, 0.1, 12.0, gpu_h=0.2)

def _bad_block():   # joint Q collapses vs control -> reject_gate rejects
    return O.BlockResult([_layer(q=0.1) for _ in range(4)], [_layer(q=1.0) for _ in range(4)],
                         0.1, 12.0, 0.1, 12.0, gpu_h=0.2)

def _baseline_block():   # committed baseline: joint==control (no λ-steering accepted)
    return O.BlockResult([_layer(q=1.0) for _ in range(4)], [_layer(q=1.0) for _ in range(4)],
                         0.1, 12.0, 0.1, 12.0, gpu_h=0.2)

class _LoopOps(_Ops):
    def __init__(self, blocks):
        super().__init__(); self._blocks = list(blocks); self.commits = 0; self.rollbacks = 0; self.lrs = []
    def epoch_block_pair(self, j, c, lr, L, clk): self.lrs.append(lr); return self._blocks.pop(0)
    def commit_pair(self, j, c): self.commits += 1
    def rollback_pair(self, j, c): self.rollbacks += 1
    def probe_committed_pair(self, j, c, L, clk): return _baseline_block()

def test_reject_loop_accept_then_stop_on_epoch_budget():
    ops = _LoopOps([_good_block()]); const = O.FrozenConstants(max_joint_epochs=2)
    block, log = O.run_reject_loop(ops, object(), object(), _Clock(), _acc(), const, 400)
    assert ops.commits == 1 and ops.rollbacks == 0 and log[-1]["reject"] is False

def test_reject_loop_two_consecutive_rejects_stop():
    ops = _LoopOps([_bad_block(), _bad_block()]); const = O.FrozenConstants(max_joint_epochs=20)
    block, log = O.run_reject_loop(ops, object(), object(), _Clock(), _acc(), const, 400)
    assert ops.rollbacks == 2 and ops.commits == 0
    assert ops.lrs == [const.lr0, const.lr0 / 2]      # halved LR on the 2nd block (paired, same on both arms)
    assert len([r for r in log if r["reject"]]) == 2

def test_reject_loop_no_accept_uses_committed_baseline():
    ops = _LoopOps([_bad_block(), _bad_block()]); const = O.FrozenConstants(max_joint_epochs=20)
    block, log = O.run_reject_loop(ops, object(), object(), _Clock(), _acc(), const, 400)
    # block is the probe_committed_pair baseline (joint==control), never the rolled-back candidate:
    assert block.joint_layers[0]["Q_struct_perp"] == block.control_layers[0]["Q_struct_perp"]


# ---------------------------------------------------------------------------
# Task 4: run_one_seed
# ---------------------------------------------------------------------------
class _SeedOps(_Ops):
    def __init__(self, *, cal, blocks=None, est=10.0, raise_on_block=False):
        super().__init__(); self._cal = cal; self._blocks = list(blocks or [_good_block()])
        self._est = est; self._raise = raise_on_block; self.forked = False
    def pretrain_encoder(self, seed, clk): return ("enc", seed)
    def train_latent_dtm(self, enc, seed, clk): return ("dtm", seed)
    def calibrate_tau(self, dtm, clk): return self._cal
    def estimate_probe_cost(self, L): return self._est
    def fork(self, dtm, wd): self.forked = True; return ("control", "joint")
    def epoch_block_pair(self, j, c, lr, L, clk):
        if self._raise: from scripts.calib_logic import BudgetWall; raise BudgetWall("over")
        return self._blocks.pop(0) if self._blocks else _good_block()
    def probe_committed_pair(self, j, c, L, clk): return _good_block()
    def commit_pair(self, j, c): pass
    def rollback_pair(self, j, c): pass

def test_seed_cal_fail_skips_stage_c():
    ops = _SeedOps(cal={"tau_hat_layers": [1, 1, 1, 99], "cal_stable": False, "failed_layer": 3})
    sr = O.run_one_seed(ops, 1, _Clock(), _acc(), O.FrozenConstants(), "/tmp/x")
    assert sr.token == "Q-CALIBRATION-FAIL" and ops.forked is False    # never forked / entered Stage C

def test_seed_budget_wall_on_reconfirm():
    ops = _SeedOps(cal={"tau_hat_layers": [1, 1, 1, 120], "cal_stable": True, "failed_layer": None}, est=99999)
    sr = O.run_one_seed(ops, 1, _Clock(exceed=True), _acc(), O.FrozenConstants(), "/tmp/x")
    assert sr.token == "BUDGET-WALL" and ops.forked is False

def test_seed_happy_path_positive_token():
    ops = _SeedOps(cal={"tau_hat_layers": [1, 1.4, 0.9, 1.1], "cal_stable": True, "failed_layer": None},
                   blocks=[_good_block()])
    sr = O.run_one_seed(ops, 1, _Clock(), _acc(), O.FrozenConstants(max_joint_epochs=2), "/tmp/x")
    assert sr.token in ("HTDML-MARGIN-POSITIVE", "HTDML-MARGIN-NEGATIVE") and ops.forked is True

def test_seed_budgetwall_raised_mid_loop_is_caught():
    ops = _SeedOps(cal={"tau_hat_layers": [1, 1, 1, 1], "cal_stable": True, "failed_layer": None}, raise_on_block=True)
    sr = O.run_one_seed(ops, 1, _Clock(), _acc(), O.FrozenConstants(), "/tmp/x")
    assert sr.token == "BUDGET-WALL"
