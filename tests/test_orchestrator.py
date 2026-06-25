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
