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
