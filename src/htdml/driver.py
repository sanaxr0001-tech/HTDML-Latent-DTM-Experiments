"""Stage A/B/C driver — orchestration + fork/out-of-band-restore + per-update reject gates + the
6-token outcome router (Task 9).

This is the integration that ties the whole companion together.  Three layers, with a hard split
between what is GPU-wired (the TRAINING) and what is CPU-unit-tested (all the LOGIC):

  ┌─ GPU-WIRED (smoke-deferred, Task 12) ───────────────────────────────────────────────────────┐
  │ Stage A   pretrain the BinaryAutoencoder (stage_a_loss = BCE + commitment + balance).        │
  │ Stage B   freeze the encoder, encode the full split ONCE via latent_adapter, train the       │
  │           LatentDTM on hard latents with ACP (dtm.train — GPU-only).                          │
  │ Stage C   FORK each Stage-B checkpoint into a matched control (λ=0) arm + a joint (λ>0) arm;  │
  │           the joint arm alternates one DTM epoch on DETACHED hard latents with one enc/dec    │
  │           epoch on reconstruction + λ·L_compat through the STE.  Control = joint at λ=0, SAME │
  │           code path (a TRACED 0.0 multiply of the FULL L_compat graph; NO python branch on λ).│
  └─────────────────────────────────────────────────────────────────────────────────────────────┘
  ┌─ CPU-UNIT-TESTED (pure logic; the Task-11 zero-compute battery proves all 6 tokens reachable)┐
  │ route_seed / route_run   the 6-token outcome router (PURE functions).                         │
  │ reject_gate / RejectState the per-update reject gates + halve-LR + stop-after-2-consecutive.  │
  │ fork_checkpoint / restore_out_of_band   the DTM.save→load×2 fork + the out-of-band restore of │
  │           autocorrelations / dtm.key / opt-state (the Task-4-confirmed mechanism).            │
  │ compat_term / compat_value_and_grad_x64   the λ-multiply + the SCOPED-x64 compat grad         │
  │           (float64 ONLY around the compat loss/grad — a global flip breaks DTM.load).         │
  └─────────────────────────────────────────────────────────────────────────────────────────────┘

The acceptance constants (ESS_min, C, L_traj, N_chains, N_R, the gain/drop thresholds, the GPU-h cap)
are FROZEN at the Task-12 local calibration into PINS; here they are an :class:`AcceptanceConstants`
the router/gate take as a PARAMETER so the logic is testable with synthetic values.

Observable ordering (design notes / Task-8): the probe returns node means in ``bias_pos`` order — the
driver consumes the probe's per-layer dicts AS-IS (it never re-orders), so the gates read exactly the
scalars the probe produced.

Generation-program refresh (design notes / Task-7): ``DTM.load`` rebuilds BOTH the negative AND the
generation programs (``_rebuild_step_interactions``), so a fork via save→load is already generation-
refreshed.  For an in-place weight edit that does NOT go through DTM.load the driver calls
:func:`refresh_all_programs` (negative + generation) before any ``.generate`` / FID.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# --- self-bootstrap: make `import htdml` work, then install the vendored path ordering ---------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from htdml.paths import bootstrap_paths  # noqa: E402

bootstrap_paths()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import htdml.compatibility as C  # noqa: E402


# ====================================================================== the 6-token vocab (companion)
# Design notes §"6-token outcome vocab".  Companion-local; NEVER an external claim-status tag.
TOKENS = (
    "BUDGET-WALL",
    "Q-CALIBRATION-FAIL",
    "PLATEAU-UNRESOLVED",
    "QUALITY-LOSS",
    "HTDML-MARGIN-NEGATIVE",
    "HTDML-MARGIN-POSITIVE",
)

# the measurement-INVALID tokens, in worst-→-best precedence (a run can't be POSITIVE unless BOTH
# seeds are measurement-valid; among invalids the worst-precedence wins).
_INVALID_PRECEDENCE = ("BUDGET-WALL", "Q-CALIBRATION-FAIL", "PLATEAU-UNRESOLVED")


# ====================================================================== acceptance constants (PINS)
@dataclass(frozen=True)
class AcceptanceConstants:
    """Frozen acceptance constants — the gate/router bars.  At runtime these are read from PINS (Task-12
    calibration); the router/gate take them as a PARAMETER so the logic is unit-testable with synthetic
    values.  The T_O/Q bias (design notes §"Half-Sokal T_O bias") is systematic → the RELATIVE bars
    (Q_GAIN, TAU_DROP, Q_DROP_MAX) cancel it; the ABSOLUTE bars (ESS_min, C) are frozen at calibration
    using the SAME biased estimator → self-consistent."""

    ESS_min: float          # window-adequacy gate: worst-layer ESS_hat ≥ ESS_min
    C: float                # trajectory-adequacy: L_traj ≥ C·τ̂ (else PLATEAU-UNRESOLVED)
    L_traj: int             # retained Y-process length (probe param)
    N_chains: int           # negative-phase chains (probe param)
    N_R: int                # fixed shared Rademacher sketches (probe param)
    Q_GAIN: float = 1.25    # improvement gate: lower-quartile joint Q ≥ Q_GAIN × control
    TAU_DROP: float = 0.25  # improvement gate (τ leg): worst-layer τ_int,Y ≥ TAU_DROP lower
    Q_DROP_MAX: float = 0.10  # reject gate: lower-quartile Q drop > Q_DROP_MAX vs control → reject
    R_GRAD50_MAX: float = 0.05  # reject + PASS gate: r_grad[50] ≤ R_GRAD50_MAX
    BCE_TOL: float = 0.05   # quality: BCE ≤ control·(1+BCE_TOL)
    FID_TOL: float = 0.10   # quality: FID ≤ control·(1+FID_TOL)
    GPU_H_CAP: float = 4.0  # the GPU-h allocation cap (BUDGET-WALL when exceeded)


# ====================================================================== per-seed metrics bundle
@dataclass
class SeedMetrics:
    """One seed's measurement: 4 joint per-layer probe dicts + 4 matched-control per-layer dicts +
    quality (BCE/FID vs control) + budget + the two measurement-validity flags.

    Each per-layer dict has the probe's headline scalars (consumed AS-IS in the probe's bias_pos order):
    ``Q_struct_perp``, ``tau_int_Y``, ``ESS_hat``, ``r_grad[50]``, ``gradient_norm``, plus the
    per-layer calibration ``cal_stable`` + ``L_traj`` + ``tau_hat`` (so trajectory adequacy can be
    recomputed: L_traj ≥ C·τ̂).
    """

    joint_layers: List[dict]
    control_layers: List[dict]
    bce: float
    fid: float
    control_bce: float
    control_fid: float
    gpu_h: float = 0.0
    budget_wall: bool = False
    cal_all_stable: bool = True      # Q-CALIBRATION-FAIL flag (T_O doubling-stability on all 4 layers)
    traj_all_resolved: bool = True   # PLATEAU-UNRESOLVED flag (L_traj ≥ C·τ̂ on all 4 layers)


# ====================================================================== per-epoch + per-reverse-layer record
@dataclass
class EpochLayerRecord:
    """The per-epoch + per-reverse-layer record the plan requires the driver to STORE (one per Stage-C
    epoch × reverse layer).  Holds every stored scalar:

      reconstruction BCE, FID, the LIVE ACP coefficient (``correlation_penalty`` read POST-``adapt_param``
      — design notes §config: ``adapt_param`` utils.py:142-149, re-floored to cp_min each adaptive epoch),
      gradient norm, r_grad[1], r_grad[50], τ_int,Y, ESS_hat, trace-Q_struct^⊥.

    ``correlation_penalty`` was MISSING from every driver record (the zero-compute battery xfail-marked
    its store-coverage); it is now a first-class field so the store-coverage is complete.
    """

    epoch: int
    layer: int                       # the reverse layer (= diffusion step, 0..3)
    bce: float                       # reconstruction BCE
    fid: float                       # FID (on the decoded 28×28)
    correlation_penalty: float       # the LIVE ACP coefficient (post-adapt_param), the missing field
    gradient_norm: float
    r_grad_1: float                  # ρ_Y(1)
    r_grad_50: float                 # ρ_Y(50) — full-window plateau sanity
    tau_int_Y: float
    ESS_hat: float
    Q_struct_perp: float             # trace-Q_struct^⊥


def make_epoch_layer_record(epoch, layer, *, bce, fid, correlation_penalty, probe_layer_dict):
    """Build an :class:`EpochLayerRecord` from a probe per-layer dict (the Task-8 ``evaluate`` output,
    consumed in bias_pos order AS-IS) + the BCE/FID/live-ACP scalars the driver tracks.

    ``correlation_penalty`` is the LIVE ACP coefficient for this reverse layer, read POST-``adapt_param``
    (i.e. ``cp_coeffs[step]`` after the adaptive update in the DTM training loop — DTM.py:354-364)."""
    pl = probe_layer_dict
    return EpochLayerRecord(
        epoch=int(epoch), layer=int(layer), bce=float(bce), fid=float(fid),
        correlation_penalty=float(correlation_penalty),
        gradient_norm=float(pl["gradient_norm"]),
        r_grad_1=float(pl["r_grad[1]"]), r_grad_50=float(pl["r_grad[50]"]),
        tau_int_Y=float(pl["tau_int_Y"]), ESS_hat=float(pl["ESS_hat"]),
        Q_struct_perp=float(pl["Q_struct_perp"]),
    )


def live_acp_coefficient(dtm, step_index, cp_coeffs):
    """Read the LIVE ACP coefficient for a reverse layer = the ``cp_coeffs[step_index]`` AFTER the
    adaptive update (``adapt_param`` + the cp_min re-floor; DTM.py:354-364).  ``cp_coeffs`` is the live
    correlation-penalty vector the DTM training loop maintains (one entry per step).  Returned as a
    plain float for the per-epoch record (the value the plan requires stored post-``adapt_param``)."""
    return float(np.asarray(cp_coeffs)[int(step_index)])


# ====================================================================== scalar helpers (pure)
def _q_values(layers) -> np.ndarray:
    return np.asarray([float(l["Q_struct_perp"]) for l in layers], dtype=np.float64)


def _lower_quartile(x: np.ndarray) -> float:
    """Lower-quartile over the 4 layers (the brief's 'lower-quartile-over-4-layers Q_struct')."""
    return float(np.quantile(np.asarray(x, dtype=np.float64), 0.25))


def _worst_ess(layers) -> float:
    """Worst (minimum) ESS_hat over the gradient-observable layers."""
    return float(min(float(l["ESS_hat"]) for l in layers))


def _worst_tau(layers) -> float:
    """Worst (maximum) τ_int,Y over the 4 layers."""
    return float(max(float(l["tau_int_Y"]) for l in layers))


def _max_r_grad50(layers) -> float:
    return float(max(float(l["r_grad[50]"]) for l in layers))


def _layer_trajectory_resolved(layer, acc: AcceptanceConstants) -> bool:
    """A layer's τ̂ is trajectory-resolved iff its retained length L_traj ≥ C·τ̂ (else unmeasurable)."""
    tau_hat = float(layer.get("tau_hat", layer.get("tau_int_Y")))
    L_traj = float(layer.get("L_traj", acc.L_traj))
    return L_traj >= acc.C * tau_hat


def _all_trajectory_resolved(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    """Both the seed-level flag AND every joint layer's own L_traj ≥ C·τ̂."""
    return bool(m.traj_all_resolved) and all(
        _layer_trajectory_resolved(l, acc) for l in m.joint_layers)


def _quality_ok(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    return (m.bce <= m.control_bce * (1.0 + acc.BCE_TOL)
            and m.fid <= m.control_fid * (1.0 + acc.FID_TOL))


def _improvement_met(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    """The HTDML improvement gate: lower-quartile joint Q ≥ Q_GAIN × (lower-quartile control Q)  OR
    worst-layer τ_int,Y ≥ TAU_DROP lower than the matched control's worst-layer τ."""
    q_joint = _lower_quartile(_q_values(m.joint_layers))
    q_ctrl = _lower_quartile(_q_values(m.control_layers))
    q_leg = q_joint >= acc.Q_GAIN * q_ctrl
    tau_joint = _worst_tau(m.joint_layers)
    tau_ctrl = _worst_tau(m.control_layers)
    tau_leg = tau_joint <= (1.0 - acc.TAU_DROP) * tau_ctrl
    return bool(q_leg or tau_leg)


def _ess_non_degraded(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    """ESS non-degradation CO-REQUIREMENT: the joint worst-layer ESS_hat must not drop below the
    matched-control worst-layer ESS_hat (catches Q inflation via T_{O,Y} shrinkage at collapsed ESS)."""
    return _worst_ess(m.joint_layers) >= _worst_ess(m.control_layers)


def _ess_adequate(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    return _worst_ess(m.joint_layers) >= acc.ESS_min


def _r_grad50_ok(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    return _max_r_grad50(m.joint_layers) <= acc.R_GRAD50_MAX


# ====================================================================== the per-seed router (PURE)
def route_seed(m: SeedMetrics, acc: AcceptanceConstants) -> str:
    """6 disjoint priority-ordered tokens for ONE seed (design notes §"Outcome router").  PURE.

    Priority (first match wins):
      1. BUDGET-WALL            — GPU-h allocation exceeded (checked FIRST, stop).
      2. Q-CALIBRATION-FAIL     — T_{O,Y} doubling-stability fails on any of the 4 layers.
      3. PLATEAU-UNRESOLVED     — calibration OK but trajectory adequacy fails (L_traj < C·τ̂).
      4. QUALITY-LOSS           — measurement valid but BCE > control+5% OR FID > control+10%.
      5. HTDML-MARGIN-NEGATIVE  — all valid + quality OK but the improvement gate NOT met.
      6. HTDML-MARGIN-POSITIVE  — all gates met.

    Returns one of :data:`TOKENS`.  The Task-11 zero-compute battery proves all 6 reachable.
    """
    # 1. BUDGET-WALL — checked first.
    if m.budget_wall or m.gpu_h > acc.GPU_H_CAP:
        return "BUDGET-WALL"
    # 2. Q-CALIBRATION-FAIL — any layer's T_O doubling-stability failed.
    if not m.cal_all_stable:
        return "Q-CALIBRATION-FAIL"
    # 3. PLATEAU-UNRESOLVED — calibration OK but a trajectory is not adequacy-resolved.
    if not _all_trajectory_resolved(m, acc):
        return "PLATEAU-UNRESOLVED"
    # 4. QUALITY-LOSS — measurement valid but quality regressed.
    if not _quality_ok(m, acc):
        return "QUALITY-LOSS"
    # 5/6. HTDML margin — the improvement gate (+ the ESS co-requirements for POSITIVE).
    if _improvement_met(m, acc) and _ess_adequate(m, acc) and _ess_non_degraded(m, acc) \
            and _r_grad50_ok(m, acc):
        return "HTDML-MARGIN-POSITIVE"
    return "HTDML-MARGIN-NEGATIVE"


def seed_passes(m: SeedMetrics, acc: AcceptanceConstants) -> bool:
    """Per-seed PASS predicate (feeds the two-seed aggregation, NOT the run verdict directly):
    BCE ≤ control+5% AND FID ≤ control+10% AND all 4 layers trajectory-resolved AND ESS-adequate
    AND r_grad[50] ≤ 0.05, AND (lower-quartile joint Q ≥ Q_GAIN×control OR worst-layer τ ≥ TAU_DROP
    lower), AND ESS_hat non-degraded.  PURE."""
    if m.budget_wall or m.gpu_h > acc.GPU_H_CAP:
        return False
    if not m.cal_all_stable:
        return False
    return bool(
        _quality_ok(m, acc)
        and _all_trajectory_resolved(m, acc)
        and _ess_adequate(m, acc)
        and _r_grad50_ok(m, acc)
        and _improvement_met(m, acc)
        and _ess_non_degraded(m, acc)
    )


def _is_measurement_valid_token(tok: str) -> bool:
    return tok not in _INVALID_PRECEDENCE


# ====================================================================== the two-seed router (PURE)
def route_run(seed_a: SeedMetrics, seed_b: SeedMetrics, acc: AcceptanceConstants) -> str:
    """Two-seed run-level aggregation (priority, disjoint, exhaustive).  PURE.

    (1) if ANY seed is measurement-invalid → the run takes that worst-precedence INVALID token
        (BUDGET-WALL > Q-CALIBRATION-FAIL > PLATEAU-UNRESOLVED); a run CANNOT be POSITIVE unless BOTH
        seeds are measurement-valid.
    (2) among measurement-valid: POSITIVE iff BOTH seeds pass all final gates.
    (3) else if ANY seed fails a quality gate → QUALITY-LOSS.
    (4) else → HTDML-MARGIN-NEGATIVE.

    Returns one of :data:`TOKENS`.
    """
    tok_a = route_seed(seed_a, acc)
    tok_b = route_seed(seed_b, acc)

    # (1) measurement-invalid → worst-precedence invalid token.
    invalids = [t for t in (tok_a, tok_b) if not _is_measurement_valid_token(t)]
    if invalids:
        for t in _INVALID_PRECEDENCE:
            if t in invalids:
                return t

    # (2) both measurement-valid: POSITIVE iff BOTH pass all final gates.
    if seed_passes(seed_a, acc) and seed_passes(seed_b, acc):
        return "HTDML-MARGIN-POSITIVE"

    # (3) any quality gate failure → QUALITY-LOSS.
    if tok_a == "QUALITY-LOSS" or tok_b == "QUALITY-LOSS":
        return "QUALITY-LOSS"

    # (4) otherwise margin-negative.
    return "HTDML-MARGIN-NEGATIVE"


# ====================================================================== per-update REJECT gate (PURE)
@dataclass(frozen=True)
class RejectDecision:
    reject: bool
    reason: Optional[str] = None


def reject_gate(joint_layers, control_layers, acc: AcceptanceConstants) -> RejectDecision:
    """Per-update reject gate (every 2 joint epochs).  Roll back + REJECT the joint candidate if ANY of:

      * lower-quartile-over-4-layers Q_struct drops > Q_DROP_MAX (10%) vs the matched control; OR
      * the worst gradient-observable layer's ESS_hat < ESS_min (window-adequacy) — PROVIDED τ̂ is
        trajectory-resolved (L_traj ≥ C·τ̂); else the read is PLATEAU-UNRESOLVED (unmeasurable); OR
      * r_grad[50] > R_GRAD50_MAX (full-window plateau sanity, absolute).

    ESS non-degradation is a CO-REQUIREMENT of any accepted Q gain.  PURE — returns a RejectDecision.

    Ordering: the trajectory-adequacy check GATES the ESS condition (an ESS read on an unresolved
    trajectory is meaningless → route to PLATEAU-UNRESOLVED, not an ESS-reject).
    """
    # Q drop > 10% (lower-quartile over 4 layers, relative — the T_O bias cancels).
    q_joint = _lower_quartile(_q_values(joint_layers))
    q_ctrl = _lower_quartile(_q_values(control_layers))
    if q_ctrl > 0 and q_joint < (1.0 - acc.Q_DROP_MAX) * q_ctrl:
        return RejectDecision(True, f"Q_drop (lower-quartile {q_joint:.4g} < "
                                    f"{(1.0 - acc.Q_DROP_MAX):.2f}×{q_ctrl:.4g})")

    # r_grad[50] absolute plateau sanity.
    r50 = _max_r_grad50(joint_layers)
    if r50 > acc.R_GRAD50_MAX:
        return RejectDecision(True, f"r_grad[50] {r50:.4g} > {acc.R_GRAD50_MAX}")

    # ESS window-adequacy — only meaningful if τ̂ is trajectory-resolved.
    worst_ess = min(float(l["ESS_hat"]) for l in joint_layers)
    if worst_ess < acc.ESS_min:
        all_resolved = all(_layer_trajectory_resolved(l, acc) for l in joint_layers)
        if not all_resolved:
            return RejectDecision(True, "PLATEAU-UNRESOLVED")
        return RejectDecision(True, f"ESS_hat {worst_ess:.4g} < ESS_min {acc.ESS_min}")

    return RejectDecision(False, None)


# ====================================================================== reject loop state (PURE)
@dataclass(frozen=True)
class RejectState:
    """The per-seed reject-loop state: the encoder LR (halved on each rejection), the CONSECUTIVE
    rejection count, and the stop flag (set after 2 consecutive rejections)."""

    encoder_lr: float
    consecutive: int = 0
    stop: bool = False


def apply_rejection(st: RejectState, max_consecutive: int = 2) -> RejectState:
    """After a rejection: HALVE the encoder LR + increment the consecutive counter.  After
    ``max_consecutive`` (=2) CONSECUTIVE rejections: set ``stop``.  PURE (returns a new RejectState)."""
    consec = st.consecutive + 1
    return RejectState(encoder_lr=st.encoder_lr * 0.5, consecutive=consec,
                       stop=(consec >= max_consecutive))


def apply_acceptance(st: RejectState) -> RejectState:
    """An ACCEPTED update resets the consecutive-rejection counter (so 'stop' needs 2 IN A ROW).
    Does NOT change the encoder LR.  PURE."""
    return RejectState(encoder_lr=st.encoder_lr, consecutive=0, stop=False)


# ====================================================================== λ-multiply + scoped-x64 compat
@contextlib.contextmanager
def _x64():
    """Scoped JAX float64 — REQUIRED ONLY around the compat loss/grad (design notes §"Compat-core
    hand-off": a GLOBAL x64 flip breaks the float32 DTM.save/load round-trip).  Restores the prior flag
    on exit so the rest of the driver (and DTM.load) is unaffected."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def compat_term(lam, clamp_spins_per_step, step_maps, beta, **mf_kwargs):
    """λ·L_compat with a TRACED λ (NO python branch on λ) + a non-finite guard.  At λ=0.0 this is
    bitwise the control (0.0 · L_compat).  Returns (value, is_finite).  This is the clean λ-multiply
    the joint update forms; control = joint at λ=0 through the SAME code path.

    Thin pass-through to :func:`htdml.compatibility.compat_loss` (which already does the traced multiply
    + the finite guard) — exposed here so the driver/tests have a single λ-multiply entry point."""
    return C.compat_loss(lam, clamp_spins_per_step, step_maps, beta, **mf_kwargs)


def build_compat_maps_x64(step):
    """Build the compat (positive-phase) maps inside a SCOPED x64 toggle (the maps read trained
    weights/biases as float64).  No global leak.  Use this from the driver instead of the bare
    ``compatibility.build_compat_maps`` so the float64 scoping is consistent."""
    with _x64():
        return C.build_compat_maps(step)


def refreshed_compat_maps_x64(step):
    """Refresh-gated compat maps (the stale-factors guard) inside the scoped x64 toggle.
    Returns (maps, proof)."""
    with _x64():
        return C.refreshed_compat_maps(step)


def compat_value_and_grad_x64(lam, clamp_spins_per_step, step_maps, beta, **mf_kwargs):
    """Compute (value, ∂(λ·L_compat)/∂clamp, is_finite) for the compat term, with float64 SCOPED ONLY
    around the loss+grad (no global x64 leak — verified by the driver test).  The gradient is over the
    full clamp matrix; in production the caller zeroes the non-image_output columns (only the
    image_output latent carries ∂L_compat/∂latent; label_output + b_t are stop_gradient'd hard draws).

    Returns
    -------
    (value: float, grad: np.ndarray (K_steps, n_clamp), is_finite: bool)
    """
    clamp = np.asarray(clamp_spins_per_step, dtype=np.float64)
    if clamp.ndim == 1:
        clamp = clamp[None, :]
    with _x64():
        clamp_j = jnp.asarray(clamp, dtype=jnp.float64)

        def loss(cl):
            val, _fin = C.compat_loss(lam, cl, step_maps, beta, **mf_kwargs)
            return val

        val, grad = jax.value_and_grad(loss)(clamp_j)
        is_finite = bool(jnp.isfinite(val) and jnp.all(jnp.isfinite(grad)))
        out_val = float(val)
        out_grad = np.asarray(grad, dtype=np.float64)
    return out_val, out_grad, is_finite


# ====================================================================== generation-program refresh
def refresh_all_programs(step):
    """Refresh BOTH the negative AND the generation programs from the CURRENT trained globals after an
    in-place weight edit (design notes §"GENERATION-program refresh gap").  A fork via DTM.save→load is
    ALREADY generation-refreshed (DTM.load calls ``_rebuild_step_interactions`` over training + gen);
    this is for an ad-hoc ``eqx.tree_at`` weight edit that bypasses DTM.load.

    Returns a NEW step with all four programs (program_positive/negative, program_free/conditioned)
    re-wired to step.model.weights/biases.  Asserts the refresh proof on the negative program first.
    """
    import equinox as eqx

    from thrmlDenoising.sampling_specs import get_new_per_block_interactions

    w, b = step.model.weights, step.model.biases
    new_pos = get_new_per_block_interactions(step.training_spec.program_positive, w, b)
    new_neg = get_new_per_block_interactions(step.training_spec.program_negative, w, b)
    new_free = get_new_per_block_interactions(step.generation_spec.program_free, w, b)
    new_cond = get_new_per_block_interactions(step.generation_spec.program_conditioned, w, b)
    return eqx.tree_at(
        lambda s: (s.training_spec.program_positive.per_block_interactions,
                   s.training_spec.program_negative.per_block_interactions,
                   s.generation_spec.program_free.per_block_interactions,
                   s.generation_spec.program_conditioned.per_block_interactions),
        step, (new_pos, new_neg, new_free, new_cond))


# ====================================================================== fork / out-of-band restore
@dataclass
class ArmState:
    """The unsaved static a fork must carry OUT-OF-BAND past DTM.load (which drops it): the dtm.key and
    the per-step ``autocorrelations`` dicts (DTM.load returns {}).  ``opt_state`` IS in the save-mask
    and IS restored by DTM.load; we capture its counts for a provenance assertion only."""

    key: object                                   # dtm.key (jax PRNGKey array)
    autocorrelations: List[dict]                  # one dict per step (deep-copied)
    opt_counts: List[List[int]] = field(default_factory=list)


def capture_arm_state(dtm) -> ArmState:
    """Snapshot the unsaved static of a DTM (its key + per-step autocorrelations) so a fork can re-inject
    it after DTM.load.  The autocorrelations dicts are SHALLOW-copied per step (values are arrays, not
    mutated in place)."""
    from harness import probe_primitives as pp

    autocorr = [dict(getattr(s, "autocorrelations", {}) or {}) for s in dtm.steps]
    opt_counts = [pp._find_counts(s.opt_state) for s in dtm.steps]
    return ArmState(key=dtm.key, autocorrelations=autocorr, opt_counts=opt_counts)


def restore_out_of_band(loaded_dtm, parent: ArmState) -> None:
    """Out-of-band restore on a freshly DTM.load'd arm (DTM.load drops the unsaved static).  Restores:

      (1) the per-step ``autocorrelations`` dicts (DTM.load returns {}; re-inject the parent's — else
          ACP ``adapt_param`` diverges between arms);
      (2) ``dtm.key`` (DTM.load builds a fresh DTM whose key is re-seeded from cfg → not the parent's);

    ``opt_state`` (incl. the LR-schedule count) IS in the save-mask and IS restored by DTM.load; the
    cosine-schedule CLOSURE is rebuilt count-consistently from cfg (same ``decay_steps =
    n_epochs_for_lrd · n_batches_per_epoch``) by the fresh DTM's constructor — so no out-of-band
    opt-state work is needed here beyond the provenance check the fork does.  MUTATES ``loaded_dtm``
    in place (DTM is a plain class, not an eqx.Module → direct field assignment is the supported path).
    """
    # (2) key
    loaded_dtm.key = parent.key
    # (1) per-step autocorrelations — re-inject the parent's dicts (replace the {} DTM.load gives).
    for i, step in enumerate(loaded_dtm.steps):
        payload = parent.autocorrelations[i] if i < len(parent.autocorrelations) else {}
        # `autocorrelations` is a mutable dict attribute on the DiffusionStep — clear + update in place
        # (so any view aliasing is broken and the arm owns its own dict).
        try:
            step.autocorrelations.clear()
            step.autocorrelations.update(dict(payload))
        except AttributeError:
            # if it is a non-dict static (shouldn't happen for DiffusionStep), set via eqx.tree_at.
            import equinox as eqx
            loaded_dtm.steps[i] = eqx.tree_at(lambda s: s.autocorrelations, step, dict(payload))


# ====================================================================== Stage-B checkpoint (RESUME_FROM)
# Persist the post-Stage-B state so a re-run can skip Stage A+B (run 5b9cbbc lost its trained DTM — the
# only on-disk save was fork_checkpoint, AFTER the cal gate it never passed).  These are PURE, CPU-testable
# helpers (plain arrays/dicts/pytrees); the GPU-binding persist/load (RealOps) reuses them + DTM.save_epoch/
# DTM.load + capture_arm_state/restore_out_of_band.  MUST run OUTSIDE any _x64() scope (float32 templates).
_CKPT_SCHEMA = "htdml-stage-b-checkpoint/v1"


def save_arm_state_to_disk(armstate: "ArmState", ae_params, manifest: dict, dirpath: str) -> None:
    """Serialize the OUT-OF-BAND fork static (dtm.key + per-step autocorrelations + opt_counts), the
    encoder Flax params, and the manifest into ``dirpath``.  The DTM weights are written separately by
    DTM.save_epoch (see RealOps.persist_stage_b)."""
    import json
    import flax.serialization as fser
    assert not jax.config.jax_enable_x64, "save_arm_state_to_disk must run under float32 (no _x64 scope)"
    os.makedirs(dirpath, exist_ok=True)
    np.save(os.path.join(dirpath, "dtm_key.npy"), np.asarray(armstate.key))
    # autocorrelations is a ragged List[dict] (epoch→scalar) — JSON with scalar coercion (NOT np.save).
    autocorr = [{str(k): float(np.asarray(v)) for k, v in d.items()} for d in armstate.autocorrelations]
    with open(os.path.join(dirpath, "autocorrelations.json"), "w") as f:
        json.dump(autocorr, f)
    with open(os.path.join(dirpath, "opt_counts.json"), "w") as f:
        json.dump([list(c) for c in armstate.opt_counts], f)
    with open(os.path.join(dirpath, "ae_params.msgpack"), "wb") as f:
        f.write(fser.to_bytes(ae_params))
    with open(os.path.join(dirpath, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _coerce_epoch_key(k):
    """Autocorrelation epoch keys are ints in production (step.py) but JSON forces them to strings —
    restore the int where numeric, else keep the string (defensive)."""
    try:
        return int(k)
    except (TypeError, ValueError):
        return k


def load_arm_state_from_disk(dirpath: str, ae_template):
    """Inverse of save_arm_state_to_disk → (ArmState, ae_params, manifest).  ``ae_template`` is a fresh
    BinaryAutoencoder().init(...) pytree giving the msgpack target treedef/dtypes.  MUST run OUTSIDE any
    _x64() scope.  Raises FileNotFoundError on a missing file (the resume path gates with
    require_checkpoint first; this is the loader, not the completeness guard)."""
    import json
    import flax.serialization as fser
    assert not jax.config.jax_enable_x64, "load_arm_state_from_disk must run under float32 (no _x64 scope)"
    key = np.load(os.path.join(dirpath, "dtm_key.npy"))
    with open(os.path.join(dirpath, "autocorrelations.json")) as f:
        autocorr = [{_coerce_epoch_key(k): float(v) for k, v in d.items()} for d in json.load(f)]
    with open(os.path.join(dirpath, "opt_counts.json")) as f:
        opt_counts = [list(c) for c in json.load(f)]
    with open(os.path.join(dirpath, "ae_params.msgpack"), "rb") as f:
        ae_params = fser.from_bytes(ae_template, f.read())
    with open(os.path.join(dirpath, "manifest.json")) as f:
        manifest = json.load(f)
    return ArmState(key=key, autocorrelations=autocorr, opt_counts=opt_counts), ae_params, manifest


def require_checkpoint(dirpath: str) -> None:
    """FAIL-CLOSED completeness guard for a RESUME_FROM Stage-B checkpoint: every artifact (arm-state
    files + the DTM epoch dir) must exist, else raise FileNotFoundError.  Raised from load_stage_b so it
    propagates out of run_one_seed (NOT a BudgetWall) — a missing/partial checkpoint aborts the run loudly,
    never silently retrains."""
    needed = ["manifest.json", "dtm_key.npy", "autocorrelations.json", "opt_counts.json",
              "ae_params.msgpack", os.path.join("dtm", "model_saving", "epoch_000")]
    for rel in needed:
        if not os.path.exists(os.path.join(dirpath, rel)):
            raise FileNotFoundError(f"Stage-B checkpoint incomplete: missing {rel} under {dirpath}")


def verify_manifest(manifest: dict, *, expect_seed: int, expect_raw_sha: str,
                    expect_mode: Optional[str] = None, expect_n_train: Optional[int] = None) -> None:
    """HARD-RAISE on a checkpoint/run mismatch that would invalidate the deterministic latent re-encode.
    MANDATORY: seed + raw-split-sha (latents would differ).  OPTIONAL (checked only when provided):
    mode + n_train — resuming a smoke checkpoint under MODE=full (or vice versa) re-encodes a DIFFERENT
    n_train → silently-wrong latents; the guard fails it closed.  Git-SHA is DELIBERATELY not checked:
    the cal-gate fix legitimately changes code; the caller records both SHAs in the run's provenance."""
    if int(manifest.get("seed", -1)) != int(expect_seed):
        raise RuntimeError(f"checkpoint seed {manifest.get('seed')} != requested {expect_seed}")
    got = manifest.get("raw_split_sha256")
    if got != expect_raw_sha:
        raise RuntimeError(f"raw-split sha mismatch: checkpoint {got} != current {expect_raw_sha} "
                           f"(dataset drift → latents would differ; refusing resume)")
    if expect_mode is not None and manifest.get("mode") != expect_mode:
        raise RuntimeError(f"checkpoint mode {manifest.get('mode')} != current {expect_mode} "
                           f"(different config → wrong latents; refusing resume)")
    if expect_n_train is not None and int(manifest.get("config", {}).get("n_train", -1)) != int(expect_n_train):
        raise RuntimeError(f"checkpoint n_train {manifest.get('config', {}).get('n_train')} != current "
                           f"{expect_n_train} (different split size → wrong latents; refusing resume)")


def fork_checkpoint(dtm, workdir: str, *, epoch: int = 0) -> Tuple[object, object]:
    """Fork a Stage-B checkpoint into a matched control arm + a joint arm.

    Mechanism (design notes §"DTM.load drops autocorrelations" + Task-4 (e)):
      1. capture the parent's unsaved static (key + per-step autocorrelations);
      2. ``DTM.save_epoch`` ONCE (eqx-partitions weights/biases/opt_state + serialises);
      3. ``DTM.load`` TWICE (control + joint) — each load builds a FRESH DTM, rebuilds BOTH the training
         AND the generation program weight VIEWS (``_rebuild_step_interactions``), restores
         weights/biases/opt_state from the save-mask, but returns ``autocorrelations == {}`` and a fresh
         key;
      4. for EACH arm, :func:`restore_out_of_band` re-injects the parent's autocorrelations + key.

    Both arms share the restored ``dtm.key`` (they start from the identical parent state); the probe uses
    an INDEPENDENT fixed diagnostic key shared across arms (the driver supplies that to the probe — it is
    NOT derived from ``dtm.key``).  The two arms are DISTINCT objects (mutating one's autocorrelations
    does not touch the other — required so ACP does not couple the arms).

    Returns ``(control_dtm, joint_dtm)``.  CPU-tested on a real perturbed 4_4 DTM (no dtm.train).
    """
    from thrmlDenoising.DTM import DTM

    parent = capture_arm_state(dtm)

    dtm.logging_and_saving_dir = workdir
    dtm.save_epoch(epoch)
    base = os.path.join(workdir, "model_saving")
    if not os.path.isdir(os.path.join(base, f"epoch_{epoch:03d}")):
        raise RuntimeError(f"save_epoch did not write epoch_{epoch:03d} under {base}")

    control_dtm = DTM.load(base, epoch=epoch)
    joint_dtm = DTM.load(base, epoch=epoch)

    restore_out_of_band(control_dtm, parent)
    restore_out_of_band(joint_dtm, parent)

    # provenance: weights + opt-counts must match the parent after restore (the save-mask round-trip).
    from harness import probe_primitives as pp
    for arm in (control_dtm, joint_dtm):
        for i, st in enumerate(arm.steps):
            if pp._weights_hash(st) != pp._weights_hash(dtm.steps[i]):
                raise RuntimeError(f"fork weights mismatch at step {i} after DTM.load")
            if pp._find_counts(st.opt_state) != parent.opt_counts[i]:
                raise RuntimeError(f"fork opt-state counts mismatch at step {i} after DTM.load")

    return control_dtm, joint_dtm


# ====================================================================== Stage A/B/C (GPU-wired)
# The TRAINING below HARD-REQUIRES a GPU (dtm.train; AE optimisation at scale) — it is WIRED here and
# exercised only at the Task-12 smoke.  The driver's LOGIC (router/gates/fork) is what is CPU-tested.

def stage_a_pretrain(ae_module, ae_params, train_images, *, key, n_steps, batch_size, lr,
                     loss_kwargs=None):
    """Stage A — pretrain the BinaryAutoencoder on ``stage_a_loss`` (BCE + commitment + balance).

    GPU-wired (smoke-deferred).  Returns ``(ae_params, opt_state, history)``.  ``ae_module`` carries
    ``stage_a_loss`` (the Task-6 autoencoder); an optax adam over the AE params is used.  The encoder
    LR-schedule is a plain adam(lr) — the per-update reject loop halves THIS lr by rebuilding the
    optimizer (see :func:`rebuild_encoder_optimizer`).
    """
    import optax

    from htdml import autoencoder as AE

    loss_kwargs = loss_kwargs or {}
    optim = optax.adam(lr)
    opt_state = optim.init(ae_params)
    history = []

    @jax.jit
    def _update(params, opt_state, x):
        (loss, aux), grads = jax.value_and_grad(
            lambda p: AE.stage_a_loss(p, x, **loss_kwargs), has_aux=True)(params)
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    n = train_images.shape[0]
    for s in range(int(n_steps)):
        key, kb = jr.split(key)
        idx = jr.randint(kb, (int(batch_size),), 0, n)
        x = jnp.asarray(train_images)[idx]
        ae_params, opt_state, loss, aux = _update(ae_params, opt_state, x)
        history.append(dict(step=s, loss=float(loss), **{k: float(v) for k, v in aux.items()}))
    return ae_params, opt_state, history


def stage_b_train_latent_dtm(latent_dtm, latent_dataset, *, n_epochs, evaluate_every):
    """Stage B — freeze the encoder, train the LatentDTM on the (pre-encoded) hard latents with ACP.

    GPU-wired (smoke-deferred): delegates to ``LatentDTM.fit`` (which injects the seam-A latent dict and
    runs ``dtm.train`` — GPU-only, asserts the reversible kernel is live).  The encoder was frozen +
    the split encoded ONCE upstream (the driver passes the already-built ``latent_dataset``).
    """
    return latent_dtm.fit(latent_dataset, n_epochs=n_epochs, evaluate_every=evaluate_every)


def rebuild_encoder_optimizer(lr):
    """Rebuild the encoder's optimizer at a new LR (the 'halve-encoder-LR = rebuild the encoder
    optimizer's schedule' step — used by the reject loop).  Returns a fresh optax adam(lr)."""
    import optax

    return optax.adam(lr)


def assemble_compat_clamp(b0_latent, label_clamp, bt_clamp, n_img):
    """Assemble the per-step compat clamp matrix per the Task-5 clamp-ordering contract
    ``[image_output (n_img), label_output (n_lab), b_t (n_bt)]`` (image_output FIRST), with the GRADIENT
    flowing ONLY through the image_output columns:

      * columns ``[0 : n_img)``        = ``b0_latent`` — the encoded hard latent (gradient-carrying via
                                          the STE; only b0 carries ∂L_compat/∂latent);
      * columns ``[n_img : n_img+n_lab)`` = ``stop_gradient(label_clamp)`` (label_output, hard);
      * columns ``[n_img+n_lab : )``   = ``stop_gradient(bt_clamp)`` (= stop_gradient(forward_noise(b0)),
                                          hard draw).

    ``b0_latent`` is ``(n_img,)`` (one row, tiled over K_steps) or ``(K_steps, n_img)``; ``label_clamp``/
    ``bt_clamp`` are the remaining clamp columns (``(n_lab+n_bt,)`` or ``(K_steps, n_lab+n_bt)``).  Returns
    ``(K_steps, n_clamp)`` with the image_output block carrying the only live gradient.  This MUST be built
    INSIDE the differentiated loss (so ``∂clamp/∂b0`` is live) — see :func:`joint_update_step`."""
    b0 = jnp.asarray(b0_latent)
    tail = jax.lax.stop_gradient(jnp.asarray(label_clamp))
    bt = jax.lax.stop_gradient(jnp.asarray(bt_clamp))
    if b0.ndim == 1:
        b0 = b0[None, :]
    rest = jnp.concatenate([tail, bt], axis=-1)
    if rest.ndim == 1:
        rest = rest[None, :]
    if rest.shape[0] == 1 and b0.shape[0] > 1:
        rest = jnp.broadcast_to(rest, (b0.shape[0], rest.shape[-1]))
    if b0.shape[0] == 1 and rest.shape[0] > 1:
        b0 = jnp.broadcast_to(b0, (rest.shape[0], b0.shape[-1]))
    assert b0.shape[-1] == n_img, f"b0 latent width {b0.shape[-1]} != n_img {n_img}"
    return jnp.concatenate([b0, rest], axis=-1)            # (K_steps, n_clamp), image_output first


def compat_steering_loss(ae_params, x_batch, label_clamp, bt_clamp, step_maps, beta, lam,
                         *, n_img, encode_fn=None):
    """The λ·L_compat-ONLY steering loss (no reconstruction term), x64-SCOPED, with the encode→clamp
    wiring INSIDE so the gradient reaches ``ae_params`` through the image_output latent.  Returns
    ``(value, is_finite)``.  Used by :func:`joint_update_step` (added to reconstruction) AND by the
    driver steering TEST (in isolation, to verify ∂≠0 at λ>0 / =0 at λ=0).

    ``∂(λ·L_compat)/∂ae_params``:  flows via ``b0 = encode(ae_params, x_batch)`` → image_output clamp
    columns → L_compat.  NON-ZERO at λ>0 (the compat STEERS the encoder), EXACTLY ZERO at λ=0 (the
    traced-0.0 multiply zeroes the whole graph — the control).  label_output + b_t are stop_gradient'd
    HARD draws (they do NOT carry ∂/∂ae_params).  Float64 scoped ONLY here (no global leak).

    ``encode_fn(params, x) -> (b0, logits)`` defaults to the production ``autoencoder.encode`` (196-wide
    latent, GPU smoke scale); a TEST may inject an ``n_img``-matched differentiable encoder so the
    steering property is exercised on a small CPU DTM whose image_output block is narrow."""
    if encode_fn is None:
        from htdml import autoencoder as AE
        encode_fn = AE.encode

    with _x64():
        b0, _logits = encode_fn(ae_params, x_batch)        # (B, n_img) hard latent {−1,+1}, STE-grad
        # one representative latent row drives the single-input compat clamp (reference phase_data_1 pattern):
        # the compat clamp is a single conditioned input; use the batch-mean latent so the gradient
        # reaches every encoded example (a faithful single-clamp surrogate; the smoke may select a row).
        b0_row = jnp.mean(jnp.asarray(b0), axis=0)          # (n_img,) — carries ∂/∂ae_params via the STE
        clamp = assemble_compat_clamp(b0_row, label_clamp, bt_clamp, n_img)
        val, is_finite = C.compat_loss(lam, clamp, step_maps, beta)
    return val, is_finite


def joint_update_step(ae_module, ae_params, ae_opt_state, ae_optim, step,
                      label_clamp=None, bt_clamp=None, beta=1.0, lam=0.0, x_batch=None, *,
                      loss_kwargs=None, step_maps=None, encode_fn=None):
    """ONE Stage-C joint enc/dec update: reconstruction + λ·L_compat through the STE.

    The compat clamp is ALWAYS assembled INSIDE the differentiated loss as [image_output (= b0 =
    encode(params, x_batch), carries ∂L_compat/∂ae_params via the STE), label_output (stop_gradient(
    label_clamp)), b_t (= stop_gradient(bt_clamp) = stop_gradient(forward_noise(b0)))] — the Task-5 clamp
    ordering contract.  The encode→clamp wiring MUST be inside the loss, else ``∂(λ·L_compat)/∂ae_params
    ≡ 0`` for all λ and Stage C is inert (the joint and control arms would produce IDENTICAL encoder
    updates — the vacuous-experiment bug).  There is NO constant-clamp path: a non-steering compat half
    produces no error/warning, so it must not be reachable.  ``encode_fn`` defaults to the production
    ``autoencoder.encode``; a test may inject an ``n_img``-matched encoder.

    Control = this SAME function at λ=0 (a TRACED 0.0 multiply of the full L_compat graph; NO python
    branch on λ) → the compat half contributes exactly 0 to the gradient, so the λ=0 update is the pure
    reconstruction update == the control.

    GPU-wired (smoke-deferred — `encode` at the production 196-latent / 44_12 scale is the smoke's job).
    ``step_maps`` is the refresh-gated, x64-scoped positive-phase maps for this diffusion step (built via
    :func:`step_maps_for`); if ``None`` it is built here.  Returns ``(ae_params, ae_opt_state, aux)``.

    NOTE: the FULL Stage-C alternation (one DTM epoch on DETACHED latents, then this enc/dec epoch) is
    wired by the smoke driver; this is the differentiable enc/dec half (the λ-traced compat steering).
    """
    import optax

    from htdml import autoencoder as AE

    loss_kwargs = loss_kwargs or {}
    if step_maps is None:
        step_maps = step_maps_for(step)
    n_img = int(step_maps[0]["n_img"])

    def _loss(params):
        # reconstruction half (STE forward = hard latent; backward = tanh surrogate) — encodes INSIDE.
        recon, aux = AE.stage_a_loss(params, x_batch, **loss_kwargs)
        # compat half: λ·L_compat with the encode→clamp wiring INSIDE (image_output = encoded b0 carries
        # the gradient).  A CONSTANT clamp would zero ∂(λ·L_compat)/∂ae_params — that path is removed.
        compat_val, _fin = compat_steering_loss(params, x_batch, label_clamp, bt_clamp,
                                                step_maps, beta, lam, n_img=n_img, encode_fn=encode_fn)
        return recon + compat_val, aux

    with _x64():
        (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(ae_params)
        updates, ae_opt_state = ae_optim.update(grads, ae_opt_state, ae_params)
        ae_params = optax.apply_updates(ae_params, updates)
    return ae_params, ae_opt_state, dict(loss=float(loss), **{k: float(v) for k, v in aux.items()})


def step_maps_for(step):
    """Refresh-gated, x64-scoped compat maps for a single diffusion step (one-step convenience wrapper
    over :func:`refreshed_compat_maps_x64`; returns a length-1 list to feed ``L_compat``)."""
    maps, _proof = refreshed_compat_maps_x64(step)
    return [maps]
