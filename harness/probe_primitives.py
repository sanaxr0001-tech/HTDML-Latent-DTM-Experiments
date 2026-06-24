"""Shared primitive layer for the htdml-latent-dtm companion — sokal / energy / refresh ports + K=50.

This module is the building block used by the 6_4 fixture (Task 4), the compat-core (Task 4), the
TrainabilityProbe (Task 6), and the driver fork/restore (Task 7). It has TWO parts:

  PART A — VERBATIM-faithful ports of the wiki exp15 / exp15-recheck / exp19 harness primitives. The
  math is already validated upstream; it is ported here UNCHANGED (only imports/bootstrap adapted —
  no path/cd/JSON-runner scaffolding). Provenance of each port is noted at its definition. Sources:
    * `…/experiments/internal-exp/pt_p0_calibrate.py`
        - `_rho_block` (:142), `_tau_half_from_rho` (:155), `_obs_chunk` (:167),
          `sokal_profile_from_spins` (:180)           — the half-Sokal τ_int / T_O estimator,
        - `build_maps` (:257), `energy_free` (:296)    — per-block maps + the THREE-term conditional,
        - `_find_counts` (:202), `_weights_hash` (:233), `_key_list` (:240)  — rollback / provenance.
    * `…/experiments/internal-exp/recheck.py` (:74-129) + `internal_exp.py` (:186-242)
        - the MANDATORY trained-weight refresh (the exp15/16 init-weight bug fix): a standalone
          `refresh_program_weights(prog, step)` + the HARD-HALT `refreshed_weight_proof(step)`.

  The PT-ladder functions (build_alpha_programs, pt_super_sweep, pt_traj, measure_swap_accept,
  select_ladder, classify_curve, _rt_account) are DELIBERATELY NOT ported — the companion uses a
  SINGLE-replica reversible kernel, no PT ladder. We reuse only the *refresh idea* that lived inside
  `build_alpha_programs_refreshed`, lifted out as the standalone `refresh_program_weights`.

  PART B — the companion's NEW K=50 retained-Y-process + Rademacher-sketch layer (build-notes
  §"Probe K=50 convention", E1/E6). Y_i = X_{B+i·s}, B=400, s=8, i=1..K, K=50. τ_int,Y (half-Sokal on
  the retained process) is the PRIMARY mixing-margin scalar; derived ESS_hat / Q_struct^⊥ / r_grad
  diagnostics; a fixed deterministic Rademacher (±1) sketch with a worst-of-N_R screening reduction.

`import htdml` (bootstrap_paths) runs on import so `thrml`/`thrmlDenoising` resolve to the vendored
copies, not conda site-packages.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- self-bootstrap: make `import htdml` work, then install the vendored path ordering ---------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from htdml.paths import bootstrap_paths  # noqa: E402

bootstrap_paths()

import hashlib  # noqa: E402

import numpy as np  # noqa: E402

# ============================================================ frozen estimator constants (exp15 §1)
# Lifted verbatim from pt_p0_calibrate.py (estimator hygiene, NOT scientific bars).
OBS_CHUNK = 2048
TAU_CHUNK = 4096

# Doubling-stability Cal-STABLE thresholds — VERBATIM from exp15 pt_p0_calibrate.py:70-72 (the SAME
# constants exp16 imports as P15.SOKAL_C / TAU_TOL / STAB_TOL). The Cal-STABLE boolean the driver's
# Q-CALIBRATION-FAIL gate reads MUST use these (NOT a laxer ad-hoc criterion).
SOKAL_C = 5.0      # half-Sokal self-consistency: L >= SOKAL_C*tau_max (NOT the tracking c=3)
TAU_TOL = 0.15     # tau_max doubling stability: |dtau_max|/tau_max < TAU_TOL
STAB_TOL = 0.15    # aggregate-T_O Cal-STABLE: |dT_O|/T_O AND ||dS||_1/sum(S) BOTH < STAB_TOL

# ============================================================ companion K=50 convention (build-notes §E1)
K_WINDOW = 50          # window_samples_K — retained samples per window (= n_samples upstream)
B_WARMUP = 400         # steps_warmup — burn-in sweeps discarded before retention
STRIDE_SWEEPS = 8      # steps_per_sample — sweeps between retained samples (= window_span/K)


# ================================================================ PART A.1 — half-Sokal estimators
# VERBATIM from exp15 pt_p0_calibrate.py:142-198 (math unchanged; only docstrings trimmed).
def _rho_block(block):
    """Normalised autocorrelation ρ(ℓ) per observable column, averaged over chains (FFT acov / acov0).
    block: (n_chains, L, b). Returns (L, b). VERBATIM exp15 pt_p0_calibrate.py:142."""
    n_chains, L, b = block.shape
    x = block - block.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < 2 * L:
        nfft *= 2
    fx = np.fft.rfft(x, n=nfft, axis=1)
    acov = np.fft.irfft(fx * np.conj(fx), n=nfft, axis=1)[:, :L, :].real / L
    acov = acov.mean(axis=0)
    a0 = acov[0:1, :]
    return np.divide(acov, a0, out=np.zeros_like(acov), where=a0 > 0)


def _tau_half_from_rho(rho):
    """half-Sokal integrated autocorrelation time per column: τ = ½ + Σ positive ρ-pairs (truncated at
    the first non-positive pair). rho: (L, b). Returns (b,). VERBATIM exp15 pt_p0_calibrate.py:155."""
    L, b = rho.shape
    npair = (L - 1) // 2
    if npair <= 0:
        return np.full(b, 0.5)
    i1 = 1 + 2 * np.arange(npair)
    i2 = 2 + 2 * np.arange(npair)
    pairs = rho[i1, :] + rho[i2, :]
    keep = np.cumprod((pairs > 0).astype(np.int8), axis=0).astype(bool)
    return 0.5 + np.sum(np.where(keep, pairs, 0.0), axis=0)


def _obs_chunk(spins, maps, lo, hi):
    """Build the observable block f_a for columns [lo,hi): edge products then bias spins (exp4 ordering).
    Never materializes the full f. VERBATIM exp15 pt_p0_calibrate.py:167."""
    n_edge = maps["n_edge"]
    cols = []
    e_lo, e_hi = max(lo, 0), min(hi, n_edge)
    if e_hi > e_lo:
        cols.append(spins[:, :, maps["edge_pos0"][e_lo:e_hi]] * spins[:, :, maps["edge_pos1"][e_lo:e_hi]])
    b_lo, b_hi = max(lo - n_edge, 0), max(hi - n_edge, 0)
    if b_hi > b_lo:
        cols.append(spins[:, :, maps["bias_pos"][b_lo:b_hi]])
    return np.concatenate(cols, axis=-1) if len(cols) > 1 else cols[0]


def sokal_profile_from_spins(spins, maps):
    """Stream the FULL per-observable half-Sokal profile; return (tau_max, T_O, S_a) where
    S_a = 2·τ_int_a·Var_a (estimated Var; the DTM is non-enumerable), T_O = ½·Σ_a S_a, and
    tau_max = max_a τ_int_a (window-sizing scalar). Never materializes the full f.
    THE core mixing-time estimator. VERBATIM exp15 pt_p0_calibrate.py:180."""
    P = maps["n_edge"] + maps["n_bias"]
    S_a = np.empty(P, dtype=np.float64)
    tau_max = 0.0
    for lo in range(0, P, OBS_CHUNK):
        hi = min(lo + OBS_CHUNK, P)
        fc = _obs_chunk(spins, maps, lo, hi)
        for s in range(0, fc.shape[-1], TAU_CHUNK):
            e = min(s + TAU_CHUNK, fc.shape[-1])
            sub = fc[:, :, s:e]
            tau = _tau_half_from_rho(_rho_block(sub))              # (chunk,)
            var = sub.reshape(-1, sub.shape[-1]).var(axis=0)       # (chunk,) estimated Var over (chain,time)
            S_a[lo + s:lo + e] = 2.0 * tau * var
            tau_max = max(tau_max, float(np.max(tau)))
    T_O = 0.5 * float(S_a.sum())
    return tau_max, T_O, S_a


# ================================================================ PART A.1b — Cal-STABLE classifier
# FAITHFUL port of exp15 pt_p0_calibrate.py:540-568 `classify_curve` STABLE-step + 2-consecutive logic
# (the Cal-STABLE leg only — the P0 ladder's RESOLVED/UNRESOLVED/WALL three-way split + LINEAR-growth
# axis are NOT needed for the companion's Q-CALIBRATION-FAIL gate, which is a binary cal_stable). The
# per-rung STABLE test is exp16's exact THREE-axis criterion; Cal-STABLE requires TWO CONSECUTIVE
# STABLE rungs. A chain whose T_O is still drifting (large dT) but whose L1-normalized S_a shape
# momentarily stabilizes on a SINGLE rung is correctly reported NOT-stable (the failure mode the laxer
# single-rung / dS_l1-only criterion missed).
def classify_calibration_stable(curve):
    """Return (cal_stable: bool, curve_annotated: list, failed_axis: list) for a doubling curve.

    `curve` is a list of rung dicts (in doubling order) each carrying:
      tau_max, T_O, self_consistent (bool, = L >= SOKAL_C*tau_max), dS_l1 (float | None for rung 0).
    A rung i>=1 is a STABLE step iff ALL THREE axes hold (exp15 classify_curve:550-551):
      * tau_stable : rel_tau = |tau_max[i]-tau_max[i-1]| / tau_max[i-1] < TAU_TOL  AND self_consistent[i]
      * TO_stable  : dT = |T_O[i]-T_O[i-1]| / T_O[i]               < STAB_TOL  AND dS_l1[i] < STAB_TOL
    cal_stable is True iff TWO CONSECUTIVE rungs are STABLE (consec_stable >= 2, exp15:559). Annotates
    each rung with step_class ('STABLE'/'NOT-STABLE') + the per-axis residuals; reports which axis (or
    axes) were unstable on the FIRST non-stable rung after any stable run, for the driver's diagnostics.
    """
    annotated = [dict(c) for c in curve]
    consec_stable = 0
    cal_stable = False
    failed_axis = []
    for i in range(1, len(annotated)):
        c, p = annotated[i], annotated[i - 1]
        rel_tau = abs(c["tau_max"] - p["tau_max"]) / max(p["tau_max"], 1e-9)
        dT = abs(c["T_O"] - p["T_O"]) / max(c["T_O"], 1e-9)
        dS_l1 = c["dS_l1"] if c.get("dS_l1") is not None else 1.0   # rung-0 / missing => non-stable
        tau_stable = bool((rel_tau < TAU_TOL) and c["self_consistent"])
        TO_stable = bool((dT < STAB_TOL) and (dS_l1 < STAB_TOL))    # BOTH dT and dS_l1 axes
        c["rel_tau"] = float(rel_tau)
        c["dT"] = float(dT)
        if tau_stable and TO_stable:
            c["step_class"] = "STABLE"
            consec_stable += 1
        else:
            c["step_class"] = "NOT-STABLE"
            # record which axis/axes drove the instability (matches exp15 classify_curve:563-566)
            if (not c["self_consistent"]) or rel_tau >= TAU_TOL:
                failed_axis.append("tau_hat")
            if dT >= STAB_TOL or dS_l1 >= STAB_TOL:
                failed_axis.append("aggregate_T_O")
            consec_stable = 0
        if consec_stable >= 2:
            cal_stable = True
            break
    return cal_stable, annotated, sorted(set(failed_axis))


# ================================================================ PART A.2 — provenance / rollback
# VERBATIM from exp15 pt_p0_calibrate.py:202-241.
def _find_counts(tree):
    """Collect every `count` leaf in an opt_state-like tree (provenance: opt_count == t·n_batches).
    VERBATIM exp15 pt_p0_calibrate.py:202."""
    out = []

    def rec(o):
        if hasattr(o, "_fields"):
            for f in o._fields:
                v = getattr(o, f)
                if f == "count":
                    try:
                        out.append(int(np.asarray(v).ravel()[0]))
                    except Exception:
                        pass
                else:
                    rec(v)
        elif isinstance(o, dict):
            for v in o.values():
                rec(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                rec(v)

    rec(tree)
    return out


def _weights_hash(step):
    """16-hex sha1 of (step.model.weights, step.model.biases) — the rollback / weights-distinct check.
    VERBATIM exp15 pt_p0_calibrate.py:233."""
    h = hashlib.sha1()
    for arr in (step.model.weights, step.model.biases):
        h.update(np.ascontiguousarray(np.asarray(arr), dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _key_list(dtm):
    """dtm.key as a plain int list — the probe-RNG isolation invariant. VERBATIM exp15:240."""
    return [int(x) for x in np.asarray(dtm.key).ravel()]


# ================================================================ PART A.3 — build_maps + energy_free
# VERBATIM from exp15 pt_p0_calibrate.py:257-309 (per-block interaction/observable maps + the
# THREE-term free-conditional energy). The negative-phase program partition is read exactly as exp15
# did: FREE = program_negative.free_blocks (4 superblocks), CLAMPED = program_negative.clamped_blocks
# (= b_t). Build-notes §"Training-negative free set — CONFIRMED" pins this for the companion too.
def build_maps(step):
    """Build the per-block interaction/observable maps for the gradient observables
    f_a = {edge products s_e0·s_e1, node spins s_n}, the trained per-edge/per-node weights aligned to
    that ordering, and the input↔output COUPLING edges (load-bearing for the conditional energy).
    VERBATIM exp15 pt_p0_calibrate.py:257."""
    g = step.model.graph
    node_map = g.node_mapping
    bias_nodes = list(g.output_nodes) + list(g.hidden_nodes)
    weight_edges = list(g.base_graph_edges)
    free_blocks = list(step.training_spec.program_negative.gibbs_spec.free_blocks)
    free_nodes = [n for blk in free_blocks for n in blk]
    free_global = [node_map[n] for n in free_nodes]
    pos = {gid: i for i, gid in enumerate(free_global)}
    e0 = np.array([pos[node_map[e.connected_nodes[0]]] for e in weight_edges], dtype=np.int32)
    e1 = np.array([pos[node_map[e.connected_nodes[1]]] for e in weight_edges], dtype=np.int32)
    bp = np.array([pos[node_map[n]] for n in bias_nodes], dtype=np.int32)
    # trained params aligned to this ordering (NEW exp15): index by the SAME base_graph_edges / output+hidden
    edge_gidx = np.array([g.edge_mapping[e] for e in weight_edges], dtype=np.int64)
    bias_gidx = np.array([node_map[n] for n in bias_nodes], dtype=np.int64)
    W_e = np.asarray(step.model.weights)[edge_gidx].astype(np.float64)
    b_n = np.asarray(step.model.biases)[bias_gidx].astype(np.float64)
    # COUPLING edges (input<->output) — LOAD-BEARING for the conditional energy: the clamped input
    # spin is identical across replicas but multiplies the DIFFERING free output spin, so the coupling
    # term does NOT cancel. Each coupling edge has one INPUT (clamped) endpoint + one OUTPUT (free)
    # endpoint; weights are the FIXED diffusion couplings (model.weights at the coupling-edge indices).
    clamped_blocks = list(step.training_spec.program_negative.gibbs_spec.clamped_blocks)
    clamp_nodes = [n for blk in clamped_blocks for n in blk]
    clamp_pos = {node_map[n]: i for i, n in enumerate(clamp_nodes)}
    inp_set = set(node_map[n] for n in g.input_nodes)
    coupling_edges = list(g.image_coupling_edges) + list(g.label_coupling_edges)
    co_pos, ci_pos, cw = [], [], []
    for e in coupling_edges:
        ga, gb = node_map[e.connected_nodes[0]], node_map[e.connected_nodes[1]]
        g_in, g_out = (ga, gb) if ga in inp_set else (gb, ga)   # input=clamped, output=free
        co_pos.append(pos[g_out])
        ci_pos.append(clamp_pos[g_in])
        cw.append(np.asarray(step.model.weights)[g.edge_mapping[e]])
    return dict(free_blocks=free_blocks, edge_pos0=e0, edge_pos1=e1, bias_pos=bp,
                n_edge=len(weight_edges), n_bias=len(bias_nodes), n_free=len(free_global),
                W_e=W_e, b_n=b_n, block_lens=[len(b) for b in free_blocks],
                coup_out_pos=np.array(co_pos, dtype=np.int32), coup_in_pos=np.array(ci_pos, dtype=np.int32),
                coup_w=np.array(cw, dtype=np.float64), n_clamp=len(clamp_nodes), n_coupling=len(coupling_edges))


def energy_free(spins_2d, clamp_spins, maps):
    """The THREE-term free-conditional energy over the FREE spins, given the CLAMPED input:
         E(s_free; s_in) = -( Σ_{base edges} W_e s_e0 s_e1 + Σ_{free bias} b_n s_n
                              + Σ_{coupling edges} cw_c s_free[out] s_in[in] )
    spins in {-1,+1}; β-FREE. The coupling term (third) is REQUIRED — s_in is replica-identical but
    multiplies the differing free output spin, so it does NOT cancel. This is the building block the
    Task-4 compat-core sits on. VERBATIM exp15 pt_p0_calibrate.py:296."""
    s0 = spins_2d[:, maps["edge_pos0"]]
    s1 = spins_2d[:, maps["edge_pos1"]]
    e_edge = -(s0 * s1) @ maps["W_e"]
    e_bias = -(spins_2d[:, maps["bias_pos"]]) @ maps["b_n"]
    e_coup = -(spins_2d[:, maps["coup_out_pos"]] * clamp_spins[:, maps["coup_in_pos"]]) @ maps["coup_w"]
    return e_edge + e_bias + e_coup


# ================================================================ PART A.4 — MANDATORY trained-weight refresh
# VERBATIM from exp15-recheck/recheck.py:74-129 + internal_exp.py:186-242. The exp15/16 init-weight
# bug fix: AnnealingIsingSamplingProgram's constructor reads step.model.FACTORS (stale INIT — DTM.train
# updates step.model.weights and the program interactions but NOT step.model.factors, DTM.py:337-340),
# so a freshly-built program samples INIT weights. The refresh re-derives per_block_interactions from
# the CURRENT trained globals via get_new_per_block_interactions + injects via eqx.tree_at — exactly
# what update_weights_and_biases does. The original `build_alpha_programs_refreshed` wrapper is NOT
# ported (PT-ladder-specific); only the standalone refresh + proof.
def refresh_program_weights(prog, step):
    """Inject the CURRENT trained step.model.weights/biases into a freshly-built sampling program's
    per_block_interactions (the program constructor read stale INIT factors). Returns the refreshed
    program. Recipe VERBATIM from exp15-recheck/recheck.py:87-88 (the `ni = get_new... ; tree_at`
    body) — MUST be called before EVERY probe and EVERY L_compat build (build-notes §refresh)."""
    import equinox as eqx

    from thrmlDenoising.sampling_specs import get_new_per_block_interactions

    ni = get_new_per_block_interactions(prog, step.model.weights, step.model.biases)
    return eqx.tree_at(lambda p: p.per_block_interactions, prog, ni)


def _collect_weight_interactions(per_block_interactions, sink):
    """Recurse a per_block_interactions tree collecting (weight_global_indices, weights) leaf pairs.
    VERBATIM the `rec` closure from exp15-recheck/recheck.py:109-118."""
    def rec(o):
        if hasattr(o, "weights") and hasattr(o, "weight_global_indices"):
            sink.append((np.asarray(o.weight_global_indices), np.asarray(o.weights)))
        if hasattr(o, "_fields"):
            [rec(getattr(o, f)) for f in o._fields]
        elif isinstance(o, dict):
            [rec(v) for v in o.values()]
        elif isinstance(o, (list, tuple)):
            [rec(v) for v in o]

    rec(per_block_interactions)


def refreshed_weight_proof(step) -> dict:
    """The HARD-HALT proof that the trained-weight refresh is correct AND necessary:
      * refresh_ok           — a rebuilt-and-refreshed program's first weight interaction == trained
                               step.model.weights (maxabs < 1e-6); the fix took.
      * constructor_was_stale — the UN-refreshed constructor would have given INIT (maxabs > 1e-6);
                               the bug premise holds.
    BOTH must be True. Keys match the wiki exactly. VERBATIM exp15-recheck/recheck.py:101-129."""
    import jax.numpy as jnp

    from thrmlDenoising.annealing_graph_ising import AnnealingIsingSamplingProgram
    from thrmlDenoising.sampling_specs import get_new_per_block_interactions

    ts = step.training_spec
    prog = AnnealingIsingSamplingProgram(step.model, list(ts.program_negative.gibbs_spec.free_blocks),
                                         list(ts.program_negative.gibbs_spec.clamped_blocks),
                                         jnp.asarray(1.0), ts.schedule_negative)
    stale = []
    _collect_weight_interactions(prog.per_block_interactions, stale)
    ni = get_new_per_block_interactions(prog, step.model.weights, step.model.biases)
    prog = eqx_tree_at_interactions(prog, ni)
    fresh = []
    _collect_weight_interactions(prog.per_block_interactions, fresh)
    wt = np.asarray(step.model.weights)
    gi, wv_stale = stale[0]
    _, wv_fresh = fresh[0]
    return dict(stale_vs_trained_maxabs=float(np.max(np.abs(wv_stale - wt[gi]))),     # >0 = constructor was stale (the bug)
                refreshed_vs_trained_maxabs=float(np.max(np.abs(wv_fresh - wt[gi]))),  # ~0 = fix took
                refresh_ok=bool(np.max(np.abs(wv_fresh - wt[gi])) < 1e-6),
                constructor_was_stale=bool(np.max(np.abs(wv_stale - wt[gi])) > 1e-6))


def eqx_tree_at_interactions(prog, ni):
    """eqx.tree_at(lambda p: p.per_block_interactions, prog, ni) — the single injection used by both
    refresh_program_weights and refreshed_weight_proof (VERBATIM exp15-recheck recheck.py:120)."""
    import equinox as eqx

    return eqx.tree_at(lambda p: p.per_block_interactions, prog, ni)


# ================================================================ PART B.1 — K=50 retained Y-process
def rho_Y(retained):
    """ρ_Y(ℓ) on the retained Y-process. `retained` is the already-strided/retained trajectory
    (n_chains, L_traj, b) where consecutive samples are STRIDE_SWEEPS apart in sweep units, so ρ_Y(ℓ)
    = ρ_X(8ℓ). Returns the per-lag autocorrelation averaged over observable columns AND chains:
    a (L_traj,) array (ρ_Y[0] == 1). Uses the verbatim _rho_block FFT estimator on the retained block."""
    rho = _rho_block(np.asarray(retained, dtype=np.float64))   # (L_traj, b)
    return rho.mean(axis=1)                                     # average over observable columns


def tau_int_Y_from_retained(retained):
    """τ_int,Y via half-Sokal on the retained Y-process (PRIMARY mixing-margin scalar). Per-column
    half-Sokal then averaged over columns. retained: (n_chains, L_traj, b). Returns float."""
    rho = _rho_block(np.asarray(retained, dtype=np.float64))   # (L_traj, b)
    tau_cols = _tau_half_from_rho(rho)                          # (b,)
    return float(np.mean(tau_cols))


def _T_O_Y_from_retained(retained):
    """T_{O,Y} = ½·Σ_a S_a with S_a = 2·τ_int,Y_a·Var_a, on the retained Y-process (sokal-style but in
    retained-sample units). retained: (n_chains, L_traj, b)."""
    arr = np.asarray(retained, dtype=np.float64)
    rho = _rho_block(arr)                                       # (L_traj, b)
    tau_cols = _tau_half_from_rho(rho)                          # (b,)
    var_cols = arr.reshape(-1, arr.shape[-1]).var(axis=0)       # (b,) over (chain, time)
    S_a = 2.0 * tau_cols * var_cols
    return float(0.5 * S_a.sum())


# ================================================================ PART B.2 — Rademacher sketch (E1/E6)
def rademacher_sketches(p, n_R, diag_key):
    """N_R FIXED, deterministic ±1 Rademacher sketch vectors of dimension p, shared across the whole
    calibration (frozen at Task-9). Deterministic given diag_key (reproducible). Returns (n_R, p)
    in {-1,+1}. Uses a numpy default_rng seeded by diag_key (independent of jax RNG)."""
    rng = np.random.default_rng(int(diag_key))
    return (rng.integers(0, 2, size=(int(n_R), int(p))) * 2 - 1).astype(np.float64)


def _project_retained(retained, sketch_row):
    """Project the per-column retained observables onto ONE Rademacher sketch vector → a scalar
    sketch-observable per (chain, time). retained: (n_chains, L_traj, b); sketch_row: (b,).
    Returns (n_chains, L_traj, 1) — a single observable column (keeps the _rho_block shape contract)."""
    proj = np.asarray(retained, dtype=np.float64) @ np.asarray(sketch_row, dtype=np.float64)  # (n_chains, L_traj)
    return proj[:, :, None]


def rademacher_sketch_scalars(retained, n_R, diag_key):
    """Project the retained observables onto N_R shared Rademacher sketches; compute per-sketch τ_int,Y
    and T_{O,Y}; report the WORST (max τ_int,Y) over the N_R sketches (a screening estimate — the
    projection SE caps the detectable margin). Deterministic given diag_key. Returns a dict with the
    worst-case (tau_int_Y, _T_O_Y) and the full per-sketch arrays."""
    arr = np.asarray(retained, dtype=np.float64)
    b = arr.shape[-1]
    sketches = rademacher_sketches(b, n_R, diag_key)           # (n_R, b)
    per_tau, per_TO = [], []
    for r in range(sketches.shape[0]):
        sk = _project_retained(arr, sketches[r])              # (n_chains, L_traj, 1)
        per_tau.append(tau_int_Y_from_retained(sk))
        per_TO.append(_T_O_Y_from_retained(sk))
    worst = int(np.argmax(per_tau))                            # worst = max τ_int,Y (least-mixed sketch)
    return dict(tau_int_Y=float(per_tau[worst]), _T_O_Y=float(per_TO[worst]),
                per_sketch_tau=[float(x) for x in per_tau],
                per_sketch_T_O=[float(x) for x in per_TO],
                worst_sketch_idx=worst, n_R=int(n_R))


# ================================================================ PART B.3 — the probe scalars API
def probe_scalars(retained_Y_obs, n_R, diag_key, gradient):
    """The clean scalar bundle the TrainabilityProbe (Task 6) calls. Given a retained Y-process of
    observables (n_chains, L_traj, b), the Rademacher count n_R, a deterministic diag_key, and the
    measured gradient vector g, return the K=50-convention scalars (worst-of-N_R for the mixing
    margin):

      r_grad[1]      = ρ_Y(1)              (diagnostic — first-lag retained autocorrelation),
      r_grad[50]     = ρ_Y(50)             (plateau sanity, if L_traj > 50),
      tau_int_Y      = worst-of-N_R half-Sokal τ_int on the retained process (PRIMARY scalar),
      ESS_hat        = K/(2·τ_int,Y),      K = 50,
      Q_struct_perp  = (K/2)·‖g‖²/T_{O,Y}, K = 50  (T_{O,Y} from the same worst sketch),
      gradient_norm  = ‖g‖.

    Deterministic: same retained + diag_key + gradient → identical output. _T_O_Y is exposed (leading
    underscore) so callers can reconstruct/verify Q_struct^⊥; it is not a headline scalar."""
    arr = np.asarray(retained_Y_obs, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    rho = rho_Y(arr)                                           # (L_traj,) ρ_Y averaged over columns + chains
    r_grad_1 = float(rho[1]) if rho.shape[0] > 1 else float("nan")
    r_grad_50 = float(rho[K_WINDOW]) if rho.shape[0] > K_WINDOW else float("nan")

    sk = rademacher_sketch_scalars(arr, n_R=n_R, diag_key=diag_key)   # worst-of-N_R mixing margin
    tau = sk["tau_int_Y"]
    T_O_Y = sk["_T_O_Y"]
    grad_norm = float(np.linalg.norm(g))
    ess_hat = K_WINDOW / (2.0 * tau)
    q_struct_perp = (K_WINDOW / 2.0) * grad_norm ** 2 / T_O_Y if T_O_Y > 0 else float("inf")
    return {
        "r_grad[1]": r_grad_1,
        "r_grad[50]": r_grad_50,
        "tau_int_Y": float(tau),
        "ESS_hat": float(ess_hat),
        "Q_struct_perp": float(q_struct_perp),
        "gradient_norm": grad_norm,
        "_T_O_Y": float(T_O_Y),
        "n_R": int(n_R),
        "worst_sketch_idx": int(sk["worst_sketch_idx"]),
    }
