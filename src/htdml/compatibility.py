"""Stage-C compatibility free energy — the compat-CORE (Task 5).

A **deterministic mean-field SURROGATE** free energy `F_MF` for the frozen-DTM positive-phase
("clamp") partition. It is the user-mandated joint-training object: `L_compat = Σ_k F_MF^(k)` is the
free-energy term added to the encoder/decoder loss so the encoder is steered toward latents the
*frozen* DTM finds low-free-energy (λ=0 ≡ control — see below).

  HONESTY LABEL (read this): this is a deterministic mean-field HEURISTIC / SURROGATE, NOT an exact
  joint DTM objective. Its only proven property is the variational UPPER bound `F_MF ≥ F_exact`
  (Gibbs–Bogoliubov, from the damped mean-field over the un-marginalized layer). A variational upper
  bound does NOT guarantee that minimizing it improves *sampled*-DTM quality; the gap `F_MF − F_exact`
  is reported as the honesty metric (fixture check d-ii). Do not call this an exact DTM free energy.

STRUCTURE (decisive, from the design notes §"Stage-C compatibility free energy — STRUCTURAL TRUTH",
re-verified against the real 4_4 graph — see tests/fixture_6_4.py d-i):

  * Clamp = the **POSITIVE-phase partition** `program_positive` (sampling_specs.py:110):
        FREE    = {upper_hidden, lower_hidden}                       (the two hidden blocks)
        CLAMPED = {image_output (= b0-derived), label_output, conditioning = b_t}
    This is NOT the training-negative partition (that frees the outputs). Maps are built from
    `program_positive`, on TRAINED globals (refresh_program_weights + refreshed_weight_proof first —
    the stale-factors bug).

  * The grid is strictly bipartite (chessboard): base edges connect the UPPER half
    {upper_hidden, image_output, label_output} to the LOWER half {lower_hidden} ONLY. So:
        neighbors(lower_hidden) ⊆ {upper_hidden, image_output, label_output}   (all free-u or clamped),
        neighbors(upper_hidden) ⊆ {lower_hidden}.
    b_t is an EXTERNAL leaf set coupling 1-to-1 to OUTPUT nodes only (image_input↔image_output,
    label_input↔label_output) with fixed forward-diffusion weights → it is a constant OUTPUT-BIAS
    correction, it does NOT enter any hidden node's local field.

  Step 1 — **exactly marginalize lower_hidden** (Poisson-binomial bipartite ⇒ each lower node's spin
  appears only in its own local field). Given upper_hidden config `u` and the clamps:
        field_j(u) = b_j + Σ_{i ∈ upper-neighbors(j)} W_ji · s_i      (s_i ∈ clamped outputs or free u),
        F_low(u)   = β·E_clamp(u) − Σ_{j∈lower} log(2 cosh(β·field_j(u))).
    `E_clamp(u)` collects every energy term NOT involving a lower node: the upper_hidden / output /
    b_t biases and the b_t→output coupling. This is EXACT (fixture d-i: residual < 1e-10 vs a
    2^{N_lower} brute force on the real graph) — the safety net of the whole Stage-C approach.
    Units: `F_low` is dimensionless (β folded in) so it equals `−log Σ_lower exp(−β E)`.

  Step 2 — **damped deterministic mean-field over upper_hidden** → the variational upper bound. Iterate
  the upper magnetizations `m_u` for a FIXED number of UNROLLED damped fixed-point steps
        m ← (1−η)·m + η·tanh(β · h_eff(m)),     h_eff_a(m) = −∂⟨F_low⟩/∂m_a  (the mean-field local field),
    then `F_MF = ⟨F_low⟩_m − H(m)/β` with `H` the per-node binary entropy. Gradients flow THROUGH the
    unrolled iterations (NOT detached at the fixed point). DRAWS NO PRNG KEY (deterministic).

  Per-step + total: `L_compat = Σ_{k=0..K_steps-1} F_MF^(k)`; each step k refreshes the trained-weight
  program (refreshed_weight_proof per step). DTM params enter under `stop_gradient`; β fixed. b_t =
  `stop_gradient(forward_noise(b0))` is a HARD draw — only b0 (the latent) carries ∂L_compat/∂latent.

  λ=0 ≡ control: `L_compat` is a clean differentiable function of the clamp latent, so the driver
  (Task 8) forms `λ · L_compat` with a TRACED λ (no Python branch); at λ=0 the product is bitwise the
  control. Non-finite-guarded by `compat_loss` (returns the value + an is_finite flag).

DTM params under stop_gradient; the only traced input is the clamp latent (image_output spins from b0).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from htdml.paths import bootstrap_paths  # noqa: E402

bootstrap_paths()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# ---- default unrolled-mean-field hyperparameters (deterministic; no PRNG) -------------------------
MF_STEPS = 50          # unrolled damped fixed-point iterations over upper_hidden
MF_DAMPING = 0.5       # η — damping factor m ← (1−η)m + η·tanh(βh)
MF_M_INIT = 0.0        # initial upper magnetizations (m=0 ⇔ maximal-entropy start)


# ====================================================================== positive-phase map builder
def build_compat_maps(step):
    """Build the per-edge / per-node maps for the COMPAT (positive-phase) partition on TRAINED globals.

    FREE  blocks = program_positive.free_blocks  = [upper_hidden, lower_hidden]   (the marginalized +
                                                    mean-field layers).
    CLAMP blocks = program_positive.clamped_blocks = [image_output, label_output, b_t].

    The clamp spin vector the core consumes is ordered EXACTLY as these clamped blocks concatenate:
    [image_output (n_img), label_output (n_lab), b_t (n_bt)] — image_output first, so the latent
    (b0-derived) clamp occupies columns [0 : n_img).

    Returns a dict of NUMPY arrays (static structure) + the trained weights/biases aligned to the
    free/clamp ordering. The DTM globals are read ONCE here (under the caller's stop_gradient); the
    returned maps carry no traced leaves.
    """
    g = step.model.graph
    nm = g.node_mapping
    W = np.asarray(step.model.weights, dtype=np.float64)
    B = np.asarray(step.model.biases, dtype=np.float64)

    fb = list(step.training_spec.program_positive.gibbs_spec.free_blocks)
    cb = list(step.training_spec.program_positive.gibbs_spec.clamped_blocks)
    assert len(fb) == 2, f"positive free blocks must be [upper_hidden, lower_hidden], got {len(fb)}"

    upper_g = [nm[n] for n in fb[0]]                       # upper_hidden global ids (mean-field layer)
    lower_g = [nm[n] for n in fb[1]]                       # lower_hidden global ids (marginalized layer)
    clamp_g = [nm[n] for blk in cb for n in blk]           # [image_output, label_output, b_t] global ids
    n_upper, n_lower, n_clamp = len(upper_g), len(lower_g), len(clamp_g)

    # the image_output clamp columns (the latent-carrying clamp): the FIRST block of cb.
    img_out_g = [nm[n] for n in cb[0]]
    n_img = len(img_out_g)
    n_lab = len(cb[1])
    n_bt = len(cb[2])

    # local positions: upper layer indexed 0..n_upper-1, lower 0..n_lower-1, clamp 0..n_clamp-1.
    pos_upper = {gid: i for i, gid in enumerate(upper_g)}
    pos_lower = {gid: i for i, gid in enumerate(lower_g)}
    pos_clamp = {gid: i for i, gid in enumerate(clamp_g)}
    lower_set = set(lower_g)
    upper_set = set(upper_g)
    clamp_set = set(clamp_g)

    # base edges: every base edge touches exactly one LOWER node (strict bipartite) and one UPPER-half
    # node that is either upper_hidden (free u) or a clamped output. Classify each base edge into:
    #   (A) lower↔upper_hidden  — couples a free-lower to a free-upper (enters field_j via u),
    #   (B) lower↔clamped-out   — couples a free-lower to a clamp (enters field_j via clamp).
    lh_uh_lower, lh_uh_upper, lh_uh_w = [], [], []        # field_j contribution from free upper u
    lh_cl_lower, lh_cl_clamp, lh_cl_w = [], [], []        # field_j contribution from clamped output
    for e in g.base_graph_edges:
        a, b = nm[e.connected_nodes[0]], nm[e.connected_nodes[1]]
        w = float(W[g.edge_mapping[e]])
        # identify the lower endpoint + the other (upper-half) endpoint
        if a in lower_set:
            jlow, other = a, b
        elif b in lower_set:
            jlow, other = b, a
        else:
            raise AssertionError(
                f"base edge {a}-{b} has no lower_hidden endpoint — premise (strict bipartite, "
                "lower↔upper only) violated for the compat marginalization")
        if other in upper_set:
            lh_uh_lower.append(pos_lower[jlow]); lh_uh_upper.append(pos_upper[other]); lh_uh_w.append(w)
        elif other in clamp_set:
            lh_cl_lower.append(pos_lower[jlow]); lh_cl_clamp.append(pos_clamp[other]); lh_cl_w.append(w)
        else:
            raise AssertionError(
                f"lower_hidden neighbor {other} is neither upper_hidden nor a clamped output — "
                "premise neighbors(lower_hidden) ⊆ {upper_hidden, outputs} violated")

    # coupling edges: b_t ↔ output, both CLAMPED in the positive partition → a constant clamp-clamp term.
    coup_cl0, coup_cl1, coup_w = [], [], []
    for e in list(g.image_coupling_edges) + list(g.label_coupling_edges):
        a, b = nm[e.connected_nodes[0]], nm[e.connected_nodes[1]]
        w = float(W[g.edge_mapping[e]])
        assert a in clamp_set and b in clamp_set, (
            f"coupling edge {a}-{b} endpoints not both clamped in positive partition")
        coup_cl0.append(pos_clamp[a]); coup_cl1.append(pos_clamp[b]); coup_w.append(w)

    return dict(
        # sizes
        n_upper=n_upper, n_lower=n_lower, n_clamp=n_clamp,
        n_img=n_img, n_lab=n_lab, n_bt=n_bt,
        # biases aligned to local layer ordering
        b_upper=B[np.array(upper_g, dtype=np.int64)],        # (n_upper,)
        b_lower=B[np.array(lower_g, dtype=np.int64)],        # (n_lower,)
        b_clamp=B[np.array(clamp_g, dtype=np.int64)],        # (n_clamp,)
        # field_j builders: lower↔upper_hidden (free) and lower↔clamped-output
        fu_lower=np.array(lh_uh_lower, dtype=np.int64), fu_upper=np.array(lh_uh_upper, dtype=np.int64),
        fu_w=np.array(lh_uh_w, dtype=np.float64),
        fc_lower=np.array(lh_cl_lower, dtype=np.int64), fc_clamp=np.array(lh_cl_clamp, dtype=np.int64),
        fc_w=np.array(lh_cl_w, dtype=np.float64),
        # b_t↔output coupling (clamp-clamp constant)
        cc0=np.array(coup_cl0, dtype=np.int64), cc1=np.array(coup_cl1, dtype=np.int64),
        cc_w=np.array(coup_w, dtype=np.float64),
        # provenance global-id lists
        upper_g=np.array(upper_g, dtype=np.int64), lower_g=np.array(lower_g, dtype=np.int64),
        clamp_g=np.array(clamp_g, dtype=np.int64), img_out_g=np.array(img_out_g, dtype=np.int64),
    )


def _jnp_maps(maps):
    """Promote the numpy map arrays to jnp once (the structural arrays are constants in the graph)."""
    out = {}
    for k, v in maps.items():
        if isinstance(v, np.ndarray) and v.dtype.kind == "f":
            out[k] = jnp.asarray(v, dtype=jnp.float64) if v.size else jnp.zeros((0,), jnp.float64)
        else:
            out[k] = v
    return out


# ====================================================================== Step 1 — lower marginalization
def lower_fields(u_spins, clamp_spins, maps):
    """Local field on every lower_hidden node given upper_hidden config `u_spins` (∈ ±1 or a mean
    magnetization in [−1,1]) and the clamp spins. field_j = b_j + Σ_{upper-nbr i} W_ji·u_i +
    Σ_{clamp-nbr c} W_jc·clamp_c. Vectorized via segment sums over the strict-bipartite edge lists.

    u_spins:     (n_upper,)   clamp_spins: (n_clamp,)   → returns (n_lower,)."""
    n_lower = maps["n_lower"]
    dt = jnp.result_type(jnp.asarray(u_spins), jnp.asarray(clamp_spins), jnp.float32)
    fld = jnp.asarray(maps["b_lower"]).astype(dt)
    # free upper_hidden contribution (segment-sum buffer dtype matched to `contrib` to avoid an
    # implicit f64→f32 scatter cast under x64).
    if maps["fu_w"].shape[0] > 0:
        contrib = (jnp.asarray(maps["fu_w"]).astype(dt) * u_spins[maps["fu_upper"]].astype(dt))
        fld = fld + jax.ops.segment_sum(contrib, maps["fu_lower"], num_segments=n_lower).astype(dt)
    # clamped-output contribution
    if maps["fc_w"].shape[0] > 0:
        contrib = (jnp.asarray(maps["fc_w"]).astype(dt) * clamp_spins[maps["fc_clamp"]].astype(dt))
        fld = fld + jax.ops.segment_sum(contrib, maps["fc_lower"], num_segments=n_lower).astype(dt)
    return fld


def clamp_energy(u_spins, clamp_spins, maps, beta):
    """β·E_clamp(u): every energy term NOT involving a lower_hidden node — the upper_hidden bias·u, the
    clamp (output + b_t) biases, and the b_t↔output coupling. Dimensionless (β-folded) to match the
    brute-force −log Σ_lower exp(−βE)."""
    e = -(jnp.asarray(maps["b_upper"]) @ u_spins)              # upper_hidden bias
    e = e - (jnp.asarray(maps["b_clamp"]) @ clamp_spins)       # output + b_t biases (constant in u)
    if maps["cc_w"].shape[0] > 0:                              # b_t↔output coupling (clamp-clamp const)
        e = e - jnp.sum(maps["cc_w"] * clamp_spins[maps["cc0"]] * clamp_spins[maps["cc1"]])
    return beta * e


def F_low(u_spins, clamp_spins, maps, beta):
    """EXACT lower_hidden free energy at a FIXED upper config u (the d-i-tested closed form):
         F_low(u) = β·E_clamp(u) − Σ_{j∈lower} log(2 cosh(β·field_j(u))).
    Equals −log Σ_{lower} exp(−β E(u, lower, clamps)) to machine precision (fixture d-i). Differentiable
    in both u_spins and clamp_spins. u_spins, clamp_spins: (n_upper,), (n_clamp,)."""
    fld = lower_fields(u_spins, clamp_spins, maps)
    logcosh = jnp.sum(_log_2cosh(beta * fld))
    return clamp_energy(u_spins, clamp_spins, maps, beta) - logcosh


def _log_2cosh(x):
    """Numerically stable log(2·cosh(x)) = |x| + log(1 + exp(−2|x|))."""
    ax = jnp.abs(x)
    return ax + jnp.log1p(jnp.exp(-2.0 * ax))


def _binary_entropy_from_m(m):
    """Per-node binary entropy H(p) of a ±1 magnetization m (p = (1+m)/2), summed. Stable at m→±1."""
    p = 0.5 * (1.0 + m)
    eps = 1e-12
    p = jnp.clip(p, eps, 1.0 - eps)
    return -jnp.sum(p * jnp.log(p) + (1.0 - p) * jnp.log1p(-p))


# ====================================================================== Step 2 — mean field over upper
def mean_F_low(m_upper, clamp_spins, maps, beta):
    """⟨F_low⟩ under a factorized mean field on upper_hidden with magnetizations `m_upper` (∈[−1,1]).

    Because F_low is LINEAR in each upper spin EXCEPT through the log-2cosh of field_j (which depends on
    u_i linearly inside the cosh argument), ⟨F_low⟩ is NOT simply F_low(m). We use the standard
    naive-mean-field treatment: the marginalized log-cosh term `Σ_j log 2cosh(β field_j(u))` is replaced
    by its value at u = m_upper (the mean-field / Plefka first-order substitution), giving a tractable
    deterministic surrogate. The linear clamp_energy(u) term is exactly ⟨·⟩-linear so m enters exactly.

    This `mean_F_low(m)` is what the damped fixed point minimizes; `h_eff = −∂ mean_F_low/∂m` is the
    mean-field local field. Returns a scalar."""
    return F_low(m_upper, clamp_spins, maps, beta)


def _mf_local_field(m_upper, clamp_spins, maps, beta):
    """h_eff(m) = −(1/β)·∂ mean_F_low/∂m_upper — the mean-field local field on each upper_hidden node
    (so the damped update is m ← (1−η)m + η·tanh(β·h_eff)). Computed by autodiff of mean_F_low."""
    grad = jax.grad(lambda m: mean_F_low(m, clamp_spins, maps, beta))(m_upper)
    return -grad / beta


def mean_field_solve(clamp_spins, maps, beta, n_steps=MF_STEPS, damping=MF_DAMPING, m_init=MF_M_INIT):
    """Run the UNROLLED damped mean-field fixed-point over upper_hidden (no PRNG; deterministic).
    Returns the final magnetizations m_upper (n_upper,). Gradients flow through every iteration (the
    unroll is a plain Python loop over jax ops — NOT lax.stop_gradient at the fixed point)."""
    m = jnp.full((maps["n_upper"],), float(m_init), dtype=jnp.float64)
    for _ in range(int(n_steps)):
        h = _mf_local_field(m, clamp_spins, maps, beta)
        m = (1.0 - damping) * m + damping * jnp.tanh(beta * h)
    return m


def F_MF(clamp_spins, maps, beta, n_steps=MF_STEPS, damping=MF_DAMPING, m_init=MF_M_INIT):
    """The deterministic mean-field surrogate free energy for ONE diffusion step:
         F_MF = mean_F_low(m*) − H(m*)/β,
    where m* is the unrolled damped fixed point (mean_field_solve) and H is the upper-layer binary
    entropy. `F_MF ≥ F_exact` (variational upper bound; fixture d-ii). Differentiable in clamp_spins;
    gradients flow through the unrolled m*. Draws NO PRNG key. Returns a scalar."""
    m_star = mean_field_solve(clamp_spins, maps, beta, n_steps=n_steps, damping=damping, m_init=m_init)
    energy = mean_F_low(m_star, clamp_spins, maps, beta)
    entropy = _binary_entropy_from_m(m_star)
    return energy - entropy / beta


# ====================================================================== per-step + total L_compat
def refreshed_compat_maps(step):
    """Refresh the trained weights into a freshly-built positive-phase program (the stale-
    factors guard), assert the proof, THEN build the compat maps from program_positive. Use this (not
    build_compat_maps directly) for any real L_compat build — it enforces the mandatory refresh-proof.

    NOTE: build_compat_maps reads step.model.weights/biases directly (the TRAINED globals, already
    correct), so the refresh is about the SAMPLING PROGRAM's per_block_interactions, not the maps. We
    still gate on refreshed_weight_proof so every compat build clears the same guard every probe does.
    """
    from harness import probe_primitives as pp

    proof = pp.refreshed_weight_proof(step)
    if not (proof["constructor_was_stale"] and proof["refresh_ok"]):
        raise AssertionError(
            f"refreshed_weight_proof failed for compat build: {proof} — the trained-weight refresh "
            "guard (stale-factors bug) did not clear; refusing to build L_compat on stale weights")
    return build_compat_maps(step), proof


def L_compat(clamp_spins_per_step, step_maps, beta, n_steps=MF_STEPS, damping=MF_DAMPING,
             m_init=MF_M_INIT):
    """The compat free-energy term summed over diffusion steps:  L_compat = Σ_k F_MF^(k).

    Args (the Task-8 driver signature):
      clamp_spins_per_step : (K_steps, n_clamp) jnp array — the per-step clamp spin vectors. Each row
                             is [image_output (= b0-derived latent), label_output, b_t]. Only the
                             image_output columns carry ∂L_compat/∂latent; label_output + b_t are
                             stop_gradient'd HARD draws by the caller (b_t = stop_gradient(
                             forward_noise(b0))).
      step_maps            : list (len K_steps) of compat maps (one per diffusion step k), each from
                             refreshed_compat_maps(step_k) — its own trained-weight-refreshed program.
                             (A single shared map may be passed as a length-1 list to reuse across
                             steps when steps share weights.)
      beta                 : inverse temperature (fixed).

    Returns a scalar = Σ_k F_MF(clamp_spins_per_step[k], step_maps[k]). Differentiable in
    clamp_spins_per_step; draws NO PRNG key. The driver multiplies by a TRACED λ (λ=0 ≡ control)."""
    clamp = jnp.asarray(clamp_spins_per_step, dtype=jnp.float64)
    if clamp.ndim == 1:
        clamp = clamp[None, :]
    K_steps = clamp.shape[0]
    maps_list = step_maps if isinstance(step_maps, (list, tuple)) else [step_maps]
    if len(maps_list) == 1 and K_steps > 1:
        maps_list = list(maps_list) * K_steps
    assert len(maps_list) == K_steps, (
        f"step_maps length {len(maps_list)} != number of clamp rows {K_steps}")
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for k in range(K_steps):
        jm = _jnp_maps(maps_list[k])
        total = total + F_MF(clamp[k], jm, beta, n_steps=n_steps, damping=damping, m_init=m_init)
    return total


def compat_loss(lam, clamp_spins_per_step, step_maps, beta, **mf_kwargs):
    """λ·L_compat with a TRACED λ (no Python branch on λ) + a non-finite guard. At λ=0.0 this is
    bitwise the control (0.0 · L_compat, plus the guard does not perturb a finite value). Returns
    (value, is_finite_flag). The driver (Task 8) wires λ; here we just expose the clean λ-multiply."""
    val = jnp.asarray(lam, dtype=jnp.float64) * L_compat(clamp_spins_per_step, step_maps, beta,
                                                         **mf_kwargs)
    is_finite = jnp.isfinite(val)
    safe = jnp.where(is_finite, val, 0.0)
    return safe, is_finite


# ====================================================================== brute-force references (tests)
def F_low_bruteforce(u_spins, clamp_spins, maps, beta):
    """Reference for fixture d-i: −log Σ_{lower ∈ {±1}^{n_lower}} exp(−β E(u, lower, clamps)) by EXACT
    2^{n_lower} enumeration, using the SAME edge/bias maps the analytic F_low uses (so any mismatch is
    a formula error, not a structural-map error). Returns a python float. NUMPY (no autodiff needed)."""
    u = np.asarray(u_spins, dtype=np.float64)
    c = np.asarray(clamp_spins, dtype=np.float64)
    n_lower = maps["n_lower"]
    fu_w, fu_lower, fu_upper = maps["fu_w"], maps["fu_lower"], maps["fu_upper"]
    fc_w, fc_lower, fc_clamp = maps["fc_w"], maps["fc_lower"], maps["fc_clamp"]
    b_lower = np.asarray(maps["b_lower"], dtype=np.float64)
    # constant (u,clamp)-only part: β·E_clamp(u)
    e_const = float(clamp_energy(jnp.asarray(u), jnp.asarray(c), _jnp_maps(maps), beta))
    # enumerate lower spins
    neg_betaE = np.empty(2 ** n_lower, dtype=np.float64)
    # precompute field_j (independent of lower spin): field_j = b_j + Σ free-u + Σ clamp
    fld = b_lower.copy()
    if fu_w.size:
        np.add.at(fld, fu_lower, fu_w * u[fu_upper])
    if fc_w.size:
        np.add.at(fld, fc_lower, fc_w * c[fc_clamp])
    for m in range(2 ** n_lower):
        l = np.array([(1.0 if (m >> k) & 1 else -1.0) for k in range(n_lower)], dtype=np.float64)
        # E = E_clamp(u) − Σ_j field_j · l_j   ⇒  −βE = −e_const + β Σ_j field_j l_j
        neg_betaE[m] = -e_const + beta * float(fld @ l)
    mx = neg_betaE.max()
    return -(np.log(np.sum(np.exp(neg_betaE - mx))) + mx)


def F_exact_full(clamp_spins, maps, beta):
    """Reference for fixture d-ii: −log Σ_{upper,lower} exp(−β E) by EXACT 2^{n_upper+n_lower}
    enumeration (the true free energy F_MF upper-bounds). Reuses F_low_bruteforce per upper config →
    F_exact = −log Σ_u exp(−F_low(u)). Returns a python float."""
    n_upper = maps["n_upper"]
    terms = np.empty(2 ** n_upper, dtype=np.float64)
    for mu in range(2 ** n_upper):
        u = np.array([(1.0 if (mu >> k) & 1 else -1.0) for k in range(n_upper)], dtype=np.float64)
        terms[mu] = -F_low_bruteforce(u, clamp_spins, maps, beta)   # = log Σ_lower exp(−βE) at this u
    mx = terms.max()
    return -(np.log(np.sum(np.exp(terms - mx))) + mx)
