"""Detailed-balance certificate for the reversible ½(P_AB+P_BA) block-Gibbs kernel.

This is the SOLE gate on the reversible-scan overlay patch
(`vendor/thrml_overlay/thrml/block_sampling.py::sample_blocks`). Its only effect is to randomize the
SUPERBLOCK visitation ORDER per super-sweep — forward `[g_0 .. g_{M-1}]` vs exactly reversed
`[g_{M-1} .. g_0]`, each with prob ½ — giving the π-self-adjoint kernel

    K = ½ (P_fwd + P_rev) = ½ (P_AB + P_BA),  with (P_fwd)* = P_rev in L²(π).

We verify it on a TINY ENUMERABLE shadow (exact π over 2^N states, exact block-Gibbs conditionals)
that has the SAME superblock structure as the PRODUCTION 44_12 DTM negative-phase kernel:

  * 4 FREE superblocks {upper_hidden, lower_hidden, image_output, label_output} (the training-negative
    free set — see design notes), each a 1-tuple in `sampling_order` — so the
    forward order is `[0,1,2,3]` and the reverse is `[3,2,1,0]`;
  * a CLAMPED conditioning block b_t (the negative-phase clamp = `[conditioning_block]` only), which
    enters the free spins' fields through fixed coupling edges (b_t couples 1-to-1 to the OUTPUT
    superblocks; see design notes on Stage-C b_t precision). The clamp is held fixed, so each free
    block's Gibbs conditional is exact and reversible w.r.t. the clamped-conditional π.

The DTM energy form (see design notes) is the Ising
    E(s; s_clamp) = -( Σ_edges W_e s_e0 s_e1 + Σ_bias b_n s_n + Σ_coupling cw_c s_free[cf] s_clamp[cc] ).

Checks (ported from internal reference harnesses):
  (1) each superblock Gibbs update P_g leaves π invariant AND is π-reversible (D P_g symmetric);
  (2) the DETERMINISTIC ordered product P_fwd is in general NOT π-reversible (DB residual ~1e-2) —
      exactly the non-reversible alternating/deterministic scan the patch replaces (the discriminator);
  (3) the adjoint identity (P_fwd)* = P_rev  (P* = D^{-1} P^T D in L²(π));
  (4) the PATCHED mixture K = ½(P_fwd + P_rev) IS π-reversible (max_asym < 1e-10) AND π-stationary.

Pure numpy; ZERO compute; no GPU; no thrml needed (the cert is INDEPENDENT of the live overlay so it
certifies the kernel MATH, not the import). `certify(...) -> dict` returns the residuals + PASS/FAIL.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

# Frozen gate constants (lifted verbatim from internal reference harnesses).
TOL_SYM = 1e-10      # detailed-balance / self-adjointness to ~machine precision
TOL_INV = 1e-10      # stationarity π P = π
TOL_ADJ = 1e-10      # adjoint identity (P_fwd)* = P_rev
MIN_NONREV = 1e-4    # P_fwd must be DEMONSTRABLY non-reversible (asymmetry well above noise)

# The 4 free superblocks of the production training-negative partition, in `sampling_order` order.
SUPERBLOCK_NAMES = ("upper_hidden", "lower_hidden", "image_output", "label_output")


# --------------------------------------------------------------------------- enumeration helpers
def spin_table(n: int) -> np.ndarray:
    """rows = all 2^n spin configs in {-1,+1}^n (bit b -> spin 2*b-1). internal convention."""
    return np.array(
        [[2 * b - 1 for b in bits] for bits in itertools.product((0, 1), repeat=n)],
        dtype=np.int64,
    )


def boltzmann_clamped(S, J, h, coupling, s_clamp, beta):
    """π(s) ∝ exp(-β E(s; s_clamp)) over the 2^n FREE configs, with a fixed CLAMPED block.

      E(s; s_clamp) = -½ Σ_ij J_ij s_i s_j - Σ_i h_i s_i - Σ_c cw_c s[cf_c] s_clamp[cc_c]
    (`coupling` = (cf, cc, cw): free endpoint, clamped endpoint, weight). The coupling term enters as
    an EXTERNAL field on the free spins (s_clamp fixed) — the analogue of b_t coupling to the outputs.
    """
    Sf = S.astype(float)
    pair = np.einsum("si,ij,sj->s", Sf, J, Sf)        # = 2 Σ_{i<j} J_ij s_i s_j (J sym, 0 diag)
    lin = Sf @ h
    cf, cc, cw = coupling
    coup = (Sf[:, cf] * s_clamp[cc][None, :]) @ cw     # Σ_c cw_c s[cf] s_clamp[cc]
    E = -0.5 * pair - lin - coup
    w = np.exp(-beta * (E - E.min()))
    return w / w.sum()


def block_gibbs_matrix(pi, S, block):
    """Exact transition matrix of a single-superblock Gibbs update of `block` (a list of free site
    indices): resample x_block ~ π(. | x_{-block}). Row-stochastic. Verbatim from the internal reference harness."""
    N = S.shape[0]
    n = S.shape[1]
    comp = [i for i in range(n) if i not in block]
    groups = defaultdict(list)
    for x in range(N):
        groups[tuple(S[x, c] for c in comp)].append(x)
    P = np.zeros((N, N))
    for members in groups.values():
        Z = pi[members].sum()
        for x in members:
            P[x, members] = pi[members] / Z   # y must agree off-block; prob ∝ π(y)
    return P


def ordered_product(block_mats, order):
    """Apply blocks left-to-right in `order` (row-stochastic composition). Verbatim from the internal reference harness."""
    P = None
    for idx in order:
        P = block_mats[idx] if P is None else P @ block_mats[idx]
    return P


def max_asym(P, pi):
    """max |D P - (D P)^T| with D = diag(π) (= detailed-balance residual). Verbatim from the internal reference harness."""
    DP = pi[:, None] * P
    return float(np.max(np.abs(DP - DP.T)))


def adjoint_in_pi(P, pi):
    """The L²(π) adjoint of P: P* = D^{-1} P^T D. Verbatim from the internal reference harness."""
    return (P.T * pi[None, :]) / pi[:, None]


# --------------------------------------------------------------------------- the production-shape cell
def make_dtm_negative_cell(rng, sizes=(1, 1, 1, 1), n_clamp=2, beta=0.9):
    """Build a tiny enumerable shadow with the SAME superblock structure as the 44_12 training-negative
    DTM kernel: 4 free superblocks {upper_hidden, lower_hidden, image_output, label_output} + a clamped
    b_t. `sizes` = #free spins per superblock (default 1 each -> 4 single-site free superblocks; the
    forward [0,1,2,3] / reverse [3,2,1,0] DTM order). Returns (blocks, J, h, coupling, s_clamp, beta).

    Structure honoring the design notes:
      * the base grid is STRICTLY BIPARTITE upper<->lower (no intra-superblock edges) — so we wire
        coupling edges between distinct superblocks only and never within a superblock;
      * b_t couples ONLY to the OUTPUT superblocks (image_output, label_output) with fixed weights
        (see design notes on b_t precision), so the coupling endpoints are output free sites.
    """
    # assign contiguous free-spin indices to each superblock
    blocks = []
    idx = 0
    for sz in sizes:
        blocks.append(list(range(idx, idx + sz)))
        idx += sz
    n = idx  # total free spins

    upper, lower, img_out, lab_out = blocks

    # base-graph edges: bipartite-ish, NON-commuting across superblocks (so the deterministic scan is
    # genuinely non-reversible). upper<->lower, lower<->{img_out,lab_out}, upper<->{img_out,lab_out}.
    J = np.zeros((n, n))

    def add_edge(a, b, w):
        J[a, b] = J[b, a] = w

    cross_pairs = []
    for u in upper:
        for l in lower:
            cross_pairs.append((u, l))
    for l in lower:
        for o in img_out + lab_out:
            cross_pairs.append((l, o))
    for u in upper:
        for o in img_out + lab_out:
            cross_pairs.append((u, o))
    for a, b in cross_pairs:
        add_edge(a, b, float(rng.normal(0.0, 0.7)))

    h = rng.normal(0.0, 0.5, size=n)

    # clamped b_t couples 1-to-1 to the OUTPUT free sites only (fixed forward-diffusion weights).
    out_sites = img_out + lab_out
    cf = np.array(out_sites, dtype=np.int64)
    cc = rng.integers(0, n_clamp, size=len(out_sites)).astype(np.int64)
    cw = rng.normal(0.0, 0.6, size=len(out_sites))    # NONZERO coupling
    s_clamp = rng.choice([-1.0, 1.0], size=n_clamp).astype(float)
    coupling = (cf, cc, cw)

    return blocks, J, h, coupling, s_clamp, beta


# --------------------------------------------------------------------------- the certificate
def certify(rng=None, sizes=(1, 1, 1, 1), n_clamp=2, beta=0.9, verbose=True):
    """Build the explicit ½(P_fwd+P_rev) transition matrix on the production-shape (44_12-structured)
    enumerable shadow and certify detailed balance. Returns a dict with residuals + 'passed'.

    Raises AssertionError on any gate failure (so a caller/test that wants HARD-HALT semantics gets it),
    but ALWAYS returns the residual dict (the assertions fire only AFTER recording the residual fields
    used by the test's discriminator check).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    blocks, J, h, coupling, s_clamp, beta = make_dtm_negative_cell(rng, sizes, n_clamp, beta)
    n = sum(len(b) for b in blocks)
    S = spin_table(n)
    pi = boltzmann_clamped(S, J, h, coupling, s_clamp, beta)

    block_mats = [block_gibbs_matrix(pi, S, b) for b in blocks]

    out = {
        "n_free_spins": n,
        "n_superblocks": len(blocks),
        "superblock_names": list(SUPERBLOCK_NAMES[: len(blocks)]),
        "block_sites": [list(b) for b in blocks],
        "block_db": [],
        "block_inv": [],
    }
    if verbose:
        print(f"[DTM-negative shadow] n_free={n} superblocks={out['superblock_names']} "
              f"sites={out['block_sites']} clamp={s_clamp.tolist()} beta={beta}")

    # (1) each superblock update: π-stationary + π-reversible
    block_ok = True
    for bi, (b, PB) in enumerate(zip(blocks, block_mats)):
        inv = float(np.max(np.abs(pi @ PB - pi)))
        rev = max_asym(PB, pi)
        out["block_db"].append(rev)
        out["block_inv"].append(inv)
        block_ok = block_ok and (inv < TOL_INV) and (rev < TOL_SYM)
        if verbose:
            print(f"   superblock {bi} ({SUPERBLOCK_NAMES[bi]}) sites {b}: "
                  f"stationarity={inv:.2e}  detailed-balance={rev:.2e}")

    fwd = list(range(len(blocks)))
    rev_order = list(reversed(fwd))
    out["fwd_order"] = fwd
    out["rev_order"] = rev_order

    P_fwd = ordered_product(block_mats, fwd)
    P_rev = ordered_product(block_mats, rev_order)

    # (2) deterministic ordered product is (in general) NOT reversible — the kernel the patch replaces
    fwd_asym = max_asym(P_fwd, pi)
    out["P_fwd_db_residual"] = fwd_asym
    if verbose:
        print(f"   P_fwd (deterministic order {fwd}): detailed-balance residual={fwd_asym:.2e}  "
              f"(expect >> 0: NON-reversible deterministic scan = the kernel the patch replaces)")

    # (3) adjoint identity (P_fwd)* = P_rev
    adj_dev = float(np.max(np.abs(adjoint_in_pi(P_fwd, pi) - P_rev)))
    out["adjoint_dev"] = adj_dev
    if verbose:
        print(f"   adjoint identity  (P_fwd)* == P_rev :  max dev={adj_dev:.2e}")

    # (4) the PATCHED mixture K = ½(P_fwd + P_rev) = ½(P_AB + P_BA) IS reversible + stationary
    K = 0.5 * (P_fwd + P_rev)
    K_inv = float(np.max(np.abs(pi @ K - pi)))
    K_asym = max_asym(K, pi)
    out["K_db_residual"] = K_asym
    out["K_inv_residual"] = K_inv
    out["max_asym"] = K_asym   # the headline residual the brief reports
    if verbose:
        print(f"   K = 1/2(P_fwd + P_rev) = 1/2(P_AB + P_BA):  stationarity={K_inv:.2e}  "
              f"detailed-balance(max_asym)={K_asym:.2e}  (expect < {TOL_SYM:.0e}: REVERSIBLE)")

    passed = bool(
        block_ok
        and fwd_asym > MIN_NONREV
        and adj_dev < TOL_ADJ
        and K_inv < TOL_INV
        and K_asym < TOL_SYM
    )
    out["passed"] = passed

    # HARD-HALT assertions (residuals already recorded above)
    assert block_ok, f"a superblock Gibbs update is not π-reversible/stationary: db={out['block_db']}"
    assert fwd_asym > MIN_NONREV, (
        f"P_fwd unexpectedly reversible ({fwd_asym}); the cert has no discriminating teeth — "
        "the deterministic scan must be genuinely non-reversible")
    assert adj_dev < TOL_ADJ, f"(P_fwd)* != P_rev ({adj_dev})"
    assert K_inv < TOL_INV, f"K not π-stationary ({K_inv})"
    assert K_asym < TOL_SYM, (
        f"K = 1/2(P_AB+P_BA) NOT π-reversible ({K_asym}) — DB CERTIFICATE FAILED")
    if verbose:
        print(f"   => DTM-negative-shape K = 1/2(P_AB+P_BA) is π-reversible (max_asym={K_asym:.2e} "
              f"< {TOL_SYM:.0e}). DB-CERT: PASS")
    return out


def main():
    rng = np.random.default_rng(0)
    print("=== Detailed-balance certificate: reversible 1/2(P_AB+P_BA) DTM-negative kernel ===")
    # primary: the 4 single-site superblocks (forward [0,1,2,3] / reverse [3,2,1,0] DTM order)
    res = certify(rng, sizes=(1, 1, 1, 1))
    # robustness: uneven superblock sizes (still the 4-superblock DTM partition)
    print("\n[robustness] uneven superblock sizes (still the 4 DTM superblocks)")
    res2 = certify(np.random.default_rng(1), sizes=(2, 1, 1, 1))
    print(f"\nDB-CERT: PASS  (max_asym primary={res['max_asym']:.2e}, robustness={res2['max_asym']:.2e} "
          f"< {TOL_SYM:.0e}; deterministic scan non-reversible "
          f"{res['P_fwd_db_residual']:.2e}>{MIN_NONREV:.0e}; (P_fwd)*==P_rev)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
