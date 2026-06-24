"""TrainabilityProbe — per-layer frozen-θ negative-phase trainability probe (the MEASUREMENT side).

For a given reverse **layer = diffusion step (0..3)** of the LatentDTM, on the FROZEN-θ negative
phase, this computes the operational gradient ``g = E_data[f] − E_model[f]`` and the K=50-convention
mixing scalars (τ_int,Y, ESS_hat, Q_struct^⊥) that the Task-9 driver's gates read.

The measurement pattern is PORTED from the wiki's exp16 operational-validation runner
(``…/experiments/internal-exp/exp16_validate.py``):

  * ``build_observable_maps`` / ``positive_moments``   → the POSITIVE (data-clamped) phase, plain
    ``estimate_moments`` Gibbs (fast, subdominant per F4);
  * exp16's PT cold-chain negative moments             → REPLACED here by the companion's
    SINGLE-replica REVERSIBLE kernel (``sample_with_observation`` through the live ½(P_AB+P_BA)
    overlay, ``order_key=None`` = per-chain diagnostics), NOT a PT ladder;
  * ``run_calibration``                                → the per-layer T_O doubling-stability probe;
  * ``g = E_data[f] − E_model[f]`` / ``Q_op`` / ``Q_struct^⊥``  → the operational read.

Everything that touches sokal / Rademacher / K=50 scalars is DELEGATED to
``harness.probe_primitives`` (``sokal_profile_from_spins``, ``probe_scalars``, the Rademacher
worst-of-N_R) — this module does NOT re-derive that math.

THE MANDATORY GUARD (the exp15/16 bug): before ANY sampling for a layer, the step's negative
program is refreshed from the CURRENT trained globals (``pp.refresh_program_weights``) and the
HARD-HALT ``pp.refreshed_weight_proof(step)`` (``constructor_was_stale ∧ refresh_ok``) is asserted.
``AnnealingIsingSamplingProgram``'s constructor reads stale INIT ``model.factors``; without the
refresh the probe would sample INIT weights (exp15 P0-RESOLVED + exp16 F4-fail were INVALID for the
trained DTM for exactly this reason — exp15-recheck confirmed).

T_O / Q bias caveat (build-notes §"Half-Sokal T_O bias"): the verbatim half-Sokal T_O estimator is
systematically ~0.86× the exact T_O (so τ_int,Y ~0.86× low, ESS_hat / Q ~1.16× high).  The probe
returns the BIASED values AS-IS (byte-identical to the validated wiki math); the driver uses RELATIVE
comparisons + Task-12 calibrated thresholds, so the systematic bias CANCELS.  Do NOT "correct" it.

CPU: ``dtm.train`` is GPU-only; the probe NEVER trains.  It is unit-tested on the small REAL 4_4
fixture DTM (perturbed via the exact ``eqx.tree_at`` write-back, ``model.factors`` left stale).
"""

from __future__ import annotations

import sys
from pathlib import Path

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

from harness import probe_primitives as pp  # noqa: E402

# K=50 retained-window convention (build-notes §E1; mirrors pp.K_WINDOW / B_WARMUP / STRIDE_SWEEPS).
K_WINDOW = pp.K_WINDOW          # 50
B_WARMUP = pp.B_WARMUP          # 400
STRIDE_SWEEPS = pp.STRIDE_SWEEPS  # 8


def _resolve_dtm(model):
    """Accept a raw vendored ``DTM`` OR a ``LatentDTM`` wrapper (``.dtm``)."""
    return model.dtm if hasattr(model, "dtm") and hasattr(model.dtm, "steps") else model


def _tile_clamp(data, n):
    """Broadcast each clamped-block tuple to the chain axis (exp16 tile_clamp)."""
    return [jnp.broadcast_to(a, (n,) + a.shape[1:]) for a in data]


def _blocks_to_free_spins(traj_blocks):
    """Concatenate per-free-block retained states → (n_chains, L, n_free) spins in {−1,+1}.

    ``traj_blocks`` is the StateObserver output under a chain-vmap: a list (one per free block, in
    ``free_blocks`` order) of arrays ``(n_chains, L, block_len)`` of bit {0,1}.  This concatenation
    order MATCHES ``build_maps``' ``free_global`` ordering (both iterate ``free_blocks`` in order),
    so the resulting columns align with the exp15 observable maps.  bit→spin via ``2x−1`` (the exp15
    ``_blocks_to_free_spins`` convention, ising.py:204)."""
    cat = jnp.concatenate([jnp.asarray(b) for b in traj_blocks], axis=-1)
    return 2.0 * np.asarray(cat).astype(np.float64) - 1.0


class TrainabilityProbe:
    """Per-layer frozen-θ negative-phase trainability probe.

    The single public entry is :meth:`evaluate` (one layer) / :meth:`evaluate_model` (all 4
    layers); :meth:`calibrate` exposes the per-layer T_O doubling-stability calibration the driver
    uses for the Q-CALIBRATION-FAIL gate.
    """

    # The 7 headline scalars (build-notes §"Probe K=50 convention" / brief §5).
    HEADLINE_KEYS = (
        "r_grad[1]", "r_grad[50]", "tau_int_Y", "ESS_hat", "Q_struct_perp", "gradient_norm", "layer",
    )

    # Diagnostics use the PER-CHAIN reversible kernel (order_key=None default in the overlay).
    ORDER_KEY = None

    # ====================================================================== map builders (exp16 port)
    @staticmethod
    def build_observable_maps(step):
        """exp16 build_observable_maps (verbatim port): observable node/edge maps for
        ``estimate_moments`` + alignment.  Same [edges (base_graph_edges), biases (output+hidden)]
        ordering as exp15 ``build_maps`` / ``_obs_chunk`` (asserted in :meth:`_assert_maps_aligned`)."""
        g = step.model.graph
        node_map = g.node_mapping
        bias_nodes = list(g.output_nodes) + list(g.hidden_nodes)
        weight_edges = list(g.base_graph_edges)
        weight_edge_tuples = [(e.connected_nodes[0], e.connected_nodes[1]) for e in weight_edges]
        free_blocks = list(step.training_spec.program_negative.gibbs_spec.free_blocks)
        free_nodes = [n for blk in free_blocks for n in blk]
        free_global = [node_map[n] for n in free_nodes]
        pos_of_global = {gid: i for i, gid in enumerate(free_global)}

        def gpos(node):
            return pos_of_global[node_map[node]]

        edge_pos0 = np.array([gpos(e.connected_nodes[0]) for e in weight_edges], dtype=np.int32)
        edge_pos1 = np.array([gpos(e.connected_nodes[1]) for e in weight_edges], dtype=np.int32)
        bias_pos = np.array([gpos(n) for n in bias_nodes], dtype=np.int32)
        return dict(bias_nodes=bias_nodes, weight_edges=weight_edges,
                    weight_edge_tuples=weight_edge_tuples, free_blocks=free_blocks,
                    edge_pos0=edge_pos0, edge_pos1=edge_pos1, bias_pos=bias_pos,
                    n_edge=len(weight_edges), n_bias=len(bias_nodes), n_free=len(free_global))

    @staticmethod
    def _assert_maps_aligned(maps15, maps3):
        """exp16 G-obs-align runtime assert: exp15 ``_obs_chunk`` positions == exp3 estimate_moments
        positions (so the cold-chain Sokal observables and the operational g share an ordering)."""
        ok = bool(maps15["n_edge"] == maps3["n_edge"] and maps15["n_bias"] == maps3["n_bias"]
                  and np.array_equal(maps15["edge_pos0"], maps3["edge_pos0"])
                  and np.array_equal(maps15["edge_pos1"], maps3["edge_pos1"])
                  and np.array_equal(maps15["bias_pos"], maps3["bias_pos"]))
        assert ok, "observable maps (exp15 cold-chain vs exp3 positive-phase) are MISALIGNED"

    # ====================================================================== MANDATORY refresh guard
    @staticmethod
    def _refresh_and_assert(step):
        """The exp15/16 stale-factors HARD-HALT.  Refresh the step's NEGATIVE program from the CURRENT
        trained globals and assert the proof (``constructor_was_stale ∧ refresh_ok``).  Returns the
        refreshed negative program (caller samples through it) + the proof dict.  RAISES if the bug
        premise fails OR the refresh did not take."""
        from thrmlDenoising.annealing_graph_ising import AnnealingIsingSamplingProgram

        ts = step.training_spec
        # build a FRESH negative program (constructor reads stale INIT factors) ...
        prog = AnnealingIsingSamplingProgram(
            step.model, list(ts.program_negative.gibbs_spec.free_blocks),
            list(ts.program_negative.gibbs_spec.clamped_blocks), jnp.asarray(1.0), ts.schedule_negative)
        # ... then refresh it with the trained weights (the fix).
        prog = pp.refresh_program_weights(prog, step)
        proof = pp.refreshed_weight_proof(step)
        assert proof["constructor_was_stale"] is True, (
            "refresh guard VACUOUS: AnnealingIsingSamplingProgram constructor was NOT stale "
            f"(stale_vs_trained_maxabs={proof.get('stale_vs_trained_maxabs')}) — the exp15/16 bug "
            "premise does not hold; aborting the probe")
        assert proof["refresh_ok"] is True, (
            "trained-weight refresh did NOT take "
            f"(refreshed_vs_trained_maxabs={proof.get('refreshed_vs_trained_maxabs')}) — the probe "
            "would sample INIT weights (the exp15/16 bug); aborting the probe")
        return prog, proof

    # ====================================================================== negative-phase sampler
    def _negative_trajectory(self, step, prog_neg, maps15, data_neg, n_chains, K, B, stride, key):
        """Sample a retained negative-phase trajectory of FREE spins through the SINGLE-replica
        REVERSIBLE kernel (per-chain, ``order_key=None``).  Returns ``rec`` of shape
        ``(n_chains, K, n_free)`` in {−1,+1}, aligned to the exp15 ``build_maps`` free ordering.

        The negative phase frees the 4 superblocks {upper_hidden, lower_hidden, image_output,
        label_output} and clamps b_t (build-notes §"Training-negative free set").  ``data_neg`` is the
        clamped-block tuple from ``step._make_training_data`` (= b_t)."""
        from thrml.block_sampling import SamplingSchedule, sample_with_observation
        from thrml.observers import StateObserver
        from thrmlDenoising.annealing_graph_ising import hinton_init_from_graph

        spec = prog_neg.gibbs_spec
        free_blocks = list(spec.free_blocks)
        beta = float(step.training_spec.beta)
        observer = StateObserver(free_blocks)
        sched = SamplingSchedule(int(B), int(K), int(stride))

        k_init, k_run = jr.split(key)
        init_state = hinton_init_from_graph(k_init, step.model, free_blocks, n_chains, beta)
        clamp = _tile_clamp(data_neg, n_chains)
        keys = jr.split(k_run, n_chains)

        def one(kc, init_per, clamp_per):
            _carry, traj = sample_with_observation(
                kc, prog_neg, sched, list(init_per), list(clamp_per),
                observer.init(), observer, order_key=self.ORDER_KEY)  # ORDER_KEY=None → per-chain
            return traj

        traj = jax.vmap(one, in_axes=(0, 0, 0))(keys, tuple(init_state), tuple(clamp))
        return _blocks_to_free_spins(traj)              # (n_chains, K, n_free) in {−1,+1}

    def _negative_window_means(self, rec, maps15):
        """Per-chain window-mean of f_a from a negative-phase rec (n_chains, K, n_free) → (n_chains, P)
        ordered [edges, biases] (exp16 obs_means_from_rec, full window)."""
        n_chains, K, _ = rec.shape
        P = maps15["n_edge"] + maps15["n_bias"]
        out = np.empty((n_chains, P), dtype=np.float64)
        for lo in range(0, P, pp.OBS_CHUNK):
            hi = min(lo + pp.OBS_CHUNK, P)
            fc = np.asarray(pp._obs_chunk(rec, maps15, lo, hi))   # (n_chains, K, chunk)
            out[:, lo:hi] = fc.mean(axis=1)
        return out

    def _negative_node_means_for_clamp(self, step, maps15, clamp_bits, *, n_chains, K, B, stride, key):
        """[test helper] Estimate E_model[node spins] at a FIXED b_t clamp (bool bits) via the
        per-chain reversible kernel — for the g-vs-exact sanity test.  Refreshes first (guard)."""
        prog_neg, _ = self._refresh_and_assert(step)
        data_neg = [jnp.asarray(np.asarray(clamp_bits, dtype=bool))[None, :]]   # (1, n_clamp) bool
        rec = self._negative_trajectory(step, prog_neg, maps15, data_neg,
                                        n_chains, K, B, stride, key)
        # node spins occupy columns [n_edge : n_edge+n_bias) in the [edges, biases] ordering; but rec is
        # the FREE-spin trajectory (n_chains, K, n_free) — the bias columns are the free node spins at
        # maps["bias_pos"].  Return the mean spin per node (free ordering).
        node_spins = rec[:, :, maps15["bias_pos"]]            # (n_chains, K, n_bias)
        return node_spins.reshape(-1, node_spins.shape[-1]).mean(axis=0)

    # ====================================================================== positive-phase (exp16)
    @staticmethod
    def _positive_window_means(step, key, prog_pos, data_pos, n_chains, maps3, K, warm):
        """exp16 positive_moments: per-chain window moment of the POSITIVE (data-clamped) phase via
        plain ``estimate_moments`` Gibbs.  Returns (n_chains, P) ordered [edges, biases]."""
        from thrml.block_sampling import SamplingSchedule
        from thrml.models.ising import estimate_moments
        from thrmlDenoising.annealing_graph_ising import hinton_init_from_graph

        pos_free = prog_pos.gibbs_spec.free_blocks
        sched = SamplingSchedule(int(warm), int(K), 1)
        k_init, k_mom = jr.split(key)
        init = hinton_init_from_graph(k_init, step.model, list(pos_free), n_chains,
                                      step.training_spec.beta)
        clamp = _tile_clamp(data_pos, n_chains)
        keys = jr.split(k_mom, n_chains)

        def one(k, init_per, clamp_per):
            return estimate_moments(k, maps3["bias_nodes"], maps3["weight_edge_tuples"],
                                    prog_pos, sched, list(init_per), list(clamp_per))

        nb, ew = jax.vmap(one, in_axes=(0, 0, 0))(keys, tuple(init), tuple(clamp))
        return np.concatenate([np.asarray(ew), np.asarray(nb)], axis=-1)   # [edges, biases]

    # ====================================================================== the per-layer evaluate
    def evaluate(self, model, layer, batch, *, n_R, L_traj, n_chains, diag_key,
                 B=B_WARMUP, stride=STRIDE_SWEEPS, K=K_WINDOW, warm_pos=None, key=None):
        """Probe ONE reverse layer (= diffusion step) of the model on the FROZEN-θ negative phase.

        Parameters
        ----------
        model : LatentDTM | DTM
            The model whose ``.steps[layer]`` is probed.
        layer : int
            The diffusion step index (0..3).
        batch : dict
            ``{"image": (N, n_img) bool, "label": (N, n_lab) bool, "idx": int}`` — the single-input
            clamp source (exp16 phase_data_1 pattern); ``idx`` selects the row.
        n_R : int
            Number of FIXED shared Rademacher sketches (worst-of-N_R reduction).
        L_traj : int
            Retained Y-process length (≫ K; B=400, stride=8 retain points).
        n_chains : int
            Negative-phase chains.
        diag_key : int
            Deterministic seed for the Rademacher sketches (reproducible).
        B, stride, K : int
            The K=50 window convention (B=400 warm, stride=8, K=50 — defaults pinned).
        warm_pos : int, optional
            Positive-phase warmup (defaults to B).
        key : jax PRNGKey, optional
            Probe RNG (defaults to ``jr.PRNGKey(diag_key)`` so the read is reproducible).

        Returns
        -------
        dict
            The 7 headline scalars {r_grad[1], r_grad[50], tau_int_Y, ESS_hat, Q_struct_perp,
            gradient_norm, layer} (worst-of-N_R) + underscore diagnostics (_g, _T_O_Y,
            _per_sketch_tau, _refresh_proof, worst_sketch_idx).
        """
        dtm = _resolve_dtm(model)
        step = dtm.steps[int(layer)]
        if key is None:
            key = jr.PRNGKey(int(diag_key))
        if warm_pos is None:
            warm_pos = int(B)

        # --- (1) MANDATORY per-layer trained-weight refresh + HARD-HALT proof (BEFORE any sampling) ---
        prog_neg, refresh_proof = self._refresh_and_assert(step)

        # --- maps (exp15 cold-chain + exp3 positive-phase) + alignment ---
        maps15 = pp.build_maps(step)
        maps3 = self.build_observable_maps(step)
        self._assert_maps_aligned(maps15, maps3)
        P = maps15["n_edge"] + maps15["n_bias"]

        # --- probe-local single-input clamps (negative b_t + positive data) ---
        img = jnp.asarray(batch["image"])
        lab = jnp.asarray(batch["label"])
        idx = int(batch.get("idx", 0))
        k_data, k_neg, k_pos, k_traj = jr.split(key, 4)
        img1, lab1 = img[idx:idx + 1], lab[idx:idx + 1]
        data_pos, data_neg = step._make_training_data(k_data, img1, lab1)

        prog_pos = step.training_spec.program_positive

        # --- (2) gradient g = E_data[f] − E_model[f] (per-chain means averaged over chains) ---
        neg_rec_g = self._negative_trajectory(step, prog_neg, maps15, data_neg,
                                              n_chains, K, B, stride, k_neg)
        E_model = self._negative_window_means(neg_rec_g, maps15).mean(axis=0)      # (P,)
        E_data = self._positive_window_means(step, k_pos, prog_pos, data_pos,
                                             n_chains, maps3, K, warm_pos).mean(axis=0)
        g = E_data - E_model                                                       # (P,)
        gradient_norm = float(np.linalg.norm(g))

        # --- (3) negative-phase autocorr trajectory: the K=50 Y-process (retained L_traj samples) ---
        #     Y_i = X_{B + i·stride}: B warm sweeps then L_traj retained, stride between retains.
        neg_traj = self._negative_trajectory(step, prog_neg, maps15, data_neg,
                                             n_chains, int(L_traj), B, stride, k_traj)
        retained_Y_obs = self._retained_observables(neg_traj, maps15)   # (n_chains, L_traj, P)

        # --- (4) Rademacher worst-of-N_R + the K=50 derived scalars (DELEGATED to probe_primitives) ---
        scal = pp.probe_scalars(retained_Y_obs, n_R=int(n_R), diag_key=int(diag_key), gradient=g)
        sk = pp.rademacher_sketch_scalars(retained_Y_obs, n_R=int(n_R), diag_key=int(diag_key))

        # --- (5) assemble the headline dict (worst-of-N_R) ---
        out = {
            "r_grad[1]": scal["r_grad[1]"],
            "r_grad[50]": scal["r_grad[50]"],
            "tau_int_Y": scal["tau_int_Y"],
            "ESS_hat": scal["ESS_hat"],
            "Q_struct_perp": scal["Q_struct_perp"],
            "gradient_norm": scal["gradient_norm"],
            "layer": int(layer),
            # diagnostics / verification (underscore = not a headline scalar)
            "_g": g,
            "_T_O_Y": scal["_T_O_Y"],
            "_per_sketch_tau": sk["per_sketch_tau"],
            "_per_sketch_T_O": sk["per_sketch_T_O"],
            "worst_sketch_idx": scal["worst_sketch_idx"],
            "n_R": int(n_R),
            "P": int(P),
            "_refresh_proof": refresh_proof,
        }
        return out

    @staticmethod
    def _retained_observables(rec, maps15):
        """Materialize the retained Y-process observables f_a (edge products + node spins) from a free-
        spin trajectory rec (n_chains, L_traj, n_free) → (n_chains, L_traj, P) ordered [edges, biases].

        This is the input ``pp.probe_scalars`` / the Rademacher sketches consume (τ_int,Y, T_O,Y on the
        retained process).  Uses the verbatim exp15 ``_obs_chunk`` to build [edge products, node spins]
        without ever materializing the full f."""
        n_chains, L, _ = rec.shape
        P = maps15["n_edge"] + maps15["n_bias"]
        obs = np.empty((n_chains, L, P), dtype=np.float64)
        for lo in range(0, P, pp.OBS_CHUNK):
            hi = min(lo + pp.OBS_CHUNK, P)
            obs[:, :, lo:hi] = np.asarray(pp._obs_chunk(rec, maps15, lo, hi))
        return obs

    # ====================================================================== all 4 layers
    def evaluate_model(self, model, batch, *, n_R, L_traj, n_chains, diag_key, **kw):
        """Probe EVERY diffusion step (0..num_diffusion_steps−1).  Returns a list of per-layer dicts —
        EXACTLY one per diffusion step (the companion has 4).  ``batch`` is shared across layers; the
        probe RNG is folded per layer so the layers are independent reads."""
        dtm = _resolve_dtm(model)
        n_layers = len(dtm.steps)
        records = []
        for layer in range(n_layers):
            key = jr.fold_in(jr.PRNGKey(int(diag_key)), int(layer))
            records.append(self.evaluate(model, layer, batch, n_R=n_R, L_traj=L_traj,
                                         n_chains=n_chains, diag_key=diag_key, key=key, **kw))
        return records

    # ====================================================================== per-layer T_O calibration
    def calibrate(self, model, layer, batch, *, n_chains, L0, warm, n_rungs, diag_key, key=None,
                  B=None, stride=STRIDE_SWEEPS, sokal_c=5.0, stab_tol=0.05):
        """Per-layer T_O DOUBLING-STABILITY calibration (exp16 ``run_calibration`` port, single-replica
        reversible kernel).  Doubles the trajectory length L over ``n_rungs`` rungs; the read is
        Cal-STABLE once the per-observable S_a profile stops moving (L1 change < ``stab_tol``) AND the
        rung is self-consistent (L ≥ ``sokal_c``·τ_max).

        Returns
        -------
        dict
            ``{tau_hat, T_O, cal_stable, curve}`` — the driver's Q-CALIBRATION-FAIL gate reads
            ``cal_stable`` (build-notes: the bias is systematic; gates are RELATIVE).
        """
        dtm = _resolve_dtm(model)
        step = dtm.steps[int(layer)]
        if key is None:
            key = jr.PRNGKey(int(diag_key) + 991)
        if B is None:
            B = int(warm)

        prog_neg, _proof = self._refresh_and_assert(step)
        maps15 = pp.build_maps(step)

        img = jnp.asarray(batch["image"]); lab = jnp.asarray(batch["label"])
        idx = int(batch.get("idx", 0))
        k_data, k_run = jr.split(key)
        _data_pos, data_neg = step._make_training_data(k_data, img[idx:idx + 1], lab[idx:idx + 1])

        curve = []
        prev_S = None
        L = int(L0)
        w = int(warm)
        tau_star = TO_star = None
        cal_stable = False
        for d in range(int(n_rungs)):
            rk = jr.fold_in(k_run, d)
            rec = self._negative_trajectory(step, prog_neg, maps15, data_neg,
                                            int(n_chains), L, int(B), int(stride), rk)
            tau_max, T_O, S_a = pp.sokal_profile_from_spins(rec, maps15)
            sc = bool(L >= sokal_c * tau_max)
            dS_l1 = (float(np.abs(S_a - prev_S).sum() / max(S_a.sum(), 1e-12))
                     if prev_S is not None else None)
            curve.append(dict(L=int(L), warm=int(w), tau_max=float(tau_max), T_O=float(T_O),
                              self_consistent=sc, dS_l1=dS_l1))
            tau_star, TO_star = float(tau_max), float(T_O)
            # Cal-STABLE: self-consistent rung AND the S_a profile stopped moving across the doubling.
            if sc and dS_l1 is not None and dS_l1 < stab_tol:
                cal_stable = True
                break
            prev_S = S_a
            w = max(w, int(round(5 * tau_max)))
            L *= 2
        return dict(tau_hat=tau_star, T_O=TO_star, cal_stable=bool(cal_stable), curve=curve)
