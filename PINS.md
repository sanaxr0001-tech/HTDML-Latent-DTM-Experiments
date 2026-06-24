# PINS — frozen source and config constants

This file records the pinned inputs for the `htdml-latent-dtm` companion study.
**Do not edit without updating the corresponding sha256 verification in `tests/test_isolation.py`.**

---

## Source pins

| Source | Pin | Notes |
|--------|-----|-------|
| dtm-replication | @ 7c22d19 | Vendored clean at `vendor/dtm-replication/`; `figures/` excluded. DTM.py sha256 `e7d48e2304e7667c55a0862a1155e08ef340b7869fc1814f4b0b7ef27913f472` (proves vendored clean, not dirty worktree). |
| wiki (internal-project) | @ 4ef4063 | Harness COPY-source only, not vendored. Used as reference for exp15/exp19 harness patterns. |
| thrml | == 0.1.3 | Overlay copy at `vendor/thrml_overlay/thrml/`; the reversible-scan patch is applied in Task 2. Site-packages thrml 0.1.3 is shadowed by the overlay at runtime via `src/htdml/paths.py`. |

---

## FID ref pin

| File | sha256 |
|------|--------|
| `vendor/dtm-replication/thrmlDenoising/fid/precomputed_stats/bw_fashion_mnist_train.npz` | `66003004dc99115b20c146bd3c2a7d9d85fb85a3c0c9e991f11951933f97c5d8` |

---

## Verified config constants

Copied verbatim from `.superpowers/sdd/build-notes.md`. Companion divergences from upstream are flagged.

| param | value | source / note |
|---|---|---|
| graph_preset_architecture | **44_12** | **COMPANION DIVERGENCE** (upstream 60_12). 1936 nodes, side 44; valid (embeds 196+labels). Not minimal (20_8/42_8 smaller) but FROZEN; comfortable on the 8GB 4060 (< wiki's 60_12). |
| base_graph_manager | poisson_binomial_ising_graph_manager | upstream; MUST assert (convolved breaks marginalization) |
| num_label_spots | 5 | upstream training_script.py:32 |
| grayscale_levels | 1 | upstream (identity pixel converter) |
| torus | True | upstream |
| num_diffusion_steps | **4** | **COMPANION DIVERGENCE** (upstream default 1) → (c,)*4 valid; extend_params_or_zeros pads-with-last |
| kind | log | upstream |
| diffusion_offset | 0.1 | upstream |
| image_rate / label_rate | 0.8 / 0.2 | upstream (forward noising) |
| batch_size | 400 | upstream |
| n_samples (= K) | 50 | upstream → window_samples_K=50 |
| steps_per_sample (= stride) | 8 | upstream → stride_sweeps=8 |
| steps_warmup (= B) | 400 | upstream → window_span_sweeps=400 = 50×8 |
| training_beta | 1.0 | upstream |
| adaptive_cp | **True** | **COMPANION DIVERGENCE** (upstream False) — the ACP mixing-aware mechanism |
| correlation_penalty (c0,)*4 | c0 = 0.001 | RESOLVED: upstream is (0.0,) w/ ACP off; companion enables ACP & seeds at cp_min. `adapt_param` (utils.py:142-149) is multiplicative but DTM.py:358 re-floors to cp_min EACH adaptive epoch → seed immaterial after epoch 0. 0.001 honors plan's "nonzero seed". |
| cp_min | 0.001 | upstream |
| adaptive_threshold | 0.016 | upstream (NOT the ACP seed — plan conflated them) |
| step_learning_rates | (0.05,) | upstream |
| momentum / b2_adam | 0.9 / 0.999 | upstream |
| alpha_cosine_decay | 0.2 | upstream |
| n_epochs_for_lrd | 50 | upstream |

---

## Probe constants (K=50 convention)

| param | value | note |
|---|---|---|
| stride_sweeps | 8 | steps_per_sample |
| window_samples_K | 50 | n_samples = K |
| window_span_sweeps | 400 | = 50 × 8 = steps_warmup (B) |

---

## TBD-at-step placeholders

These values are NOT invented; they are frozen at their respective calibration gates.

| Item | Status |
|------|--------|
| dataset-split sha256 | `9e7e99291d61ddfc8623256c146d49a9203938f8b02fb994b7ee6fdd64f4fd8b` |
| InceptionV3-FID-weights sha256 | `4e030efa5bccac3222d975f658d1884f9e00fab24f2812082884539220b90d77` |
| Probe acceptance constants: L_traj, N_chains, N_R (target ≈16), C, ESS_min | TBD — frozen at Task 9 local calibration |
