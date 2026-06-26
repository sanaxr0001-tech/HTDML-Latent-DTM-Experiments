# htdml-latent-dtm — paid H200 Stage-C run REPORT (2026-06-26)

**Outcome: `Q-CALIBRATION-FAIL` (both seeds) — a calibration-GATE artifact, NOT bad mixing.**
MEASURE-ONLY: no wiki edit, no tag move.

## Provenance / config
- git_sha `57eac01`, jax backend `gpu`, reversible patch live (`is_patch_live=true`).
- MODE=full, SEEDS=1,2, λ_joint=1.0; frozen constants L_traj=400, N_chains=4, N_R=16, C=5.0, ESS_min=10.0.
- Budget: **9.29 GPU-h** of a 16.0 cap (no budget wall). Measured ~82 s/epoch (×2 seeds × 200-epoch Stage B).

## Per-seed calibration (the verdict driver)
| seed | τ̂ layer0 | τ̂ layer1 | τ̂ layer2 | τ̂ layer3 | worst τ̂ | cal_stable | failed_layer | Stage C? |
|------|------|------|------|------|------|------|------|------|
| 1 | 2.06 | 2.46 | 2.28 | 5.07 | 5.07 | **false** | **0** | not entered |
| 2 | 1.97 | 4.00 | 3.19 | 1.42 | 4.00 | **false** | **0** | not entered |

## Interpretation (factual)
- **The trained DTM mixes excellently.** Every τ̂ is tiny (2–5), far inside the adequacy threshold:
  `L_traj ≥ C·τ̂` ⇒ need L_traj ≥ ~25; we ran 400 → a **16× margin**. Consistent with the Stage-B
  ACP autocorrelations, which sat at ~0.00–0.05 (worst step) across all 200 epochs for both seeds.
- **cal_stable=False fired at layer 0 — the FASTEST-mixing layer (smallest τ̂ ≈ 2).** The
  doubling-stability test (`classify_calibration_stable`, T_O measured at L=400/800/1600) mis-fired:
  at very small τ̂ the autocorrelation decays almost immediately, so the doubling curve is
  **noise-dominated** and the stability criterion can't lock a stable T_O. **It failed for the OPPOSITE
  reason to stickiness** — the model mixes *too fast* for that test to register a stable autocorrelation time.
- **Reproducible across both independent seeds** (same `failed_layer=0`, same small-τ̂ regime) ⇒
  **SYSTEMATIC** (calibration-criterion × architecture), not seed luck. (This is why letting seed 2 run
  was worth it: it confirmed reproducibility and gave us both seeds' τ̂, which a mid-run stop would have lost.)
- **The joint steering (Stage C) was NEVER reached** for either seed ⇒ **no result on the actual
  hypothesis** (does guarded joint steering preserve per-layer mixing + image quality). The run was
  blocked at the measurement gate, before the novel mechanism was exercised.

## Relation to prior work
- Mirrors the wiki's **exp13** finding ("T_O calibratable; S-ADQ was a gate-spec artifact, windows fine"):
  the cal-stability gate spec can reject an adequate, well-mixing model. (Reference only — no tag move.)

## Recommended next step (researcher decision — NOT auto-applied)
The model is healthy; the **gate** is the blocker. Candidate fixes (pick at conferral):
1. **Make cal_stable regime-aware:** accept when `L_traj ≥ C·τ̂` is comfortably satisfied (it is, 16×)
   instead of additionally requiring doubling-stability that noise-fails at τ̂≈2.
2. **Widen the doubling tolerance / use a more robust small-τ̂ T_O estimator.**
- Then re-run A→B→C (≈9–10 GPU-h for 2 seeds; Stage B must be re-trained — checkpoints were not persisted
  to a reusable path). Only then does the joint-steering test actually run.
- This is a pre-Stage-C gate fix, not a change to the steering/guard logic or any frozen acceptance threshold.
