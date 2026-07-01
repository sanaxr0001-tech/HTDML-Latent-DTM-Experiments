# htdml-latent-dtm

**Scope:** Isolated feasibility and signal study — NOT operational-theorem validation, NOT a
hardware-energy claim. This companion repo implements the guarded joint encoder–binary-latent-DTM–decoder
pipeline for the HTDML latent-DTM feasibility study.

---

## The one claim

> Guarded joint training (encoder + binary-latent-DTM + decoder) retains image quality while keeping
> a measured finite-budget mixing margin over the registered gradient-observable probe set —
> worst of N_R ≈ 16 Rademacher sketches — across all 4 reverse EBM layers.

This is a feasibility/signal claim. A positive result is evidence that the joint-training objective
does not destroy the mixing budget; it does not validate any operational trainability theorem.

---

## Isolation discipline

- **Vendored read-only inputs:** `vendor/dtm-replication/` (clean @ 7c22d19) and
  `vendor/thrml_overlay/thrml/` (0.1.3 + Task-2 reversible-scan patch) are the only mutable
  dependency surfaces.
- **Companion is the only mutable target:** `src/htdml/`, `harness/`, `tests/`, `scripts/` are
  written here and nowhere else.
- **Outcome tokens only:** results are reported as the companion-local outcome tokens below;
  they are not claim-status tags.
- **Path isolation:** `src/htdml/paths.py::bootstrap_paths()` prepends the vendor paths at import
  time, shadowing conda site-packages thrml. All scripts and tests must `import htdml` before
  importing `thrml` or `thrmlDenoising`.

---

## 6-token outcome vocabulary

Results are expressed using exactly these tokens (companion-local; never wiki tags):

| Token | Meaning |
|-------|---------|
| `HTDML-MARGIN-POSITIVE` | Joint training retains image quality AND mixing margin across all gates |
| `HTDML-MARGIN-NEGATIVE` | Mixing margin gate fails (finite-budget probe below ESS_min threshold) |
| `QUALITY-LOSS` | FID gate fails (decoded image quality drops below acceptable threshold) |
| `PLATEAU-UNRESOLVED` | Trajectory too short to certify autocorrelation (L_traj < C·τ̂) |
| `Q-CALIBRATION-FAIL` | Probe calibration gate fails (gradient-SNR estimate not well-defined) |
| `BUDGET-WALL` | Compute budget exceeded before any gate is cleared |

**Two-seed aggregation rule:** the overall result is `HTDML-MARGIN-POSITIVE` if and only if
BOTH seeds pass all final gates independently. Any failure in either seed yields the corresponding
negative token.

---

## Quick start

```bash
# All commands use conda base python
python -m pytest tests/ -v
```

Pins are recorded in `PINS.md`. Config constants are frozen there; do not change them without
updating the corresponding sha256 verification in `tests/test_isolation.py`.
