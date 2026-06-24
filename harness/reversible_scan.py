"""Reversible-scan order-coin toggle helper + patch-live detector.

Ported from the wiki harness COPY-source
`internal-project/experiments/internal-exp/apply_shared_coin_toggle.py`
(the v2 shared/per-chain order-coin toggle). The PATCH itself lives in the vendored overlay
`vendor/thrml_overlay/thrml/block_sampling.py` (v1 reversible scan + v2 toggle, applied at build
time). THIS module is the *consumer-side* helper:

  * `is_patch_live()` — mirrors exp15's `is_patch_live`: asserts the reversible kernel + v2 order-coin
    toggle is actually the live `thrml.block_sampling.sample_blocks` (marker constant present AND the
    forward/reverse coin + `order_subkey` toggle are in the live source). A detector, not a patcher.
  * `make_order_key(...)` / `SHARED` / `PER_CHAIN` — the two order-coin modes the runner selects:
      - TRAINING uses a SHARED order_key (one coin per super-sweep across the whole batch -> non-batched
        under the chain-vmap -> `lax.cond` stays true control flow -> ONE sweep, the speedup).
      - DIAGNOSTICS use a PER-CHAIN order_key (each chain draws its own coin -> independent across-chain
        SEM). This is the `order_key=None` default path in the overlay.
    Each chain's MARGINAL kernel is the identical ½(P_fwd+P_rev) either way; the modes differ ONLY in
    cross-chain coin correlation.

Calls `bootstrap_paths()` on import so `import thrml` resolves to the patched overlay.
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

# Order-coin mode labels (the runner threads one of these into the overlay's `order_key`).
SHARED = "shared_per_sweep"        # training: one coin/super-sweep for the whole batch (fast path)
PER_CHAIN = "per_chain_per_sweep"  # diagnostics: independent coin per chain (order_key=None default)

# The constant the overlay defines when the reversible v2 patch is live (mirror exp15 marker check).
EXPECTED_MARKER = "HTDML-REVERSIBLE-SCAN-v2:fwd-rev-symmetrized-block-gibbs;K=half(P_AB+P_BA);order-coin-toggle"


def is_patch_live() -> tuple[bool, str]:
    """Detect whether the reversible ½(P_AB+P_BA) v2 overlay patch is the LIVE thrml kernel.

    Mirrors exp15's `is_patch_live`: checks (a) the marker constant is importable from
    `thrml.block_sampling`, and (b) the live `sample_blocks` source carries the v2 reversible
    forward/reverse coin + `order_subkey` toggle. Returns `(live, detail)`.
    """
    import inspect

    import thrml.block_sampling as bs

    marker = getattr(bs, "REVERSIBLE_SCAN_MARKER", None)
    if marker is None:
        return False, "REVERSIBLE_SCAN_MARKER absent from thrml.block_sampling"
    if marker != EXPECTED_MARKER:
        return False, f"REVERSIBLE_SCAN_MARKER mismatch: {marker!r} != {EXPECTED_MARKER!r}"

    try:
        src = inspect.getsource(bs.sample_blocks)
    except (OSError, TypeError) as exc:  # pragma: no cover - source must be available
        return False, f"could not read sample_blocks source: {exc}"

    required = [
        ("v2 marker", "HTDML-REVERSIBLE-SCAN PATCH v2"),
        ("fair coin", "bernoulli"),
        ("fwd/rev branch", "jax.lax.cond"),
        ("exact reversal", "reversed(fwd_order)"),
        ("order-coin toggle", "order_subkey"),
    ]
    missing = [name for name, tok in required if tok not in src]
    if missing:
        return False, "live sample_blocks missing reversible v2 tokens: " + ", ".join(missing)

    thrml_file = str(Path(bs.__file__).resolve())
    overlay_prefix = str((_REPO_ROOT / "vendor" / "thrml_overlay").resolve())
    if not thrml_file.startswith(overlay_prefix):
        return False, f"thrml.block_sampling resolved OUTSIDE the overlay: {thrml_file}"

    return True, (
        f"reversible v2 kernel live in overlay ({thrml_file}); marker + fwd/rev coin + "
        f"order_subkey toggle present"
    )


def make_order_key(key, mode: str):
    """Return the `order_key` to thread into the overlay sampler for a given coin mode.

    - `mode == PER_CHAIN` (diagnostics): returns `None` -> the overlay draws a per-chain coin from
      each chain's own key (independent across-chain SEM).
    - `mode == SHARED` (training): returns the supplied `key` -> a closure-constant order key shared
      across the chain-vmap (non-batched -> one sweep). The runner must pass a key NOT vmapped over
      the batch axis for this to stay non-batched.

    The kernel math is identical in both modes; only how the coin is realized across chains differs.
    """
    if mode == PER_CHAIN:
        return None
    if mode == SHARED:
        return key
    raise ValueError(f"unknown order-coin mode {mode!r}; expected {SHARED!r} or {PER_CHAIN!r}")


if __name__ == "__main__":
    live, detail = is_patch_live()
    print(f"is_patch_live() -> {live}\n  {detail}")
    raise SystemExit(0 if live else 1)
