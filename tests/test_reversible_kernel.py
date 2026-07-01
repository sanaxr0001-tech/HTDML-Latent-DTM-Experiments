"""Task 2 — reversible kernel overlay + detailed-balance certificate (TDD gate).

The detailed-balance certificate is the gate: it is MANDATORY before any GPU plan, so it must be
unambiguous in the test output. Tests:

  1. `harness.reversible_scan.is_patch_live()` returns True for the overlay (marker present + the
     reversible v2 forward/reverse coin + order_subkey toggle are the LIVE thrml.block_sampling kernel).
  2. The DB certificate PASSES (max_asym < 1e-10) on the production-shape (44_12-structured / 4 DTM
     superblocks {upper_hidden, lower_hidden, image_output, label_output} + clamped b_t) enumerable shadow.
  3. SANITY / discriminator: the unpatched DETERMINISTIC scan (P_fwd) FAILS detailed balance (residual
     ~1e-2) — proving the cert actually discriminates reversible from non-reversible.

conftest.py installs the vendored isolation; harness modules also self-bootstrap on import.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# conftest installs src/ + vendored paths; be explicit so the file is runnable directly too.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import htdml  # noqa: E402,F401  (triggers bootstrap_paths)

from harness import reversible_scan, selfadjoint_cert  # noqa: E402

OVERLAY_PREFIX = str((_REPO_ROOT / "vendor" / "thrml_overlay").resolve())


# --------------------------------------------------------------------------- (1) patch-live detector
def test_marker_constant_present():
    """The overlay must define the patch-live marker constant (mirror the reference is_patch_live)."""
    import thrml.block_sampling as bs

    assert hasattr(bs, "REVERSIBLE_SCAN_MARKER"), "REVERSIBLE_SCAN_MARKER absent from overlay"
    assert bs.REVERSIBLE_SCAN_MARKER == reversible_scan.EXPECTED_MARKER


def test_is_patch_live_true():
    """is_patch_live() returns True: the reversible v2 kernel is the LIVE thrml.block_sampling."""
    live, detail = reversible_scan.is_patch_live()
    assert live, f"reversible v2 patch is NOT live: {detail}"
    assert "overlay" in detail


def test_live_kernel_resolves_to_overlay():
    """The live thrml.block_sampling must be the vendored overlay copy, not conda site-packages."""
    import thrml.block_sampling as bs

    assert str(Path(bs.__file__).resolve()).startswith(OVERLAY_PREFIX)


def test_live_sample_blocks_has_reversible_v2_tokens():
    """The live sample_blocks source carries the v1 reversible scan + v2 order-coin toggle."""
    import inspect

    import thrml.block_sampling as bs

    src = inspect.getsource(bs.sample_blocks)
    for tok in ("HTDML-REVERSIBLE-SCAN PATCH v2", "bernoulli", "jax.lax.cond",
                "reversed(fwd_order)", "order_subkey"):
        assert tok in src, f"live sample_blocks missing reversible-v2 token {tok!r}"


def test_order_coin_toggle_modes():
    """The toggle helper maps the two modes correctly (PER_CHAIN -> None, SHARED -> key)."""
    import jax

    key = jax.random.PRNGKey(0)
    assert reversible_scan.make_order_key(key, reversible_scan.PER_CHAIN) is None
    shared = reversible_scan.make_order_key(key, reversible_scan.SHARED)
    assert shared is key
    with pytest.raises(ValueError):
        reversible_scan.make_order_key(key, "bogus_mode")


# --------------------------------------------------------------------------- (2) DB certificate PASSES
def test_db_certificate_passes_production_shape():
    """MANDATORY GATE: the ½(P_AB+P_BA) DB certificate PASSES on the production-shape (4-DTM-superblock
    + clamped b_t) enumerable shadow with max_asym < 1e-10."""
    res = selfadjoint_cert.certify(np.random.default_rng(0), sizes=(1, 1, 1, 1), verbose=False)

    assert res["n_superblocks"] == 4, "must mirror the 4 DTM training-negative superblocks"
    assert res["superblock_names"] == ["upper_hidden", "lower_hidden", "image_output", "label_output"]
    assert res["fwd_order"] == [0, 1, 2, 3] and res["rev_order"] == [3, 2, 1, 0]
    assert res["passed"] is True
    assert res["max_asym"] < selfadjoint_cert.TOL_SYM, (
        f"DB residual max_asym={res['max_asym']:.2e} NOT < {selfadjoint_cert.TOL_SYM:.0e} "
        "— reversible kernel REJECTED, no GPU plan")
    assert res["K_inv_residual"] < selfadjoint_cert.TOL_INV
    assert res["adjoint_dev"] < selfadjoint_cert.TOL_ADJ
    # report the residual unambiguously in the test log
    print(f"\nDB-CERT (production-shape): max_asym(1/2(P_AB+P_BA), pi) = {res['max_asym']:.3e} "
          f"< {selfadjoint_cert.TOL_SYM:.0e}  PASS")


@pytest.mark.parametrize("sizes", [(1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 1, 1)])
def test_db_certificate_robustness(sizes):
    """The cert passes across several superblock-size shapes (still the 4 DTM superblocks)."""
    res = selfadjoint_cert.certify(np.random.default_rng(7), sizes=sizes, verbose=False)
    assert res["passed"] is True
    assert res["max_asym"] < selfadjoint_cert.TOL_SYM


# --------------------------------------------------------------------------- (3) discriminator sanity
def test_deterministic_scan_fails_db_cert():
    """SANITY: the unpatched DETERMINISTIC scan (P_fwd) is genuinely non-reversible (~1e-2), proving
    the certificate discriminates reversible from non-reversible (it would FAIL on the pristine kernel)."""
    res = selfadjoint_cert.certify(np.random.default_rng(0), sizes=(1, 1, 1, 1), verbose=False)
    # the deterministic scan's DB residual must be well above noise AND above the reversible K residual
    assert res["P_fwd_db_residual"] > selfadjoint_cert.MIN_NONREV, (
        f"deterministic P_fwd residual {res['P_fwd_db_residual']:.2e} not > "
        f"{selfadjoint_cert.MIN_NONREV:.0e}; cert has no discriminating teeth")
    assert res["P_fwd_db_residual"] > 1e3 * res["max_asym"], (
        "deterministic scan should be many orders less reversible than the symmetrized kernel")
    print(f"\nDISCRIMINATOR: deterministic P_fwd max_asym = {res['P_fwd_db_residual']:.3e} "
          f"(non-reversible, ~1e-2) vs reversible K = {res['max_asym']:.3e}")
