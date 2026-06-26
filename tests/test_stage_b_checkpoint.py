# tests/test_stage_b_checkpoint.py — Stage-B checkpoint serialization helpers (exp2 RESUME_FROM).
# CPU-ONLY: the pure save/load/verify helpers in src/htdml/driver.py round-trip plain arrays/dicts/pytrees
# (the encoder Flax params + the out-of-band ArmState) with NO real DTM. The GPU-binding persist_stage_b/
# load_stage_b (RealOps) are exercised only by the MODE=smoke run, mirroring RealOps.fork.
import os

import htdml  # noqa: F401  (conftest also bootstraps paths)
import jax.numpy as jnp
import numpy as np
import pytest

from htdml import driver as DRV


def _fake_ae(vals):
    return {"params": {"Encoder_0": {"w": jnp.asarray(vals)}}}


def _manifest(seed=1, raw_sha="abc123", git="deadbeef", mode="full", n_train=6000):
    return {"schema": DRV._CKPT_SCHEMA, "seed": seed, "raw_split_sha256": raw_sha,
            "code_git_sha": git, "mode": mode, "config": {"stage_b_epochs": 200, "n_train": n_train}}


def test_save_load_arm_state_roundtrip(tmp_path):
    """The out-of-band ArmState (key + per-step autocorrelations + opt_counts) + the encoder Flax params
    + the manifest round-trip through disk byte-faithfully."""
    arm = DRV.ArmState(key=np.array([7, 11], np.uint32),
                       autocorrelations=[{0: 0.5}, {2: 0.25}],
                       opt_counts=[[400], [400]])
    DRV.save_arm_state_to_disk(arm, _fake_ae([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                               _manifest(), str(tmp_path))
    loaded, ae, manifest = DRV.load_arm_state_from_disk(str(tmp_path), _fake_ae(np.zeros((3, 2))))
    assert np.array_equal(np.asarray(loaded.key), np.array([7, 11], np.uint32))
    assert loaded.autocorrelations == [{0: 0.5}, {2: 0.25}]
    assert loaded.opt_counts == [[400], [400]]
    assert np.allclose(np.asarray(ae["params"]["Encoder_0"]["w"]),
                       np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert manifest["seed"] == 1 and manifest["schema"] == DRV._CKPT_SCHEMA


def test_arm_state_autocorrelations_coercion(tmp_path):
    """autocorrelations is a ragged List[dict] of numpy-scalar values with int epoch keys — the field
    np.save can't take. JSON round-trip must restore python float values + INT epoch keys."""
    arm = DRV.ArmState(key=np.array([1, 2], np.uint32),
                       autocorrelations=[{0: np.float64(0.3), 5: np.float32(0.1)}],
                       opt_counts=[[1]])
    DRV.save_arm_state_to_disk(arm, _fake_ae([[0.0]]), _manifest(), str(tmp_path))
    loaded, _ae, _m = DRV.load_arm_state_from_disk(str(tmp_path), _fake_ae(np.zeros((1, 1))))
    d = loaded.autocorrelations[0]
    assert set(d.keys()) == {0, 5} and all(isinstance(k, int) for k in d)   # int keys, not "0"/"5"
    assert isinstance(d[0], float) and abs(d[0] - 0.3) < 1e-6


def test_save_writes_all_expected_files(tmp_path):
    arm = DRV.ArmState(key=np.array([0, 0], np.uint32), autocorrelations=[{}], opt_counts=[[0]])
    DRV.save_arm_state_to_disk(arm, _fake_ae([[1.0]]), _manifest(), str(tmp_path))
    for name in ("dtm_key.npy", "autocorrelations.json", "opt_counts.json",
                 "ae_params.msgpack", "manifest.json"):
        assert os.path.exists(tmp_path / name), f"missing {name}"


def test_require_checkpoint_fail_closed(tmp_path):
    """RESUME_FROM with an incomplete checkpoint must FAIL CLOSED (no silent retrain)."""
    with pytest.raises(FileNotFoundError):
        DRV.require_checkpoint(str(tmp_path))                       # empty dir
    # only the arm-state files present, but the DTM epoch dir missing → still fail closed
    arm = DRV.ArmState(key=np.array([0, 0], np.uint32), autocorrelations=[{}], opt_counts=[[0]])
    DRV.save_arm_state_to_disk(arm, _fake_ae([[1.0]]), _manifest(), str(tmp_path))
    with pytest.raises(FileNotFoundError):
        DRV.require_checkpoint(str(tmp_path))                       # no dtm/model_saving/epoch_000
    os.makedirs(tmp_path / "dtm" / "model_saving" / "epoch_000")
    DRV.require_checkpoint(str(tmp_path))                           # now complete → no raise


def test_verify_manifest_passes_on_match():
    DRV.verify_manifest(_manifest(seed=1, raw_sha="abc"), expect_seed=1, expect_raw_sha="abc")


def test_verify_manifest_raises_on_seed_mismatch():
    with pytest.raises(RuntimeError):
        DRV.verify_manifest(_manifest(seed=1, raw_sha="abc"), expect_seed=2, expect_raw_sha="abc")


def test_verify_manifest_raises_on_raw_sha_mismatch():
    with pytest.raises(RuntimeError):
        DRV.verify_manifest(_manifest(seed=1, raw_sha="abc"), expect_seed=1, expect_raw_sha="DIFFERENT")


def test_verify_manifest_allows_git_sha_mismatch():
    """The cal-gate fix legitimately changes code, so a git-SHA difference must NOT block resume —
    verify_manifest checks only seed + raw-split sha (+ optional mode/n_train), never the code sha."""
    m = _manifest(seed=1, raw_sha="abc", git="OLD_pre_fix_sha")
    DRV.verify_manifest(m, expect_seed=1, expect_raw_sha="abc")     # no raise despite a stale git sha


def test_verify_manifest_raises_on_mode_mismatch():
    """Hardening: resuming a smoke checkpoint under MODE=full (or vice versa) re-encodes a different
    n_train → silently-wrong latents. The optional mode guard fails it closed."""
    with pytest.raises(RuntimeError):
        DRV.verify_manifest(_manifest(mode="smoke"), expect_seed=1, expect_raw_sha="abc123",
                            expect_mode="full")


def test_verify_manifest_raises_on_n_train_mismatch():
    with pytest.raises(RuntimeError):
        DRV.verify_manifest(_manifest(n_train=600), expect_seed=1, expect_raw_sha="abc123",
                            expect_n_train=6000)


def test_verify_manifest_optional_checks_skipped_when_none():
    """Backward-compat: omitting expect_mode/expect_n_train (the pre-hardening call sites) does NOT
    check them — only seed + raw-sha are mandatory."""
    DRV.verify_manifest(_manifest(mode="smoke", n_train=600), expect_seed=1, expect_raw_sha="abc123")
