"""Regression for the observed duplicate-launch race; invented temp outputs."""

import multiprocessing
import stat

import pytest

from dmvmat_ecg.phase4b.stage_lock import stage_lock


def contender(root, out):
    try:
        with stage_lock(root, "baselines"):
            out.put("ENTERED")
    except RuntimeError:
        out.put("REFUSED")


def test_duplicate_process_refused_before_body_and_lock_released(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    with stage_lock(tmp_path, "baselines"):
        other = ctx.Process(target=contender, args=(str(tmp_path), queue))
        other.start()
        assert queue.get(timeout=20) == "REFUSED"
        other.join(timeout=20)
        assert other.exitcode == 0
    with stage_lock(tmp_path, "baselines"):
        pass
    assert stat.S_IMODE((tmp_path / "stage_locks/baselines.lock").stat().st_mode) == 0o600
    queue.close()


def test_exception_releases_lock_and_independent_stages_can_coexist(tmp_path):
    with pytest.raises(ValueError, match="invented failure"):
        with stage_lock(tmp_path, "source_grid"), stage_lock(tmp_path, "baselines"):
            raise ValueError("invented failure")
    with stage_lock(tmp_path, "source_grid"):
        pass


def test_invalid_stage_and_symlink_refused(tmp_path):
    with pytest.raises(ValueError):
        with stage_lock(tmp_path, "../other"):
            pass
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "stage_locks").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError):
        with stage_lock(tmp_path, "analysis"):
            pass
