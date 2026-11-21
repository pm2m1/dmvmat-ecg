"""OS-owned stage mutex: prevent duplicate controllers before fitting."""

import contextlib
import fcntl
import os
import stat
from pathlib import Path

STAGES = frozenset(("baselines", "source_grid", "aligned_targets", "analysis"))


@contextlib.contextmanager
def stage_lock(run, stage):
    if stage not in STAGES:
        raise ValueError("Unregistered Phase 4B stage")
    folder = Path(run) / "stage_locks"
    folder.mkdir(mode=0o700, exist_ok=True)
    if folder.is_symlink() or folder.stat().st_uid != os.geteuid():
        raise PermissionError("Unsafe stage-lock directory")
    path = folder / (stage + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as handle:
        info = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise PermissionError("Unsafe stage-lock file")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("PHASE4B_STAGE_ALREADY_RUNNING:" + stage) from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
