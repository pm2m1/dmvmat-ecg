"""Trusted full-inventory verification and the one real Phase2D ledger location.

The inventory rule is shared with phase2d_prepare.py. Neither the project root
nor the real ledger location is selected by a caller's inventory or role string.
These application guards are not an OS sandbox against arbitrary file edits.
"""

from pathlib import Path

from dmvmat_ecg.phase2b.runtime import code_identity, environment_receipt, execution_inventory

from .contract import digest

REAL_PROJECT_ROOT = (Path.home() / "Desktop/dmvmat-ecg").resolve()
REAL_LEDGER_PATH = REAL_PROJECT_ROOT / "private/phase2d_joint44301_v1/study.sqlite"


def real_ledger_path(folder):
    """Require the direct canonical path, rejecting aliases and missing databases."""
    requested = Path(folder).expanduser().absolute() / "study.sqlite"
    if (
        requested != REAL_LEDGER_PATH
        or requested.resolve() != REAL_LEDGER_PATH
        or requested.is_symlink()
        or not requested.is_file()
    ):
        raise PermissionError("Only the existing canonical real ledger may be opened; no aliases")
    return requested


def verify_execution_identity(identity, supplied_inventory=None):
    """Recompute all trusted pins, never hash a caller-selected subset."""
    try:
        expected = execution_inventory(REAL_PROJECT_ROOT)
        if supplied_inventory is not None and dict(supplied_inventory) != expected:
            raise PermissionError("Full expected inventory required; missing/extra/changed pins")
        if code_identity(expected) != identity["code_hash"]:
            raise PermissionError("Live full-inventory code identity differs from ledger")
        if digest(environment_receipt(REAL_PROJECT_ROOT)) != identity["dependency_hash"]:
            raise PermissionError("Live dependency identity differs from ledger")
    except PermissionError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise PermissionError("Cannot verify full code/dependency identity") from error
    return expected
