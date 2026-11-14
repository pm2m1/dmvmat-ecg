"""Development-only authority; separate from the unchanged fictional-data guard."""

from pathlib import Path

from dmvmat_ecg.phase3b.jobs import file_hash, source_code_hash
from dmvmat_ecg.phase3b.protocol import digest

NAMESPACE = "real:phase3b-cpsc-v1"
BASE_HASH = "d4f0dd6361cf115d79c8384905682d76a46ee2f9fb79446680dfadafc7cc7ebb"
MAPPING_HASH = "ecce890b4678b8f2a06cf0796476221adceb8156c7d476ecfbeba910a43891d6"
PARENT_HASH = "84bea540fef6e04d5638f7b09f1107de5b0335a0e58df1d2b93007e8bed16cf1"


def real_only(namespace):
    if namespace != NAMESPACE or source_code_hash() != BASE_HASH:
        raise PermissionError("Real namespace or frozen Phase3B base identity differs")


def implementation_hash():
    folder = Path(__file__).resolve().parent
    return digest({p.name: file_hash(p) for p in sorted(folder.glob("*.py"))})


def permit_record(record, role, memberships):
    """Call BEFORE a header, annotation, waveform, cache or file checksum read."""
    if role == "O" or record in set(memberships["O"]):
        raise PermissionError("O waveform/label/representation/prediction access is NOT authorized")
    if record in set(memberships["FEAS"]):
        raise PermissionError("CPSC FEAS access is forbidden in Phase3B")
    allowed = memberships.get(role)
    if allowed is None or record not in set(allowed):
        raise PermissionError("Record not in authorized development role")
