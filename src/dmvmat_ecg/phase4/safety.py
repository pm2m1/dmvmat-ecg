"""Explicit metadata projections and record guards; no evaluation/seal imports."""

import hashlib
import json
import mmap
import os
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def safe_path(path, root):
    p, root = Path(path).absolute(), Path(root).resolve()
    if not p.is_relative_to(root) or any(x.is_symlink() for x in (p, *p.parents)):
        raise PermissionError("Outside explicit input root or symlink")
    if any(
        "sealed" in x.lower() or "reserved" in x.lower() or "evaluation" in x.lower()
        for x in p.relative_to(root).parts
    ):
        raise PermissionError("Forbidden sealed/reserved/evaluation path")
    return p


def project_json(path, selected):
    """Decode only allowlisted JSON fields; structurally skip all other values.

    Unlike json.load, this never materializes unselected subtrees. Memory mapping
    is used only as a byte scanner, not as a model/data loader. Duplicate keys
    fail closed. selected is an iterable of exact tuple paths.
    """
    selected = {tuple(p) for p in selected}
    result = {}
    with Path(path).open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
        n = len(b)

        def whitespace(i):
            while i < n and b[i] in b" \r\n\t":
                i += 1
            return i

        def skip(i):
            i = whitespace(i)
            if b[i] == 34:
                i += 1
                while i < n:
                    if b[i] == 92:
                        i += 2
                    elif b[i] == 34:
                        return i + 1
                    else:
                        i += 1
            elif b[i] in (123, 91):
                stack = [b[i]]
                i += 1
                while i < n and stack:
                    if b[i] == 34:
                        i = skip(i)
                        continue
                    if b[i] in (123, 91):
                        stack.append(b[i])
                    elif b[i] in (125, 93):
                        if stack.pop() != {125: 123, 93: 91}[b[i]]:
                            raise ValueError("Malformed skipped JSON structure")
                    i += 1
                if not stack:
                    return i
            else:
                while i < n and b[i] not in b",]} \r\n\t":
                    i += 1
                return i
            raise ValueError("Unterminated JSON")

        def walk(i, prefix):
            i = whitespace(i)
            if prefix in selected:
                end = skip(i)
                result[prefix] = json.loads(b[i:end])
                return end
            if not any(p[: len(prefix)] == prefix for p in selected):
                return skip(i)
            if b[i] != 123:
                raise ValueError("Selected field parent must be an object")
            i = whitespace(i + 1)
            seen = set()
            while b[i] != 125:
                end = skip(i)
                key = json.loads(b[i:end])
                if not isinstance(key, str) or key in seen:
                    raise ValueError("Invalid/duplicate JSON field")
                seen.add(key)
                i = whitespace(end)
                if b[i] != 58:
                    raise ValueError("Expected colon")
                i = whitespace(walk(i + 1, (*prefix, key)))
                if b[i] == 44:
                    i = whitespace(i + 1)
                elif b[i] != 125:
                    raise ValueError("Expected object separator")
            return i + 1

        end = whitespace(walk(0, ()))
        if end != n or set(result) != selected:
            raise ValueError("Missing allowed fields or trailing JSON")
    return result


def exclusion_hash(record):
    return digest(["phase4-test-exclusion-only", record])


class RecordGuard:
    def __init__(self, pool, exclusion_hashes):
        self.pool = set(pool)
        self.exclusions = set(exclusion_hashes)
        if len(self.pool) != len(pool):
            raise PermissionError("Duplicate pool ID")
        self.check(pool)

    def check(self, records):
        if any(r not in self.pool or exclusion_hash(r) in self.exclusions for r in records):
            raise PermissionError("Test/excluded/non-adaptation record access rejected")


def bundle_only_audit(project, bundle, output):
    """Deny project data/old private paths at OS-open time after construction."""
    project, bundle, output = map(lambda p: Path(p).resolve(), (project, bundle, output))

    def audit(event, args):
        if event not in {"open", "os.listdir", "os.scandir"} or not args:
            return
        if not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).absolute()
        if p.is_relative_to(bundle) or p.is_relative_to(output):
            return
        if p.is_relative_to(project / "private") or p.is_relative_to(project / "data"):
            raise PermissionError("Phase4 may read only its isolated bundle")
        if p.is_relative_to(project / "results") or p.is_relative_to(project / "report_ecg"):
            raise PermissionError("Old result/report/evaluation access forbidden")
        if p.is_relative_to(project) and any(
            "sealed" in part.lower() or "reserved" in part.lower() for part in p.parts
        ):
            raise PermissionError("Sealed/reserved access forbidden")

    sys.addaudithook(audit)
