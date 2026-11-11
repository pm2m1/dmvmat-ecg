import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def source_hash(root):
    root = Path(root)
    return digest(
        {
            str(p.relative_to(root)): sha(p)
            for base in ("src", "configs")
            for p in sorted((root / base).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        }
    )
