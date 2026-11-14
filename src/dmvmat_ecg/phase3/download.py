"""Bounded official five-source download, separate from all experiment ledgers."""

import hashlib
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

BASE = "https://physionet.org/files/challenge-2021/1.0.3/"
SOURCES = ("chapman_shaoxing", "ningbo", "cpsc_2018", "cpsc_2018_extra", "georgia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected_entries(text):
    result = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        checksum, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = PurePosixPath(name)
        if not re.fullmatch(r"[0-9a-f]{64}", checksum) or path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe official-index entry")
        if len(path.parts) >= 3 and path.parts[0] == "training" and path.parts[1] in SOURCES:
            if name in result:
                raise ValueError("Duplicate checksum entry")
            result[name] = checksum
    if {PurePosixPath(n).parts[1] for n in result} != set(SOURCES):
        raise ValueError("All five requested sources must be present")
    return result


def prepare(root):
    root = Path(root).resolve()
    entries = selected_entries((root / "SHA256SUMS.txt").read_text())
    ready, missing, removed = Counter(), [], []
    for name, checksum in entries.items():
        p = root / name
        if not p.resolve().is_relative_to(root):
            raise ValueError("Local download path escapes data root")
        if p.is_file() and sha(p) == checksum:
            ready[PurePosixPath(name).parts[1]] += 1
            continue
        if p.exists():
            p.unlink()  # Explicitly authorized bad/partial copy removal, inside target roots only.
            removed.append(name)
        missing.append(name)
    stray = []
    for p in root.rglob("index.html*"):
        if p.is_file() and p.resolve().is_relative_to(root):
            stray.append(str(p.relative_to(root)))
            p.unlink()
    checksum_path = root / "TARGET_SHA256SUMS.txt"
    checksum_path.write_text("".join(f"{h}  {n}\n" for n, h in entries.items()))
    lines = []
    for name in missing:
        p = root / name
        lines.extend(
            [
                BASE + name,
                " dir=" + str(p.parent),
                " out=" + p.name,
                " checksum=sha-256=" + entries[name],
            ]
        )
    (root / "aria2_input.txt").write_text("\n".join(lines) + "\n")
    record = {
        "release": "1.0.3",
        "base": BASE,
        "allowed_sources": SOURCES,
        "publisher_index_sha256": sha(root / "SHA256SUMS.txt"),
        "expected": dict(Counter(PurePosixPath(n).parts[1] for n in entries)),
        "already_verified": dict(ready),
        "missing_to_download": len(missing),
        "bad_copies_removed": removed,
        "stray_wget_indices_removed": stray,
        "maximum_connections": 8,
    }
    (root / "DOWNLOAD_PREPARATION.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def download(root):
    root = Path(root).resolve()
    started = time.time()
    status = root / "DOWNLOAD_STATUS.json"
    status.write_text(json.dumps(dict(status="running", started=started, maximum_connections=8)))
    command = [
        "caffeinate",
        "-i",
        "aria2c",
        "-i",
        str(
            root
            / (
                "aria2_remaining.txt"
                if (root / "aria2_remaining.txt").is_file()
                else "aria2_input.txt"
            )
        ),
        "-j",
        "8",
        "-x",
        "1",
        "-s",
        "1",
        "-c",
        "--max-tries=10",
        "--retry-wait=5",
        "--auto-file-renaming=false",
        "--check-integrity=true",
        "--file-allocation=none",
        "--connect-timeout=30",
        "--timeout=60",
        "--enable-http-pipelining=true",
        "--summary-interval=30",
        "--console-log-level=warn",
        "--download-result=hide",
        "--enable-color=false",
        "--save-session=" + str(root / "aria2_remaining.txt"),
        "--save-session-interval=30",
        "--log=" + str(root / "aria2_errors.log"),
        "--log-level=warn",
    ]
    result = subprocess.run(command, check=False)
    status.write_text(
        json.dumps(
            dict(
                status="download-process-completed",
                returncode=result.returncode,
                started=started,
                elapsed_seconds=time.time() - started,
                verification_required=True,
            ),
            indent=2,
        )
    )
    return result.returncode


def verify(root):
    root = Path(root).resolve()
    entries = selected_entries((root / "SHA256SUMS.txt").read_text())
    sources = defaultdict(lambda: dict(expected_files=0, downloaded_files=0, bytes=0, mismatches=0))
    failures = []
    for name, checksum in entries.items():
        summary = sources[PurePosixPath(name).parts[1]]
        summary["expected_files"] += 1
        p = root / name
        if not p.is_file() or sha(p) != checksum:
            failures.append(name)
            summary["mismatches"] += 1
        else:
            summary["downloaded_files"] += 1
            summary["bytes"] += p.stat().st_size
    # The requested independent shasum invocation checks the exact filtered list.
    with (root / "SHASUM_VERIFICATION.log").open("w") as stream:
        run = subprocess.run(
            ["shasum", "-a", "256", "-c", "TARGET_SHA256SUMS.txt"],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report = dict(
        release="1.0.3",
        sources=dict(sources),
        failures=failures,
        shasum_returncode=run.returncode,
        complete=not failures and run.returncode == 0,
        excluded_sources=["ptb", "ptb-xl", "st_petersburg_incart"],
    )
    (root / "DOWNLOAD_VERIFICATION.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
