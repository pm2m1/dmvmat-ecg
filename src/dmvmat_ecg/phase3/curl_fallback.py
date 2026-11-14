"""Permitted parallel curl fallback; eight reusable single-stream clients.

Each client owns disjoint URLs. HTTP2/keepalive reuse avoids a fresh process and
TLS handshake for each tiny header. No more than eight active connections.
"""

import json
import subprocess
import time
from pathlib import Path

from .download import BASE, SOURCES, selected_entries, sha


def configs(root, names, *, prefix):
    root = Path(root).resolve()
    paths = []
    workers = min(8, len(names))
    for worker in range(workers):
        # Contiguous balanced blocks mix headers and waveforms. Round-robin
        # individual files would put only MATs on half the clients.
        group = names[len(names) * worker // workers : len(names) * (worker + 1) // workers]
        path = root / f"{prefix}_worker{worker}.cfg"
        lines = [
            "fail",
            "silent",
            "show-error",
            "ipv4",
            "http2",
            "create-dirs",
            "connect-timeout = 30",
            "max-time = 600",
            "speed-limit = 1",
            "speed-time = 60",
            "retry = 9",
            "retry-all-errors",
            "retry-delay = 5",
            "continue-at = -",
            'write-out = "%{url_effective}\\t%{http_version}\\t%{size_download}\\t%{time_total}\\n"',
        ]
        for name in group:
            local = root / name
            parts = Path(name).parts
            if (
                not local.resolve().is_relative_to(root)
                or len(parts) < 3
                or parts[0] != "training"
                or parts[1] not in SOURCES
            ):
                raise ValueError("Unsafe fallback path")
            lines.extend(["url = " + json.dumps(BASE + name), "output = " + json.dumps(str(local))])
        with path.open("x") as stream:
            stream.write("\n".join(lines) + "\n")
        paths.append(path)
    return paths


def run(root, *, probe=False):
    root = Path(root).resolve()
    started = time.time()
    if not probe:
        (root / "DOWNLOAD_STATUS.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "started": started,
                    "maximum_connections": 8,
                    "client": "curl8 reusable single-stream workers",
                }
            )
        )
    expected = selected_entries((root / "SHA256SUMS.txt").read_text())
    pending = []
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha(path) != checksum:
            if path.exists():
                path.unlink()  # User explicitly authorized bad/partial copy deletion.
            pending.append(name)
    if probe:
        pending = pending[:16]
    prefix = "curl_probe" if probe else "curl_download"
    paths = configs(root, pending, prefix=prefix)
    result = (
        subprocess.run(
            ["caffeinate", "-i", "xargs", "-0", "-P", "8", "-I", "{}", "curl", "--config", "{}"],
            input=b"".join(str(path).encode() + b"\0" for path in paths),
            check=False,
        )
        if paths
        else subprocess.CompletedProcess([], 0)
    )
    record = {
        "probe": probe,
        "files": len(pending),
        "elapsed_seconds": time.time() - started,
        "returncode": result.returncode,
        "maximum_connections": 8,
        "total_tries_per_transfer_max": 10,
        "retry_wait_seconds": 5,
    }
    if probe:
        record["sha_verified_files"] = sum(
            (root / name).is_file() and sha(root / name) == expected[name] for name in pending
        )
    else:
        (root / "DOWNLOAD_STATUS.json").write_text(
            json.dumps(
                {"status": "download-process-completed", **record, "verification_required": True}
            )
        )
    with (root / (prefix + "_RESULT.json")).open("x") as stream:
        json.dump(record, stream, indent=2)
    return record
