"""Resume/checksum the official public S3 mirror: metadata and records100 only."""

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path, PurePosixPath

import httpx

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/ptbxl_v1_0_3"
BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
CLIENT = httpx.Client(
    limits=httpx.Limits(max_connections=16, max_keepalive_connections=16, keepalive_expiry=60),
    timeout=90,
    follow_redirects=True,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(name, expected):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or "\\" in name:
        raise ValueError("Unsafe official-manifest path")
    target = RAW / name
    if target.exists() and sha(target) == expected:
        return "verified_existing"
    if target.exists():
        raise ValueError("Existing final file checksum mismatch: " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    if part.exists() and sha(part) == expected:
        part.rename(target)
        return "verified_existing"
    errors = []
    for retry in range(4):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with CLIENT.stream(
                "GET", BASE + name, headers={"Range": f"bytes={offset}-"} if offset else {}
            ) as response:
                response.raise_for_status()
                append = offset and response.status_code == 206
                with part.open("ab" if append else "wb") as f:
                    for block in response.iter_bytes():
                        f.write(block)
            if sha(part) != expected:
                raise ValueError("Checksum mismatch: " + name)
            part.rename(target)
            return "downloaded_verified"
        except Exception as error:
            errors.append(type(error).__name__ + ": " + str(error))
            if retry < 3:
                time.sleep(2**retry)
    raise RuntimeError(json.dumps(dict(file=name, failures=errors)))


def main():
    manifest = RAW / "SHA256SUMS.txt"
    assert manifest.exists(), "Fetch official HTTPS SHA256SUMS.txt first"
    entries = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        entries[name.lstrip("*")] = digest
    names = [n for n in entries if n.startswith("records100/") and n.endswith((".dat", ".hea"))]
    metadata = [n for n in entries if "/" not in n]
    assert names and not any("records500" in n for n in names)
    for n in metadata:
        one(n, entries[n])
    print(
        json.dumps(
            dict(
                stage="metadata_verified",
                waveform_files=len(names),
                checksum_manifest=sha(manifest),
            )
        ),
        flush=True,
    )
    start = time.perf_counter()
    counts = dict(verified_existing=0, downloaded_verified=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        pending = {pool.submit(one, n, entries[n]): n for n in names}
        for k, future in enumerate(concurrent.futures.as_completed(pending), 1):
            counts[future.result()] += 1
            if k % 1000 == 0 or k == len(names):
                print(
                    json.dumps(
                        dict(
                            verified_files=k,
                            required_files=len(names),
                            seconds=time.perf_counter() - start,
                            **counts,
                        )
                    ),
                    flush=True,
                )
    receipt = dict(
        release="1.0.3",
        official_mirror=BASE,
        checksum_manifest_sha256=sha(manifest),
        metadata_files=metadata,
        required_waveform_files=len(names),
        verified_waveform_files=len(names),
        downloaded_500_hz=False,
        seconds=time.perf_counter() - start,
        **counts,
    )
    dest = ROOT / "private/download-receipt.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
    print("DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
