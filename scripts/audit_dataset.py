"""Verify the publisher files, patient isolation and waveform integrity; no models."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import wfdb

from dmvmat_ecg.evaluation.state import Study
from dmvmat_ecg.labels.superclasses import CLASSES, encode, label_map
from dmvmat_ecg.preprocessing.signals import canonical_leads
from dmvmat_ecg.utils.io import digest, sha, write_json


def metadata():
    raw = ROOT / "data/raw/ptbxl_v1_0_3"
    sums = dict(
        (line.split(maxsplit=1)[1], line.split(maxsplit=1)[0])
        for line in (raw / "SHA256SUMS.txt").read_text().splitlines()
    )
    for name in (
        "ptbxl_database.csv",
        "scp_statements.csv",
        "LICENSE.txt",
        "ptbxl_v103_changelog.txt",
    ):
        assert sha(raw / name) == sums[name], name
    df = pd.read_csv(raw / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    mapping = label_map(statements)
    y = np.stack([encode(c, mapping) for c in df.scp_codes])
    assert (
        df.index.is_unique
        and df.patient_id.notna().all()
        and df.strat_fold.isin(range(1, 11)).all()
    )
    assert np.equal(df.patient_id, df.patient_id.astype(int)).all()
    # Check isolation before excluding empty task annotations as well.
    roles = np.where(
        df.strat_fold <= 8, "train", np.where(df.strat_fold == 9, "validation", "test")
    )
    patients = {
        r: set(df.loc[roles == r, "patient_id"].astype(int))
        for r in ("train", "validation", "test")
    }
    for a in patients:
        for b in patients:
            if a != b:
                assert not patients[a] & patients[b], f"Patient crosses {a}/{b}"
    eligible = y.sum(1) > 0
    summary = dict(
        release="1.0.3",
        raw_records=len(df),
        raw_patients=int(df.patient_id.nunique()),
        eligible_records=int(eligible.sum()),
        eligible_patients=int(df.loc[eligible].patient_id.nunique()),
        excluded_without_diagnostic_superclass=int((~eligible).sum()),
        multilabel_records=int((y.sum(1) > 1).sum()),
        class_order=CLASSES,
        likelihood_policy="all diagnostic keys, including likelihood0; no numeric likelihood threshold",
        publisher_manifest_sha256=sha(raw / "SHA256SUMS.txt"),
        metadata_sha256=sha(raw / "ptbxl_database.csv"),
        statements_sha256=sha(raw / "scp_statements.csv"),
        mapping_hash=digest(mapping),
        patient_cross_partition_overlap=0,
        source_url="https://physionet.org/content/ptb-xl/1.0.3/",
        license="CC-BY-4.0",
        partitions={},
    )
    for r in patients:
        ix = eligible & (roles == r)
        summary["partitions"][r] = dict(
            records=int(ix.sum()),
            patients=int(df.loc[ix].patient_id.nunique()),
            positives=dict(zip(CLASSES, y[ix].sum(0).astype(int).tolist(), strict=True)),
            prevalence=dict(zip(CLASSES, y[ix].mean(0).tolist(), strict=True)),
            excluded_empty=int(((~eligible) & (roles == r)).sum()),
        )
    return raw, sums, df, y, eligible, roles, summary


def main(metadata_only):
    raw, sums, df, y, eligible, roles, summary = metadata()
    if metadata_only:
        write_json(ROOT / "results/dataset_metadata_audit.json", summary)
        print(json.dumps(summary, sort_keys=True))
        return
    assert (ROOT / "private/download-receipt.json").exists(), "Complete verified download required"
    study = Study(ROOT)
    assert study.get("phase") == "data" and not study.runs(), (
        "Data preparation may not alter an existing study"
    )
    cache = ROOT / "private/cache"
    cache.mkdir(parents=True, exist_ok=True)
    arrays = {}
    memberships = {}
    mappings = {}
    for role in ("train", "validation", "test"):
        ix = np.flatnonzero(eligible & (roles == role))
        memberships[role] = df.index[ix].astype(int).tolist()
        arrays[role] = np.lib.format.open_memmap(
            cache / f"{role}_x.npy", mode="w+", dtype=np.float32, shape=(len(ix), 12, 1000)
        )
        mappings[role] = {int(row): k for k, row in enumerate(ix)}
        np.savez(
            cache / f"{role}_meta.npz",
            ecg_id=df.index[ix].to_numpy(dtype=np.int64),
            patient_id=df.iloc[ix].patient_id.to_numpy(dtype=np.int64),
            y=y[ix],
            fold=df.iloc[ix].strat_fold.to_numpy(dtype=np.int64),
        )
    seen_dat = {}
    seen_physical = {}
    cross_duplicates = []
    within_duplicates = 0
    constant_leads = 0
    verified = 0
    total = 0.0
    squares = 0.0
    observations = 0
    start = time.perf_counter()
    for k, (ecg_id, row) in enumerate(df.iterrows()):
        base = str(row.filename_lr)
        for suffix in (".dat", ".hea"):
            filename = base + suffix
            assert (raw / filename).is_file() and sha(raw / filename) == sums[filename], (
                f"Waveform missing/changed: {filename}"
            )
            verified += 1
        header = wfdb.rdheader(str(raw / base))
        assert header.fs == 100 and header.sig_len == 1000 and header.n_sig == 12
        assert all(u.lower() == "mv" for u in header.units), "Unexpected amplitude units"
        signal, fields = wfdb.rdsamp(str(raw / base))
        order = canonical_leads(fields["sig_name"])
        assert signal.shape == (1000, 12) and np.isfinite(signal).all()
        x = np.ascontiguousarray(signal[:, order].T, dtype=np.float32)
        constant_leads += int((np.ptp(x, axis=1) == 0).sum())
        for key, store in (
            (sums[base + ".dat"], seen_dat),
            (hashlib.sha256(x.tobytes()).hexdigest(), seen_physical),
        ):
            if key in store:
                if store[key] != roles[k]:
                    cross_duplicates.append([store[key], str(roles[k])])
                else:
                    within_duplicates += 1
            store[key] = str(roles[k])
        if eligible[k]:
            role = str(roles[k])
            arrays[role][mappings[role][k]] = x
            if role == "train":
                # Fit a single global scalar as upstream, on training only.
                total += float(signal.sum(dtype=np.float64))
                squares += float(np.square(signal).sum(dtype=np.float64))
                observations += signal.size
        if (k + 1) % 2000 == 0:
            print(
                json.dumps(
                    dict(
                        audited_records=k + 1,
                        required_records=len(df),
                        seconds=time.perf_counter() - start,
                    )
                ),
                flush=True,
            )
    assert not cross_duplicates, "Cross-partition exact duplicate waveform: stop before training"
    for a in arrays.values():
        a.flush()
    mu = total / observations
    sd = float(np.sqrt(squares / observations - mu * mu))
    assert np.isfinite(sd) and sd > 0
    stats = dict(
        mean=mu,
        std=sd,
        fit_role="train",
        fit_rows=len(memberships["train"]),
        fit_membership_hash=digest(memberships["train"]),
        method="global scalar population mean/std over all training samples and leads",
    )
    write_json(cache / "standardization.json", stats)
    summary.update(
        waveform_shape=[12, 1000],
        sampling_frequency=100,
        units="mV",
        verified_waveform_files=verified,
        cross_partition_raw_or_physical_duplicates=0,
        within_partition_duplicate_detections=within_duplicates,
        constant_lead_observations=constant_leads,
        canonical_lead_order=[
            "I",
            "II",
            "III",
            "AVR",
            "AVL",
            "AVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        waveform_cache_dtype="float32",
        waveform_cache_bytes=sum(a.nbytes for a in arrays.values()),
        cache_hashes={p.name: sha(p) for p in cache.iterdir() if p.is_file()},
        standardized_fit_membership_hash=stats["fit_membership_hash"],
        seconds=time.perf_counter() - start,
    )
    write_json(ROOT / "results/dataset_audit.json", summary)
    write_json(ROOT / "data/manifests/memberships.json", memberships)
    study.set("memberships", memberships)
    study.set("dataset", summary)
    study.set("phase", "development")
    print(
        json.dumps(
            dict(
                status="DATA_AUDIT_COMPLETE",
                records=len(df),
                eligible=int(eligible.sum()),
                patient_overlap=0,
                waveform_overlap=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-only", action="store_true")
    main(parser.parse_args().metadata_only)
