"""One-way allowlisted import of Experiment 4 ADAPTATION data; no test reader."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase2.contracts import MODULES
from dmvmat_ecg.phase3.signals import header, physical_wave, to_100hz_first10
from dmvmat_ecg.phase3b.models import fresh_model, random_count_cost
from dmvmat_ecg.phase3b.protocol import (
    BUDGETS,
    DRAW_SEEDS,
    OPTIMIZER,
    SOURCE_SEEDS,
    seed,
)
from dmvmat_ecg.phase3b_eval_v2.decoder import decode_labels

from .safety import RecordGuard, digest, exclusion_hash, new_json, project_json, sha
from .safety import safe_path as base_safe_path

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "private/phase3b_cpsc_v1"
BUNDLE = ROOT / "private/phase4_oracle_bundle"
MAPPING_HASH = "ecce890b4678b8f2a06cf0796476221adceb8156c7d476ecfbeba910a43891d6"


def safe_path(path, root):
    """Builder input paths also reject lexical traversal before filesystem reads."""
    if ".." in Path(path).parts:
        raise PermissionError("Input path traversal forbidden")
    result = base_safe_path(path, root)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise PermissionError("Resolved input path escapes its allowlisted root")
    return result


def make_bundle(root=ROOT, output=BUNDLE):
    root, output = Path(root).resolve(), Path(output).absolute()
    old = root / "private/phase3b_cpsc_v1"
    raw = root / "data/raw/cinc2021_v1_0_3"
    if output != root / "private/phase4_oracle_bundle" or output.exists():
        raise FileExistsError("Only a fresh explicit Phase4 bundle destination is permitted")
    accesses = []

    def fields(relative, paths):
        p = safe_path(old / relative, old)
        value = project_json(p, paths)
        # Only the selected-field digest is computed for mixed metadata files.
        accesses.append(
            dict(
                file=relative,
                fields=[list(x) for x in paths],
                projection_sha256=digest({"/".join(k): v for k, v in value.items()}),
            )
        )
        return value

    paths = [
        ("membership", "R"),
        ("membership", "O"),
        ("episodes",),
        ("split_seed",),
        ("mapping_hash",),
    ]
    c = fields("MEMBERSHIPS.json", paths)
    pool, test = c["membership", "R"], c["membership", "O"]
    if len(pool) != 3859 or len(test) != 3860 or len(set(test)) != 3860:
        raise PermissionError("Original role assignment count mismatch")
    guard = RecordGuard(pool, [exclusion_hash(r) for r in test])
    if set(pool) & set(test) or c["split_seed",] != 77101 or c["mapping_hash",] != MAPPING_HASH:
        raise PermissionError("Role assignment or mapping mismatch")
    episodes = c["episodes",]
    if set(episodes) != {f"{d}/B{b}" for d in DRAW_SEEDS for b in BUDGETS}:
        raise PermissionError("Episode inventory mismatch")
    used = set()
    for name, ep in episodes.items():
        budget = int(name.split("/B")[1])
        for role in ("F", "G"):
            guard.check(ep[role])
            if len(ep[role]) != budget // 2 or digest(ep[role]) != ep["hashes"][role]:
                raise PermissionError("Original label-draw identity mismatch")
            used.update(ep[role])
        if set(ep["F"]) & set(ep["G"]):
            raise PermissionError("F/G overlap")
    map_path = safe_path(root / "results/phase3a/LABEL_MAPPING.json", root / "results/phase3a")
    if sha(map_path) != MAPPING_HASH:
        raise PermissionError("Frozen mapping changed")
    mapping = json.loads(map_path.read_text())  # codebook only, no participant data

    sources = {}
    for s in SOURCE_SEEDS:
        selected = [(x,) for x in ("source_seed", "artifact", "sha", "state_hash", "normalization")]
        v = fields(f"source_completions/{s}.json", selected)
        artifact = safe_path(v["artifact",], old / "artifacts")
        if v["source_seed",] != s or sha(artifact) != v["sha",]:
            raise PermissionError("Original source checkpoint mismatch")
        saved = torch.load(artifact, map_location="cpu", weights_only=True)
        model = fresh_model(s)
        model.load_state_dict(saved["state"], strict=True)
        if (
            state_hash(model) != v["state_hash",]
            or saved["state_hash"] != v["state_hash",]
            or saved["normalization"] != v["normalization",]
            or saved["identity"]["seed"] != s
        ):
            raise PermissionError("Source state/normalization identity mismatch")
        sources[str(s)] = dict(
            checkpoint=f"sources/{s}.pt",
            sha256=v["sha",],
            state_hash=v["state_hash",],
            normalization=v["normalization",],
        )
    references, masks, target_entries = {}, {}, {}
    ref_fields = [
        (x,)
        for x in (
            "method",
            "mask",
            "state_hash",
            "drift",
            "final_training_loss",
            "completed_updates",
            "selected_update",
            "normalization",
        )
    ]
    for s in SOURCE_SEEDS:
        for d in DRAW_SEEDS:
            key = f"{s}/{d}"
            references[key], masks[key] = {}, {}
            for method in ("full_cost", "head_cost", "selective_a", "random_a"):
                v = fields(f"target_completions/{s}-{d}-B128/{method}.json", ref_fields)
                record = {k[0]: value for k, value in v.items()}
                if (
                    record["method"] != method
                    or record["completed_updates"] != 200
                    or record["selected_update"] != 200
                    or set(record["mask"]) != set(MODULES)
                    or record["normalization"] != sources[str(s)]["normalization"]
                ):
                    raise PermissionError("Original target completion settings mismatch")
                references[key][method] = record
                masks[key][method] = record["mask"]
            dec = fields(
                f"decisions/{s}-{d}-B128.json",
                [
                    ("mask",),
                    ("configuration",),
                    ("source_checkpoint",),
                    ("normalization",),
                    ("modules",),
                ],
            )
            if dec["mask",] != masks[key]["selective_a"]:
                raise PermissionError("Recorded selector mask disagrees with evidence")
            target_entries[key] = {k[0]: value for k, value in dec.items()}
            model = fresh_model(s)
            expected, _ = random_count_cost(
                model, masks[key]["selective_a"], seed([78601, s, d, 128])
            )
            if masks[key]["random_a"] != expected:
                raise PermissionError("Original random mask seed mismatch")

    # Only an adaptation-only original cache and its used-record annotations.
    cp = fields("CACHE_RECEIPT.json", [("caches", "target_development"), ("target_annotations",)])
    cache = cp["caches", "target_development"]
    guard.check(cache["records"])
    if set(cache["records"]) != used or set(cp["target_annotations",]) != used:
        raise PermissionError("Development-only cache membership mismatch")
    cache_path = safe_path(cache["path"], old / "cache")
    if sha(cache_path) != cache["sha"]:
        raise PermissionError("Development waveform cache mismatch")
    prior = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    index = {rid: i for i, rid in enumerate(cache["records"])}
    if prior.shape != (len(used), 12, 1000):
        raise ValueError("Original development cache shape mismatch")
    output.mkdir(mode=0o700)
    (output / "sources").mkdir(mode=0o700)
    for s in SOURCE_SEEDS:
        # Byte-copy an existing source checkpoint, never torch.save a new model.
        p = project_json(old / f"source_completions/{s}.json", [("artifact",)])["artifact",]
        shutil.copyfile(safe_path(p, old / "artifacts"), output / sources[str(s)]["checkpoint"])
    new_json(output / "records.json", pool)
    new_json(output / "exclusion_hashes.json", sorted(guard.exclusions))
    new_json(output / "episodes.json", episodes)
    new_json(output / "sources.json", sources)
    new_json(output / "reference_fits.json", references)
    new_json(output / "recorded_masks.json", masks)
    new_json(output / "selector_evidence.json", target_entries)
    np.save(output / "labels.npy", np.zeros((len(pool), 2), dtype=np.float32), allow_pickle=False)
    labels = np.lib.format.open_memmap(output / "labels.npy", mode="r+")
    waves = np.lib.format.open_memmap(
        output / "signals.npy", mode="w+", dtype=np.float32, shape=(len(pool), 12, 1000)
    )
    raw_hashes, matched = {}, 0
    for i, rid in enumerate(pool):
        guard.check([rid])  # BEFORE any header, signal, annotation or checksum read.
        p = safe_path(raw / (rid + ".hea"), raw)
        y, obs = decode_labels(p.read_bytes(), mapping)
        if obs != [True, True]:
            raise ValueError("Unknown adaptation label: do not manufacture a negative")
        metadata = header(p)
        signal = safe_path(p.parent / metadata["signal_file"], raw)
        if signal.stem != p.stem:
            raise PermissionError("Shared/non-record-specific signal forbidden")
        x = to_100hz_first10(physical_wave(p.with_suffix(""), metadata), metadata["fs"])
        labels[i], waves[i] = y, x
        raw_hashes[rid] = dict(header_sha256=sha(p), signal_sha256=sha(signal))
        if rid in used:
            entry = cp["target_annotations",][rid]
            q = safe_path(entry["path"], old / "target_annotations")
            if sha(q) != entry["sha"]:
                raise PermissionError("Original adaptation annotation hash mismatch")
            v = project_json(q, [("y",), ("observed",)])
            if v["y",] != y or v["observed",] != obs or not np.array_equal(prior[index[rid]], x):
                raise PermissionError("V2/current signal differs from original training inputs")
            matched += 1
        if (i + 1) % 500 == 0:
            print(f"ADAPTATION_BUNDLE_RECORDS={i + 1}", flush=True)
    labels.flush()
    waves.flush()
    counts = {
        f"{a}{b}": int(np.all(labels == [a, b], axis=1).sum()) for a in (0, 1) for b in (0, 1)
    }
    new_json(output / "raw_input_hashes.json", raw_hashes)
    files = {p.relative_to(output).as_posix(): sha(p) for p in output.rglob("*") if p.is_file()}
    manifest = dict(
        schema="phase4-adaptation-only-bundle-v1",
        adaptation_records=len(pool),
        test_record_ids_in_bundle=0,
        test_exclusion_hash_count=len(test),
        adaptation_membership_sha256=digest(pool),
        test_overlap=0,
        all_included_records_in_adaptation_pool=True,
        label_combination_counts=counts,
        v2_decoder_sha256=sha(root / "src/dmvmat_ecg/phase3b_eval_v2/decoder.py"),
        mapping_sha256=MAPPING_HASH,
        original_training_records_matched=matched,
        all_budget_draw_union_records=len(used),
        source_checkpoint_hashes_match=True,
        source_seeds=list(SOURCE_SEEDS),
        draw_seeds=list(DRAW_SEEDS),
        optimizer=OPTIMIZER,
        updates=200,
        batch_size=32,
        files=files,
        field_allowlist_and_projection_hashes=accesses,
    )
    new_json(output / "MANIFEST.json", manifest)
    return {
        k: v
        for k, v in manifest.items()
        if k not in {"files", "field_allowlist_and_projection_hashes"}
    }


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(make_bundle(), indent=2))
