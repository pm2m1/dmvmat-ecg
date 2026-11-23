"""Compare allowlisted original source states; no target or test loader."""

import json

import numpy as np
import torch

from dmvmat_ecg.phase2.adaptation import state_hash
from dmvmat_ecg.phase3b.jobs import source_code_hash
from dmvmat_ecg.phase3b.models import fresh_model
from dmvmat_ecg.phase3b.protocol import adopted_settings
from dmvmat_ecg.phase3b_real.boundary import implementation_hash
from dmvmat_ecg.phase4.safety import digest, new_json, sha

from .data import BUNDLE, PLACEBO, RUN, SOURCE, SourceInputs, setup


def compare_states(original, retrained, parameter_names):
    if set(original) != set(retrained):
        raise PermissionError("Source architecture/state keys differ")
    changed = []
    for name in sorted(original):
        a, b = original[name].cpu(), retrained[name].cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise PermissionError("Source tensor shape/dtype differs")
        if a.numpy().tobytes(order="C") != b.numpy().tobytes(order="C"):
            changed.append(
                dict(
                    name=name,
                    kind="parameter" if name in parameter_names else "buffer",
                    max_absolute_difference=float((a.double() - b.double()).abs().max()),
                )
            )
    return dict(
        full_state_bitwise_identical=not changed,
        parameters_bitwise_identical=not any(v["kind"] == "parameter" for v in changed),
        buffers_bitwise_identical=not any(v["kind"] == "buffer" for v in changed),
        changed_parameter_tensors=sum(v["kind"] == "parameter" for v in changed),
        changed_buffer_tensors=sum(v["kind"] == "buffer" for v in changed),
        changed_tensors=changed,
    )


def run():
    setup("compare")
    if json.loads((RUN / "SOURCES_COMPLETE.json").read_text())["status"] != "COMPLETE":
        raise PermissionError("All three retrained sources required")
    features = SourceInputs()
    old_entries = json.loads((BUNDLE / "sources.json").read_text())
    if set(old_entries) != {"15101", "15102", "15103"}:
        raise PermissionError("Original source roster differs")
    roles = features.source.manifest["roles"]
    identities = [roles[r]["identity"] for r in ("source_train", "source_stop")]
    cache_equal = {}
    for role, entry in features.manifest["roles"].items():
        for kind, relative in entry["paths"].items():
            cache_equal[role + "/" + kind] = sha(SOURCE / "features" / f"{role}_{kind}.npy") == sha(
                PLACEBO / relative
            )
    if not all(cache_equal.values()):
        raise PermissionError("Source views differ from aligned-source cache")
    rows = []
    for s in (15101, 15102, 15103):
        entry = old_entries[str(s)]
        original_path = BUNDLE / entry["checkpoint"]
        current = json.loads((RUN / "source_models" / f"retrain_s{s}.json").read_text())
        retrain_path = RUN / current["checkpoint"]
        if sha(original_path) != entry["sha256"] or sha(retrain_path) != current["sha256"]:
            raise PermissionError("Source checkpoint file identity differs")
        original = torch.load(original_path, map_location="cpu", weights_only=True)
        retrain = torch.load(retrain_path, map_location="cpu", weights_only=True)
        if original["identity"]["data"] != identities or original["identity"]["seed"] != s:
            raise PermissionError("Original source membership/seed differs")
        if (
            original["normalization"] != retrain["normalization"]
            or retrain["normalization"] != features.norm.__dict__
        ):
            raise PermissionError("Original source normalization differs")
        model = fresh_model(s)
        model.load_state_dict(original["state"], strict=True)
        if (
            state_hash(model) != entry["state_hash"]
            or original["state_hash"] != entry["state_hash"]
        ):
            raise PermissionError("Original source state hash differs")
        model.load_state_dict(retrain["state"], strict=True)
        if state_hash(model) != current["state_hash"]:
            raise PermissionError("Retrained source state hash differs")
        if original["identity"]["settings"] != adopted_settings():
            raise PermissionError("Original source settings differ")
        comparison = compare_states(
            original["state"], retrain["state"], set(dict(model.named_parameters()))
        )
        rows.append(
            dict(
                source_seed=s,
                checkpoint_file_bytes_identical=sha(original_path) == sha(retrain_path),
                **comparison,
                original_selected_epoch=original["selected_epoch"],
                retrain_selected_epoch=retrain["selected_epoch"],
                original_factory=original["identity"]["factory"],
                retrain_factory=f"{fresh_model.__module__}.{fresh_model.__qualname__}",
                source_factory_identical=original["identity"]["factory"]
                == f"{fresh_model.__module__}.{fresh_model.__qualname__}",
                source_normalization_identical=True,
                original_source_code_matches_current_public=original["identity"]["code"]
                == source_code_hash(),
                original_real_executor_matches_current_public=original["identity"].get(
                    "implementation"
                )
                == implementation_hash(),
                original_has_optimizer_state=any("optim" in k.lower() for k in original),
                original_has_rng_state=any("rng" in k.lower() for k in original),
                original_has_library_version_record=any(
                    "version" in k.lower() or "environment" in k.lower() for k in original
                ),
                original_BCE_history_sha256=digest(
                    [e["training_bce"] for e in original["history"]]
                ),
                retrain_BCE_history_sha256=digest([e["BCE"] for e in retrain["history"]]),
                original_source_sha256=entry["sha256"],
                retrain_source_sha256=current["sha256"],
                retrain_fold9_macro_AUROC=current["fold9_macro_AUROC"],
            )
        )
    result = dict(
        schema="plain-retrain-source-comparison-v1",
        sources=rows,
        source_cache_payloads_match_alignment=cache_equal,
        source_bundle_sha256=features.source.identity,
        no_CPSC_data_read=True,
        original_executor="src/dmvmat_ecg/phase3b_real/engine.py:train_source",
        retrain_executor="src/dmvmat_ecg/phase4c_retrain/source.py:train",
        current_libraries=dict(torch=str(torch.__version__), numpy=np.__version__),
    )
    new_json(RUN / "SOURCE_COMPARISON.json", result)
    print(
        json.dumps(
            {
                "source_comparison_complete": True,
                "sources": [
                    {
                        k: r[k]
                        for k in (
                            "source_seed",
                            "checkpoint_file_bytes_identical",
                            "full_state_bitwise_identical",
                            "parameters_bitwise_identical",
                            "buffers_bitwise_identical",
                            "original_source_code_matches_current_public",
                            "original_real_executor_matches_current_public",
                        )
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
