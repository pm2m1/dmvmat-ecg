"""Replay original-source controls from permitted source/G inputs, never B."""

import argparse
import json

import torch

from dmvmat_ecg.phase3b.models import random_count_cost
from dmvmat_ecg.phase3b.protocol import seed
from dmvmat_ecg.phase4.runtime import mask_dict, mask_index
from dmvmat_ecg.phase4.safety import new_json

from .aligned_targets import decision, source_bank
from .common import BUNDLE, RUN, load_source, setup
from .features import Features
from .source_data import SourceData


def replay(gate_only=False):
    torch.set_num_threads(4)
    target = setup()
    source = SourceData()
    features = Features()
    expected = json.loads((BUNDLE / "recorded_masks.json").read_text())
    rows = []
    contexts = {}
    for b in (128, 64):
        for s in (15101, 15102, 15103):
            if s not in contexts:
                model, norm, _, _ = load_source(target, s)
                contexts[s] = (model, norm, source_bank(model, norm, source, features))
            model, norm, bank = contexts[s]
            for d in (16101, 16102, 16103):
                m = decision(target, source, model, norm, bank, s, d, b, "original_reconstruction")
                random, info = random_count_cost(model, mask_dict(m), seed([78601, s, d, b]))
                if b == 128:
                    reference = expected[f"{s}/{d}"]
                    if mask_dict(m) != reference["selective_a"] or random != reference["random_a"]:
                        raise RuntimeError("ORIGINAL_CONTROL_B128_RECONCILIATION_FAILED")
                rows.append(
                    dict(
                        source_seed=s,
                        draw_seed=d,
                        budget=b,
                        selector_mask=f"{m:07b}",
                        random_mask=f"{mask_index(random):07b}",
                        random_control=info,
                        B_used=False,
                        origin="deterministic production replay from source and G",
                        matches_retained_original_record=True if b == 128 else None,
                    )
                )
                print(f"ORIGINAL_CONTROL_REPLAY source={s} draw={d} budget={b}", flush=True)
                if gate_only:
                    destination = RUN / "ORIGINAL_CONTROL_REPLAY_GATE.json"
                    if not destination.exists():
                        new_json(destination, dict(status="PASS", verified=rows[0]))
                    return rows
    destination = RUN / "RECONSTRUCTED_ORIGINAL_CONTROLS.json"
    value = dict(
        status="PASS",
        all_nine_B128_selector_and_random_masks_match_retained_records=True,
        B64_masks_are_reconstructed_not_independently_retained_original_receipts=True,
        production_monte_carlo=True,
        B_used=False,
        rows=rows,
    )
    if destination.exists():
        if json.loads(destination.read_text()) != value:
            raise PermissionError("Prior control reconstruction differs")
    else:
        new_json(destination, value)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-only", action="store_true")
    replay(parser.parse_args().gate_only)
