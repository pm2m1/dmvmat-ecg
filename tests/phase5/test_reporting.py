"""Invented aggregate reporting fixture, no model/data files."""

import json
from types import SimpleNamespace

import pytest

from dmvmat_ecg.phase5.common import ARMS, BUDGETS, DRAWS, SEEDS


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_protocol_and_results_from_complete_invented_aggregates(tmp_path, monkeypatch):
    from dmvmat_ecg.phase5 import report

    run = tmp_path / "private/replication"
    bundle = tmp_path / "private/georgia"
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "RUN", run)
    monkeypatch.setattr(report, "GEORGIA", bundle)
    monkeypatch.setattr(report, "Scope", lambda _: SimpleNamespace(install=lambda: None))
    monkeypatch.setattr(
        report,
        "descriptor",
        lambda a, s: (
            None,
            None,
            None,
            dict(
                arm=a,
                source_seed=s,
                fold9_macro_AUROC=0.8 if a == "plain" else 0.81,
                fold9_AUROC=[0.8, 0.8],
                selected_epoch=2,
            ),
        ),
    )
    put(run / "SOURCES_COMPLETE.json", dict(status="COMPLETE", source_models=20))
    counts = dict(records=500, positive=[100, 150], prevalence=[0.2, 0.3])
    summary = dict(
        pool=dict(counts, records=6000),
        A=counts,
        B=counts,
        test_records=4001,
        survey_records=2000,
        all_draws_union=700,
    )
    put(bundle / "MANIFEST.json", dict(summary=summary))
    for h in ("CPSC", "Georgia"):
        ci = dict(interval=[-0.01, 0.03], valid=10000)
        metrics = [
            dict(
                arm=a,
                budget=b,
                macro_AP=0.6,
                macro_AUROC=0.8,
                A_macro_AP=0.6,
                A_macro_AUROC=0.8,
                CD_AP=0.5,
                STTC_AP=0.7,
            )
            for b in BUDGETS
            for a in ARMS
        ]
        contrasts = [
            dict(budget=b, AP=dict(estimate=0.01, CI95=ci), AUROC=dict(estimate=0.01, CI95=ci))
            for b in BUDGETS
        ]
        perseed = [
            dict(source_seed=s, budget=b, macro_AP_difference=0.01) for b in BUDGETS for s in SEEDS
        ]
        put(
            run / h / "VALIDATION_ANALYSIS.json",
            dict(hospital=h, metrics=metrics, contrasts=contrasts, per_seed=perseed),
        )
        for s in SEEDS:
            for a in ARMS:
                for b in BUDGETS:
                    for d in DRAWS:
                        put(
                            run / h / "fits" / f"{a}_s{s}_d{d}_b{b}.json",
                            dict(predictions_sha256="INVENTED", state_hash="INVENTED"),
                        )
    report.run()
    protocol = (tmp_path / "docs/phase5/PHASE5_GEORGIA_PROTOCOL.md").read_text()
    assert "label-blind" in protocol and "BITWISE" in protocol
    assert "190 physical" in protocol and "all270 logical" in protocol
    assert "test prevalence is deliberately unknown" in protocol.lower()
    text = (tmp_path / "docs/phase5/PHASE5_VALIDATION_RESULTS.md").read_text()
    assert "Not accessed" in text and "10 paired source" in text
    assert len(list((tmp_path / "figures/phase5").glob("*.svg"))) == 2
    with pytest.raises(FileExistsError):
        report.run()
