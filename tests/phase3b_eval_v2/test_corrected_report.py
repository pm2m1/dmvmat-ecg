import numpy as np

from dmvmat_ecg.phase3b.protocol import roster
from dmvmat_ecg.phase3b_eval.analysis import analyze
from dmvmat_ecg.phase3b_eval.statistics import metrics


def test_complete_synthetic_report(tmp_path):
    import json

    from dmvmat_ecg.phase3b_eval_v2.report_corrected import report

    y = np.array([[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]], float)
    p = y * 0.6 + 0.2
    obs = np.ones_like(y, bool)
    out = tmp_path / "results/phase3b_eval_v2"
    out.mkdir(parents=True)
    folder = tmp_path / "private/phase3b_cpsc_v1"
    folder.mkdir(parents=True)
    result = analyze(
        y,
        obs,
        {"one": p},
        {name: "one" for name in roster()},
        np.arange(len(y)),
        folder,
        out,
        draws=50,
        minimum_valid=1,
    )
    source = tmp_path / "results/phase3b"
    source.mkdir()
    point = metrics(y, p, obs)
    (source / "SOURCE_VALIDATION_RESULTS.json").write_text(
        json.dumps(
            {
                str(s): dict(
                    report=dict(
                        source_stop_per_label=[
                            dict(AP=r["AP"], AUROC=r["AUROC"]) for r in point["per_label"]
                        ]
                    )
                )
                for s in (15101, 15102, 15103)
            }
        )
    )
    (folder / "MEMBERSHIPS.json").write_text(
        json.dumps(dict(source=dict(source_stop=[dict(y=row) for row in y.tolist()])))
    )
    (tmp_path / "docs/phase3").mkdir(parents=True)
    report(
        tmp_path,
        result,
        y,
        obs,
        dict(evaluated=6, excluded_short=0, truncated_first10=0),
        "fictional-seal",
        {"one": dict(seconds=0.1)},
    )
    text = (tmp_path / "docs/phase3/PHASE3B_LOCKED_EVALUATION_RESULTS_V2.md").read_text()
    assert "Inconclusive" in text and "fictional-seal" in text
    assert "corrected\nsecond outcome read" in text and "Pre-access revision" in text
    assert len(list((out / "figures").glob("*.pdf"))) == 2
