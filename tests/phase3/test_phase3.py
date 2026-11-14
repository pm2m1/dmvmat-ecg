"""Fictional fixtures only. No real target or historical test data is loaded."""

import csv
import json

import numpy as np
import pytest
import torch
from scipy.io import savemat
from sklearn.metrics import average_precision_score, roc_auc_score

from dmvmat_ecg.phase3.download import SOURCES, selected_entries
from dmvmat_ecg.phase3.labels import CONDITIONS, mapping, targets
from dmvmat_ecg.phase3.prepare import source_proxy_labels, write_new
from dmvmat_ecg.phase3.signals import (
    LEADS,
    header,
    lead_order,
    native_identity,
    physical_wave,
    to_100hz_first10,
    unit_factor,
)
from dmvmat_ecg.phase3.splits import require_feas, require_fold9, split
from dmvmat_ecg.phase3.statistics import bootstrap, mmd, percentile, prior_adjusted_AP, ranking
from dmvmat_ecg.preprocessing.signals import normalize, views


def test_publisher_filter_excludes_other_sources_and_unsafe_paths():
    index = "\n".join("a" * 64 + f"  training/{s}/x.hea" for s in SOURCES)
    result = selected_entries(index + "\n" + "b" * 64 + "  training/ptb-xl/x.hea")
    assert len(result) == 5
    assert all("ptb-xl" not in name for name in result)
    with pytest.raises(ValueError):
        selected_entries(index + "\n" + "a" * 64 + "  ../x")


def test_curl_fallback_disjoint_eight_clients_and_safe_configuration(tmp_path):
    from dmvmat_ecg.phase3.curl_fallback import configs

    names = [f"training/georgia/fake{i}.hea" for i in range(25)]
    paths = configs(tmp_path, names, prefix="fictional")
    assert len(paths) == 8
    assert sum(p.read_text().count("url = ") for p in paths) == 25
    assert all("retry = 9" in p.read_text() and "ipv4" in p.read_text() for p in paths)
    assert all("parallel" not in p.read_text() for p in paths)
    with pytest.raises(ValueError):
        configs(tmp_path, ["training/ptb-xl/fake.mat"], prefix="forbidden")
    with pytest.raises(ValueError):
        configs(tmp_path, ["../fake.mat"], prefix="escape")


def test_curl_sets_running_before_initial_checksum_scan(tmp_path, monkeypatch):
    from dmvmat_ecg.phase3 import curl_fallback

    (tmp_path / "SHA256SUMS.txt").write_text("fictional publisher index")

    def checked_index(text):
        status = json.loads((tmp_path / "DOWNLOAD_STATUS.json").read_text())
        assert status["status"] == "running"
        return {}

    monkeypatch.setattr(curl_fallback, "selected_entries", checked_index)
    result = curl_fallback.run(tmp_path)
    assert result["files"] == 0 and result["returncode"] == 0


@pytest.mark.parametrize("unit,factor", [("mV", 1), ("uV", 0.001), ("μV", 0.001), ("V", 1000)])
def test_units(unit, factor):
    assert unit_factor(unit) == factor
    with pytest.raises(ValueError):
        unit_factor("unknown")


def test_header_mat_and_physical_conversion(tmp_path):
    names = list(reversed(LEADS))
    lines = ["fake 12 500 5000"]
    for name in names:
        lines.append(f"fake.mat 16+24 1000/mV 16 0 0 0 0 {name}")
    lines.append("#Dx: 270492004,59931005")
    (tmp_path / "fake.hea").write_text("\n".join(lines) + "\n")
    wave = np.array([np.full(5000, 1000 * (i + 1)) for i in range(12)], dtype=np.int16)
    savemat(tmp_path / "fake.mat", {"val": wave})
    meta = header(tmp_path / "fake.hea")
    actual = physical_wave(tmp_path / "fake", meta)
    assert meta["fs"] == 500 and meta["duration"] == 10
    np.testing.assert_allclose(actual[:, 0], np.arange(12, 0, -1))
    assert len(meta["dx"]) == 2


def test_wfdb_header_declared_alias_not_assumed_stem(tmp_path):
    lines = (
        ["fictional_header 12 100 1000"]
        + [f"actual_signal.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in LEADS]
        + ["#Dx: 270492004"]
    )
    path = tmp_path / "fictional_header.hea"
    path.write_text("\n".join(lines) + "\n")
    wave = np.arange(12000, dtype=np.int16).reshape(12, 1000)
    savemat(tmp_path / "actual_signal.mat", {"val": wave})
    meta = header(path)
    assert meta["signal_file"] == "actual_signal.mat"
    np.testing.assert_allclose(physical_wave(path, meta), wave / 1000)


def test_leads_normalization_and_view_shapes():
    assert lead_order(list(reversed(LEADS))) == list(range(11, -1, -1))
    with pytest.raises(ValueError):
        lead_order(["I"] * 12)
    x = np.arange(12000, dtype=np.float32).reshape(1, 12, 1000)
    result = normalize(x, {"mean": 3, "std": 7})
    np.testing.assert_allclose(result, ((x.astype(float) - 3) / 7).astype(np.float32))
    a, b, c = views(result)
    assert a.shape == (1, 12, 1000)
    assert b.shape == (1, 3, 1000)
    assert c.shape == (1, 12, 396)


def test_first_ten_seconds_antialias_and_no_padding():
    fs = 500
    time = np.arange(6000) / fs
    signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 150 * time)
    wave = np.tile(signal, (12, 1))
    actual = to_100hz_first10(wave, fs)
    assert actual.shape == (12, 1000) and actual.dtype == np.float32
    np.testing.assert_allclose(actual, to_100hz_first10(wave[:, :5000], fs))
    expected = np.sin(2 * np.pi * 10 * np.arange(1000) / 100)
    assert np.mean((actual[0, 20:-20] - expected[20:-20]) ** 2) < 0.001
    with pytest.raises(ValueError, match="never padded"):
        to_100hz_first10(wave[:, :4999], fs)


def test_native_duplicate_identity_no_resampling():
    x = np.random.default_rng(2).normal(size=(12, 5000))
    assert native_identity(x, 500) == native_identity(x.copy(), 500)
    assert native_identity(x, 500) != native_identity(x, 100)
    x[0, 0] += 0.001
    assert native_identity(x, 500) != native_identity(x[:, 1:], 500)


def test_mapping_coverage_and_annotation_not_clinical_absence():
    observed = ["270492004", "59931005", "426783006", "164909002", "777"]
    frozen = mapping(observed)
    assert targets(observed[:2], frozen) == [1, 1]
    assert targets(["426783006", "777", "164909002"], frozen) == [0, 0]
    assert "not confirmed clinical absence" in frozen["negative_semantics"]
    assert "164909002" not in frozen["proxy_snomed"]["CD"]
    with pytest.raises(ValueError):
        targets(["unknown"], frozen)
    for label in ["CD", "STTC"]:
        assert set(frozen["proxy_scp"][label]) == {
            scp for value in CONDITIONS.values() if value["label"] == label for scp in value["scp"]
        }
    assert "2AVB" not in frozen["proxy_scp"]["CD"]
    assert "ISC_" not in frozen["proxy_scp"]["STTC"]


def test_source_proxy_only_decodes_fold9(tmp_path):
    path = tmp_path / "fake.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ecg_id", "strat_fold", "scp_codes"])
        writer.writeheader()
        writer.writerows(
            [
                {"ecg_id": 1, "strat_fold": 9, "scp_codes": "{'1AVB': 100, 'STD_': 100}"},
                {"ecg_id": 2, "strat_fold": 10, "scp_codes": "not parsed; would raise"},
                {"ecg_id": 3, "strat_fold": 9, "scp_codes": "{'WPW': 100, 'ISC_': 100}"},
            ]
        )
    y, codes = source_proxy_labels(path, [1, 3], mapping([]))
    assert y.tolist() == [[1, 1], [0, 0]]
    assert codes.shape == (2, 12)
    with pytest.raises(PermissionError):
        source_proxy_labels(path, [2], mapping([]))


def test_source_metadata_hash_must_match_original_audit(tmp_path):
    from dmvmat_ecg.phase3.download import sha
    from dmvmat_ecg.phase3.inference import source_metadata_identity

    folder = tmp_path / "data/raw/ptbxl_v1_0_3"
    folder.mkdir(parents=True)
    a, b = folder / "ptbxl_database.csv", folder / "scp_statements.csv"
    a.write_text("fictional metadata")
    b.write_text("fictional SCP dictionary")
    write_new(
        tmp_path / "results/dataset_audit.json",
        {"release": "1.0.3", "metadata_sha256": sha(a), "statements_sha256": sha(b)},
    )
    assert len(source_metadata_identity(tmp_path)) == 2
    a.write_text("modified fictional metadata")
    with pytest.raises(PermissionError, match="Source metadata changed"):
        source_metadata_identity(tmp_path)


def test_split_determinism_disjoint_and_reserved_guard():
    y = np.tile([[0, 0], [1, 0], [0, 1], [1, 1]], (26, 1))
    ids = ["fake-" + str(i) for i in range(len(y))]
    result = split(ids, y, "fictional")
    reversed_result = split(ids[::-1], y[::-1], "fictional")
    assert result == reversed_result and len(result["FEAS"]) == round(len(ids) * 0.2)
    assert not set(result["FEAS"]) & set(result["RESERVED"])
    assert set(result["FEAS"]) | set(result["RESERVED"]) == set(ids)
    require_feas(result["FEAS"][0], result)
    with pytest.raises(PermissionError):
        require_feas(result["RESERVED"][0], result)
    with pytest.raises(ValueError, match="seed"):
        split(ids, y, "fictional", seed=1)
    require_fold9([9, 9])
    with pytest.raises(PermissionError):
        require_fold9([9, 10])


def test_freezes_are_exclusive_and_valid_json(tmp_path):
    write_new(tmp_path / "frozen.json", {"a": [1]})
    assert json.loads((tmp_path / "frozen.json").read_text()) == {"a": [1]}
    with pytest.raises(FileExistsError):
        write_new(tmp_path / "frozen.json", {})


def test_exact_continuous_metrics_bootstrap_and_undefined_handling():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    p = np.array([[0.2, 0.7], [0.6, 0.6], [0.3, 0.9], [0.5, 0.1]])
    result = ranking(y, p)
    assert result[0]["ap"] == average_precision_score(y[:, 0], p[:, 0])
    assert result[1]["auroc"] == roc_auc_score(y[:, 1], p[:, 1])
    values = bootstrap(y, {"A": [p, p], "B": [p]}, draws=100)
    for metric in ["ap", "auroc"]:
        np.testing.assert_allclose(values["A"][metric], values["B"][metric], equal_nan=True)
    assert percentile(np.array([np.nan, np.nan]))["undefined"] == 2
    adjusted = prior_adjusted_AP(y, [p], [0.5, 0.5])
    np.testing.assert_allclose(adjusted, [row["ap"] for row in result])


def test_mmd_controls_and_reproducibility():
    x = np.random.default_rng(3).normal(size=(30, 8))
    same = mmd(x, x.copy(), permutations=99)
    changed = mmd(x, x + 4, permutations=99)
    assert same["p"] > 0.9 and same["statistic"] < 1e-8
    assert changed["p"] <= 0.02 and changed["statistic"] > same["statistic"]
    assert changed == mmd(x, x + 4, permutations=99)
    assert mmd(np.zeros((5, 3)), np.zeros((5, 3)))["valid"] is False


def test_frozen_four_model_execution_and_aggregate_report(tmp_path, monkeypatch):
    """An entire fictional inference-only run, no real data or fitted predictors."""
    from dmvmat_ecg.models.networks import build, parameters
    from dmvmat_ecg.phase3 import run
    from dmvmat_ecg.phase3.download import sha
    from dmvmat_ecg.phase3.prepare import prepare
    from dmvmat_ecg.phase3.report import report

    torch.set_num_threads(2)
    raw = tmp_path / "data/raw/cinc2021_v1_0_3"
    raw.mkdir(parents=True)
    checksum = []
    for s, source in enumerate(SOURCES):
        parent = raw / "training" / source
        parent.mkdir(parents=True)
        for i in range(24):
            record = "fake" + str(i)
            codes = ["426783006"]
            if i % 2:
                codes.append("270492004")
            if (i // 2) % 2:
                codes.append("59931005")
            lines = (
                [f"{record} 12 100 1000"]
                + [f"{record}.mat 16+24 1000/mV 16 0 0 0 0 {lead}" for lead in LEADS]
                + ["#Dx: " + ",".join(codes)]
            )
            (parent / (record + ".hea")).write_text("\n".join(lines) + "\n")
            wave = np.random.default_rng(10000 * s + i).integers(
                -2000, 2000, (12, 1000), dtype=np.int16
            )
            savemat(parent / (record + ".mat"), {"val": wave})
            for suffix in [".hea", ".mat"]:
                p = parent / (record + suffix)
                checksum.append(sha(p) + "  " + str(p.relative_to(raw)))
    (raw / "SHA256SUMS.txt").write_text("\n".join(checksum))
    source_counts = {
        s: {"expected_files": 48, "downloaded_files": 48, "bytes": 1, "mismatches": 0}
        for s in SOURCES
    }
    write_new(
        raw / "DOWNLOAD_VERIFICATION.json",
        {"complete": True, "shasum_returncode": 0, "sources": source_counts},
    )
    prepare(tmp_path)
    stats = {"mean": -0.001, "std": 0.23}
    private = tmp_path / "private/cache"
    private.mkdir(parents=True)
    for name in ["validation_x.npy", "validation_meta.npz", "standardization.json"]:
        (private / name).write_text("synthetic placeholder: loader is monkeypatched")
    refs = []
    for model in ["multiview", "flat_cnn"]:
        for seed in [101, 202]:
            network = build(model, seed)
            assert parameters(network) == (212261 if model == "multiview" else 206117)
            path = tmp_path / f"fictional_{model}_{seed}.pt"
            torch.save(network.state_dict(), path)
            refs.append(
                {
                    "model": model,
                    "seed": seed,
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha(path),
                }
            )
    monkeypatch.setattr(run, "reference_models", lambda root: (refs, stats))
    x = np.random.default_rng(42).normal(0, 0.2, (24, 12, 1000)).astype(np.float32)
    y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (6, 1))
    five = np.zeros((24, 5), np.int8)
    five[:, [3, 2]] = y
    monkeypatch.setattr(run, "fold9", lambda root: (x, {"ecg_id": np.arange(24), "y": five}))
    source_folder = tmp_path / "data/raw/ptbxl_v1_0_3"
    source_folder.mkdir()
    with (source_folder / "ptbxl_database.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ecg_id", "strat_fold", "scp_codes"])
        writer.writeheader()
        for i in range(24):
            codes = {}
            if y[i, 0]:
                codes["1AVB"] = 100
            if y[i, 1]:
                codes["INVT"] = 100
            writer.writerow({"ecg_id": i, "strat_fold": 9, "scp_codes": repr(codes)})
    run.freeze(tmp_path)
    run.predict(tmp_path)
    assert (tmp_path / "results/phase3a/PREDICTION_SEAL.json").exists()
    run.analyze(tmp_path)
    run.shifts(tmp_path)
    report(tmp_path)
    assert (tmp_path / "docs/phase3/CROSS_HOSPITAL_FEASIBILITY.md").exists()
    assert (tmp_path / "results/phase3a/REPORT_COMPLETE.json").exists()
