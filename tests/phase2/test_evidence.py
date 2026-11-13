"""Fictional patients and synthetic features only; never loads ECG records."""

from dataclasses import replace

import numpy as np
import pytest

from dmvmat_ecg.phase2.contracts import MODULES, Embeddings, observed_labels
from dmvmat_ecg.phase2.evidence import (
    EvidenceConfig,
    analyze,
    holm,
    logical_vote,
    patient_weights,
    permutation_mmd,
    relative_geometry,
)


@pytest.fixture
def cfg():
    # Explicit reduced Monte Carlo counts for fast fixtures, not study settings.
    return EvidenceConfig(permutations=39, bootstraps=199)


def fixture(n=160, seed=17):
    rng = np.random.default_rng(seed)
    y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (n // 4, 1)).astype(float)
    z = rng.normal(0, 0.2, (len(y), 4))
    z[:, :2] += y * 3
    return z, y, np.ones_like(y, bool)


def embeddings(z, y, mask, role):
    return Embeddings(
        dict.fromkeys(MODULES, z),
        y,
        mask,
        tuple(f"fictional-{role}-{i}" for i in range(len(y))),
        role,
        "fictional-source-checkpoint",
        "fictional-preprocessing",
    )


@pytest.mark.parametrize("formulation", ["A", "B"])
def test_identical_geometry_retains_signed_direction(formulation, cfg):
    z, y, obs = fixture()
    r = relative_geometry(z, y, obs, z, y, obs, label=0, formulation=formulation, config=cfg)
    assert r["valid"] and r["retention"] == pytest.approx(1)
    assert r["upper"] >= 1
    assert r["quantile"] == pytest.approx(1 - 0.05 / 14)


@pytest.mark.parametrize("formulation", ["A", "B"])
def test_class_direction_reversal_and_degradation(formulation, cfg):
    z, y, obs = fixture()
    flipped = -z
    r = relative_geometry(z, y, obs, flipped, y, obs, label=0, formulation=formulation, config=cfg)
    assert r["valid"] and r["retention"] == pytest.approx(-1)
    assert r["upper"] < 0.75


@pytest.mark.parametrize("formulation", ["A", "B"])
def test_translation_and_scale_preserve_geometry(formulation, cfg):
    z, y, obs = fixture()
    r = relative_geometry(
        z, y, obs, 4 * z + 20, y, obs, label=1, formulation=formulation, config=cfg
    )
    assert r["valid"] and r["retention"] == pytest.approx(1)


@pytest.mark.parametrize("what", ["unknown_negative", "few_positive", "constant", "no_separation"])
def test_invalid_geometry_abstains(what, cfg):
    z, y, obs = fixture()
    if what == "unknown_negative":
        obs[y[:, 0] == 0, 0] = False
        y[~obs] = np.nan
    elif what == "few_positive":
        y[:, 0] = 0
        y[:4, 0] = 1
    elif what == "constant":
        z[:] = 1
    else:
        z = np.tile(np.arange(4)[:, None], (40, 4)).astype(float)
        # Repeated equal feature distribution for positive and negative groups.
        z[:, 0] = np.arange(len(z)) % 4
        y = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (40, 1)).astype(float)
        z = np.tile(np.arange(40)[:, None], (1, 4)).repeat(4, axis=0).astype(float)
    r = relative_geometry(z, y, obs, z, y, obs, label=0, config=cfg)
    assert not r["valid"]
    assert not logical_vote(r, dict(valid=True, q=0.001), cfg)


def test_uncertain_bootstrap_denominator_abstains(cfg):
    z, y, obs = fixture()
    # Valid point geometry, but all bootstrap mass on the same joint cell.
    w = np.zeros((cfg.bootstraps, len(y)), int)
    w[:, 0] = len(y)
    r = relative_geometry(z, y, obs, z, y, obs, label=0, config=cfg, weights=(w, w))
    assert not r["valid"] and r["bootstrap_valid_fraction"] == 0


def test_joint_cells_preserve_overlapping_diagnoses_and_abstain_when_missing(cfg):
    z, y, obs = fixture()
    obs[(y == [1, 1]).all(1)] = False
    r = relative_geometry(z, y, obs, z, y, obs, label=0, formulation="B", config=cfg)
    assert not r["valid"]
    # The marginal one-label formulation still has verified positives/negatives.
    assert relative_geometry(z, y, obs, z, y, obs, label=0, config=cfg)["valid"]


def test_cooccurrence_standardization_removes_measured_mixture_change(cfg):
    rng = np.random.default_rng(20)

    def domain(counts):
        labels = np.concatenate(
            [
                np.tile(pair, (count, 1))
                for pair, count in zip([[0, 0], [0, 1], [1, 0], [1, 1]], counts, strict=True)
            ]
        )
        noise = rng.normal(0, 0.15, (len(labels), 3))
        noise[:, 0] += labels[:, 0] + 4 * labels[:, 1]
        return noise, labels.astype(float), np.ones_like(labels, bool)

    s, sy, sm = domain([100, 20, 20, 100])
    t, ty, tm = domain([20, 100, 100, 20])
    a = relative_geometry(s, sy, sm, t, ty, tm, label=0, config=cfg)
    b = relative_geometry(s, sy, sm, t, ty, tm, label=0, formulation="B", config=cfg)
    assert a["valid"] and b["valid"]
    assert a["retention"] < 0 and b["retention"] == pytest.approx(1, abs=0.12)


def test_whole_patient_bootstrap_weights_are_shared_not_label_stratified():
    w = patient_weights(80, 199, 99)
    np.testing.assert_array_equal(w.sum(1), 80)
    np.testing.assert_array_equal(w, patient_weights(80, 199, 99))
    # A person's two diagnoses always carry the same multiplicity.
    _, y, _ = fixture(80)
    np.testing.assert_array_equal(w @ (y[:, 0] * y[:, 1]), w[:, 3::4].sum(1))


def test_mmd_p_value_uses_plus_one_and_fixed_null(cfg):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 4))
    r = permutation_mmd(x, x + 5, config=cfg)
    expected = (1 + np.sum(np.asarray(r["null"]) >= r["statistic"])) / 40
    assert r["valid"] and r["p"] == expected and r["p"] == 0.025
    assert permutation_mmd(x, x + 5, config=cfg) == r
    identical = permutation_mmd(x, x, config=cfg)
    assert identical["statistic"] < 1e-12 and identical["p"] > 0.5


@pytest.mark.parametrize("n", [0, 1])
def test_mmd_insufficient_patients(n, cfg):
    result = permutation_mmd(np.ones((n, 3)), np.ones((4, 3)), config=cfg)
    assert not result["valid"] and result["p"] == 1


def test_constant_mmd_not_evidence(cfg):
    result = permutation_mmd(np.ones((20, 3)), np.ones((20, 3)), config=cfg)
    assert not result["valid"] and result["reason"] == "constant_embeddings"


def test_holm_known_values():
    np.testing.assert_allclose(holm([0.03, 0.01, 0.8]), [0.06, 0.03, 0.8])
    np.testing.assert_allclose(holm([1] * 7), 1)


@pytest.mark.parametrize(
    "r,u,q,valid,expected",
    [
        (0.4, 0.74, 1, True, True),
        (0.7, 0.9, 0.05, True, True),
        (0.7, 0.9, 0.051, True, False),
        (0.8, 0.9, 0.001, True, False),
        (0.2, 1.0, 0.001, True, False),
        (0, 0, 0, False, False),
    ],
)
def test_policy_boundaries(r, u, q, valid, expected, cfg):
    assert (
        logical_vote(dict(valid=valid, retention=r, upper=u), dict(valid=True, q=q), cfg)
        is expected
    )


def test_seven_modules_shift_controls_reproducibility_and_caps(cfg):
    z, y, obs = fixture()
    source = embeddings(z, y, obs, "source_train")
    translated = embeddings(z + 8, y, obs, "evidence")
    r = analyze(source, translated, config=cfg)
    assert set(r["mask"]) == set(MODULES)
    assert set(r["mask"].values()) == {"SHARE"}  # MMD alone is not an adaptation vote.
    assert all(v["mmd"]["p"] <= 0.05 for v in r["modules"].values())
    assert r == analyze(source, translated, config=cfg)
    reverse = analyze(source, embeddings(-z, y, obs, "evidence"), config=cfg)
    assert set(reverse["mask"].values()) == {"ADAPT"}
    capped = analyze(source, translated, config=replace(cfg, source_class_cap=20, mmd_cap=20))
    assert np.max(capped["source_class_counts"]) <= 20
    assert all(v["mmd"]["n_source"] == 20 for v in capped["modules"].values())


def test_all_unknown_labels_produce_share_abstention(cfg):
    z, y, obs = fixture(40)
    obs[:] = False
    y[:] = np.nan
    result = analyze(
        embeddings(z, y, obs, "source_train"), embeddings(z, y, obs, "evidence"), config=cfg
    )
    assert set(result["mask"].values()) == {"SHARE"}
    assert all(v["abstained_labels"] == 2 for v in result["modules"].values())


def test_b_module_pipeline_and_partial_label_abstention(cfg):
    z, y, obs = fixture()
    source = embeddings(z, y, obs, "source_train")
    target = embeddings(-z, y, obs, "evidence")
    result = analyze(source, target, formulation="B", config=cfg)
    assert set(result["mask"].values()) == {"ADAPT"}
    obs[:, 1] = False
    target = embeddings(-z, y, obs, "evidence")
    a = analyze(source, target, config=cfg)
    b = analyze(source, target, formulation="B", config=cfg)
    assert set(a["mask"].values()) == {"ADAPT"}
    assert all(v["abstained_labels"] == 1 for v in a["modules"].values())
    assert set(b["mask"].values()) == {"SHARE"}


@pytest.mark.parametrize(
    "changes",
    [
        {"alpha": float("nan")},
        {"minimum": 1},
        {"permutations": 3},
        {"bootstraps": 4},
        {"source_d_floor": float("inf")},
    ],
)
def test_invalid_evidence_configuration_fails_closed(changes):
    with pytest.raises(ValueError):
        EvidenceConfig(**changes)


def test_default_registered_draw_counts_execute_on_small_fictional_sample():
    z, y, obs = fixture(48)
    result = analyze(embeddings(z, y, obs, "source_train"), embeddings(z, y, obs, "evidence"))
    assert result["configuration"]["permutations"] == 999
    assert result["configuration"]["bootstraps"] == 4999
    assert set(result["mask"].values()) == {"SHARE"}


@pytest.mark.parametrize("bad", ["groups", "checkpoint", "preprocessing", "dimensions"])
def test_evidence_lineage_shape_and_overlap_rejected(bad, cfg):
    z, y, obs = fixture(40)
    s, t = embeddings(z, y, obs, "source_train"), embeddings(z, y, obs, "evidence")
    if bad == "groups":
        t = replace(t, groups=s.groups)
    elif bad == "checkpoint":
        t = replace(t, checkpoint_hash="another-source")
    elif bad == "preprocessing":
        t = replace(t, preprocessing_hash="another-preprocessor")
    else:
        t = replace(t, features={**t.features, "view_1": np.ones((40, 2))})
    with pytest.raises((ValueError, PermissionError)):
        analyze(s, t, config=cfg)


def test_unknown_placeholders_ignored_observed_corruption_rejected():
    y = np.array([[1, np.nan], [0, -999]])
    m = np.array([[True, False], [True, False]])
    clean, mask = observed_labels(y, m)
    assert clean[:, 1].sum() == 0 and not mask[:, 1].any()
    y[0, 0] = np.nan
    with pytest.raises(ValueError):
        observed_labels(y, m)
    with pytest.raises(ValueError):
        observed_labels(np.zeros((2, 2)), np.ones((2, 2)))


def test_b_is_not_silently_generalized_to_five_labels(cfg):
    z, y, obs = fixture(40)
    y = np.tile(y[:, :1], (1, 5))
    with pytest.raises(ValueError):
        relative_geometry(
            z,
            y,
            np.ones_like(y, bool),
            z,
            y,
            np.ones_like(y, bool),
            label=0,
            formulation="B",
            config=cfg,
        )
