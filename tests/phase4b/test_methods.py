"""Invented tensors and identifiers only; never reads the project datasets."""

import numpy as np
import pytest
import torch

from dmvmat_ecg.phase4b.alignment import alignment_loss, mk_mmd
from dmvmat_ecg.phase4b.common import Target
from dmvmat_ecg.phase4b.source_data import project_source_csv


def test_U_labels_fail_before_id_lookup():
    t = object.__new__(Target)
    with pytest.raises(PermissionError, match="UNLABELLED"):
        t.labels("U")


def test_unknown_label_role_fails():
    t = object.__new__(Target)
    with pytest.raises(PermissionError):
        t.labels("test")


def test_allowed_label_getter_does_not_read_other_records():
    class Labels:
        def __getitem__(self, ix):
            assert list(ix) == [0, 2]
            return np.array([[0, 1], [1, 0]], dtype=np.float32)

    t = object.__new__(Target)
    t._labels = Labels()
    t.label_access = []
    t.ids = lambda role, draw, budget: ["INVENTED_ALLOWED_A", "INVENTED_ALLOWED_B"]
    t.index = lambda ids: np.array([0, 2])
    assert t.labels("F", 16101, 64).shape == (2, 2)


def test_L2SP_sum_and_source_anchor_gradient():
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    ref = torch.tensor([0.0, 1.0])
    penalty = 1e-3 * torch.square(p - ref).sum()
    penalty.backward()
    assert np.allclose(p.grad.numpy(), [0.002, 0.002])


def test_LPFT_add_group_keeps_head_moments():
    head = torch.nn.Parameter(torch.ones(2))
    backbone = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([head], lr=3e-4, betas=(0.9, 0.99))
    head.sum().backward()
    opt.step()
    opt.zero_grad()
    before = opt.state[head]["exp_avg"].clone()
    opt.add_param_group({"params": [backbone]})
    assert torch.equal(opt.state[head]["exp_avg"], before)
    assert not opt.state[backbone]


def test_MMD_identical_samples_and_nonnegative_shift():
    torch.manual_seed(7)
    x = torch.randn(12, 5)
    assert abs(float(mk_mmd(x, x))) < 1e-6
    assert float(mk_mmd(x, x + 2)) > 0


def test_MMD_backpropagates_to_both_domains_all_seven_modules():
    torch.manual_seed(7)
    a = {str(i): torch.randn(8, 5, requires_grad=True) for i in range(7)}
    b = {str(i): (torch.randn(9, 5) + 2).requires_grad_() for i in range(7)}
    alignment_loss(a, b).backward()
    for v in [*a.values(), *b.values()]:
        assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0


def test_source_CSV_skips_forbidden_row_without_decoding_annotation(tmp_path):
    path = tmp_path / "INVENTED_SOURCE.csv"
    path.write_text(
        "ecg_id,patient_id,scp_codes,strat_fold\n"
        "1,1,\"{'1AVB': 100}\",1\n"
        "FORBIDDEN_ID,FORBIDDEN_PATIENT,NOT_PYTHON_ANNOTATION,10\n"
        "2,2,\"{'NORM': 100}\",9\n"
    )
    result = project_source_csv(path, {"1", "2"})
    assert set(result) == {"1", "2"}
    assert result["1"]["fold"] == 1 and result["2"]["fold"] == 9


def test_source_CSV_fails_instead_of_loading_out_of_scope_record(tmp_path):
    path = tmp_path / "INVENTED_SOURCE.csv"
    path.write_text("ecg_id,patient_id,scp_codes,strat_fold\nFORBIDDEN,NEVER_DECODE,NOT_VALID,10\n")
    with pytest.raises(PermissionError):
        project_source_csv(path, {"FORBIDDEN"})


def test_AdaBN_reads_only_signals_preserves_parameters_and_is_deterministic():
    from dmvmat_ecg.phase3b.roles import Normalization
    from dmvmat_ecg.phase4b.target_fit import adabn

    class Toy(torch.nn.Module):
        input_schema = "fictional-vector"

        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.dropout = torch.nn.Dropout(0.8)
            self.prediction_head = torch.nn.Linear(3, 2)

        def forward(self, wave):
            return self.prediction_head(self.dropout(self.bn(wave)))

    class Unlabelled:
        x = np.random.default_rng(8).normal(size=(64, 3)).astype(np.float32) + 3
        label_access = []

        def ids(self, role):
            assert role == "U"
            return list(range(64))

        def index(self, ids):
            return np.asarray(ids)

        def labels(self, *args):
            raise AssertionError("Unlabelled labels must never be requested")

    torch.manual_seed(9)
    source = Toy().eval()
    norm = Normalization(0.0, 1.0, "INVENTED", "fresh-source-only")
    a, b = adabn(source, norm, Unlabelled()), adabn(source, norm, Unlabelled())
    for name, parameter in source.named_parameters():
        assert torch.equal(parameter, dict(a.named_parameters())[name])
    assert a.bn.num_batches_tracked.item() == 2
    assert not a.dropout.training and not a.bn.training
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
