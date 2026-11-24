"""Synthetic paths only; never open clinical files."""

import os

import pytest

from dmvmat_ecg.phase5.common import BUNDLE, GEORGIA, ROOT, RUN, SOURCE, Scope


@pytest.mark.parametrize("mode", ["source", "CPSC", "Georgia"])
@pytest.mark.parametrize(
    "name",
    [
        "data/raw/cinc2021_v1_0_3/training/georgia/g1/INVENTED.hea",
        "data/raw/cinc2021_v1_0_3/training/cpsc_2018/g1/INVENTED.mat",
        "private/phase4c_confirm_run/outcomes.npy",
        "private/phase4c_confirm_run/predictions/INVENTED.npy",
        "data/raw/ptbxl_v1_0_3/records100/INVENTED_fold10.dat",
    ],
)
def test_no_raw_or_old_test_capability(mode, name):
    with pytest.raises(PermissionError):
        Scope(mode).check("open", (str(ROOT / name), "rb", os.O_RDONLY))


def test_source_mode_never_target():
    for p in (BUNDLE / "signals.npy", BUNDLE / "labels.npy", GEORGIA / "signals.npy"):
        with pytest.raises(PermissionError):
            Scope("source").check("open", (str(p), "rb", os.O_RDONLY))
    Scope("source").check("open", (str(SOURCE / "source_train_x.npy"), "rb", os.O_RDONLY))


@pytest.mark.parametrize("mode", ["source", "CPSC", "Georgia"])
def test_existing_inputs_read_only(mode):
    with pytest.raises(PermissionError):
        Scope(mode).check("open", (str(SOURCE / "source_train_x.npy"), "wb", os.O_WRONLY))
    Scope(mode).check("open", (str(RUN / "invented.json"), "wb", os.O_WRONLY | os.O_CREAT))


def test_no_test_mode():
    with pytest.raises(ValueError):
        Scope("test")
