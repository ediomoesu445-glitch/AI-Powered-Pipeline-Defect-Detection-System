from pathlib import Path

import pytest
import torch

from src.dataset import CLASSES, NEUDataset, make_splits, parse_label, scan_dataset


def find_project_root(marker="CLAUDE.md"):
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looking for {marker})")


PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DATA_AVAILABLE = any(DATA_ROOT.rglob("*.jpg")) or any(DATA_ROOT.rglob("*.bmp"))

requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE, reason="NEU dataset not present in data/raw/ - download it first"
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Cr_1.bmp", "crazing"),
        ("In_23.bmp", "inclusion"),
        ("Pa_100.bmp", "patches"),
        ("PS_5.bmp", "pitted_surface"),
        ("RS_9.bmp", "rolled-in_scale"),
        ("Sc_2.bmp", "scratches"),
        ("crazing_1.jpg", "crazing"),
        ("inclusion_42.jpg", "inclusion"),
        ("rolled-in_scale_123.jpg", "rolled-in_scale"),
        ("PITTED_SURFACE_7.JPG", "pitted_surface"),
        ("scratches_9.jpeg", "scratches"),
    ],
)
def test_parse_label_both_conventions(filename, expected):
    assert parse_label(filename) == expected


def test_parse_label_raises_on_unrecognised_filename():
    with pytest.raises(ValueError):
        parse_label("not_a_real_class_1.jpg")


@pytest.fixture(scope="module")
def items():
    return scan_dataset(DATA_ROOT)


@pytest.fixture(scope="module")
def splits(items):
    return make_splits(items, seed=42)


@requires_data
def test_labels_in_range(items):
    for _, label in items:
        assert 0 <= label < len(CLASSES)


@requires_data
def test_splits_disjoint_by_filepath(splits):
    train, val, test = splits
    train_paths = {p for p, _ in train}
    val_paths = {p for p, _ in val}
    test_paths = {p for p, _ in test}

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)


@requires_data
def test_split_lengths_sum_to_total(items, splits):
    train, val, test = splits
    assert len(train) + len(val) + len(test) == len(items)


@requires_data
def test_all_classes_in_every_split(splits):
    train, val, test = splits
    for split in (train, val, test):
        assert {label for _, label in split} == set(range(len(CLASSES)))


@requires_data
def test_getitem_returns_chw_tensor_and_int_label(items):
    dataset = NEUDataset(items[:5])
    tensor, label = dataset[0]

    assert torch.is_tensor(tensor)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 3
    assert isinstance(label, int)
