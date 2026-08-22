"""Dataset loading, label parsing, and train/val/test splitting for the NEU steel
defect dataset.
"""
import json
from pathlib import Path

import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

IMAGE_EXTS = (".bmp", ".jpg", ".jpeg")

_CLS_ABBREVIATIONS = {
    "cr": "crazing",
    "in": "inclusion",
    "pa": "patches",
    "ps": "pitted_surface",
    "rs": "rolled-in_scale",
    "sc": "scratches",
}


def parse_label(filename) -> str:
    """Parse a class name from a NEU-CLS (e.g. 'Cr_1.bmp') or NEU-DET
    (e.g. 'crazing_1.jpg') filename. Case-insensitive. Raises ValueError on an
    unrecognised filename.
    """
    stem = Path(filename).stem.lower()

    for cls in CLASSES:
        if stem.startswith(f"{cls}_"):
            return cls

    abbreviation = stem.split("_")[0]
    if abbreviation in _CLS_ABBREVIATIONS:
        return _CLS_ABBREVIATIONS[abbreviation]

    raise ValueError(
        f"Unrecognised filename, cannot parse a class label from {filename!r}"
    )


def scan_dataset(root) -> list:
    """Walk `root` for NEU-CLS/NEU-DET images and return a list of
    (filepath, label_idx) pairs. Labels are parsed from filenames, not folder
    structure, so this works for either dataset variant.
    """
    root = Path(root)
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return [(p, CLASSES.index(parse_label(p.name))) for p in paths]


def make_splits(items, seed: int = 42):
    """Stratified 70/15/15 train/val/test split of `items` (as produced by
    scan_dataset), split once via sklearn's train_test_split (twice: 70/30,
    then 50/50 of the remaining 30%).

    `items` must be the original, unaugmented file list — splitting after any
    augmentation/duplication would leak the same underlying image across
    splits.
    """
    paths = [p for p, _ in items]
    labels = [label for _, label in items]

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=seed,
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels, test_size=0.50, stratify=rest_labels, random_state=seed,
    )

    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))
    return train, val, test


def save_splits(splits, path) -> None:
    train, val, test = splits
    data = {
        "train": [[str(p), label] for p, label in train],
        "val": [[str(p), label] for p, label in val],
        "test": [[str(p), label] for p, label in test],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_splits(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return tuple(
        [(Path(p), label) for p, label in data[split]] for split in ("train", "val", "test")
    )


class NEUDataset(Dataset):
    """Reads images grayscale, replicates to 3 channels (ImageNet-pretrained
    backbones expect 3 channels), and optionally applies an Albumentations
    transform.
    """

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if not torch.is_tensor(img):
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return img, label
