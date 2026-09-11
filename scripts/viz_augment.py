"""Save a grid of 16 augmented versions of one sample image."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.augment import IMAGENET_MEAN, IMAGENET_STD, get_train_transform
from src.dataset import scan_dataset
from src.utils import set_seed


def find_project_root(marker="CLAUDE.md") -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looking for {marker})")


def denormalize(tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


def main() -> None:
    set_seed(42)

    project_root = find_project_root()
    data_root = project_root / "data" / "raw"
    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    items = scan_dataset(data_root)
    sample_path, _ = items[0]

    gray = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    transform = get_train_transform()

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for ax in axes.ravel():
        augmented = transform(image=rgb)["image"]
        ax.imshow(denormalize(augmented))
        ax.axis("off")
    fig.suptitle(f"Train augmentations: {sample_path.name}")
    plt.tight_layout()

    out_path = fig_dir / "augmentation_examples.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
