"""Verify the downloaded NEU dataset: detect variant, count classes, check image integrity."""
from collections import Counter
from pathlib import Path
from typing import Optional

from PIL import Image, UnidentifiedImageError

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EXPECTED_SIZE = (200, 200)
IMAGE_EXTS = (".bmp", ".jpg", ".jpeg")

CLS_PREFIX_MAP = {
    "cr": "crazing",
    "in": "inclusion",
    "pa": "patches",
    "ps": "pitted_surface",
    "rs": "rolled-in_scale",
    "sc": "scratches",
}
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def classify_cls(stem: str) -> Optional[str]:
    return CLS_PREFIX_MAP.get(stem.split("_")[0].lower())


def classify_det(stem: str) -> Optional[str]:
    lower = stem.lower()
    for cls in CLASSES:
        if lower.startswith(f"{cls}_"):
            return cls
    return None


def gather_images(root: Path) -> list:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def detect_variant(images: list) -> str:
    bmp_cls_hits = sum(1 for p in images if p.suffix.lower() == ".bmp" and classify_cls(p.stem))
    jpg_det_hits = sum(1 for p in images if p.suffix.lower() in (".jpg", ".jpeg") and classify_det(p.stem))
    if bmp_cls_hits == 0 and jpg_det_hits == 0:
        return "unknown"
    return "NEU-CLS" if bmp_cls_hits >= jpg_det_hits else "NEU-DET"


def classify(path: Path, variant: str) -> Optional[str]:
    if variant == "NEU-CLS":
        return classify_cls(path.stem)
    if variant == "NEU-DET":
        return classify_det(path.stem)
    return classify_cls(path.stem) or classify_det(path.stem)


def check_image(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size, img.mode, None
    except (UnidentifiedImageError, OSError) as exc:
        return None, None, str(exc)


def main() -> None:
    if not DATA_DIR.exists():
        print(f"FAIL: {DATA_DIR} does not exist. Run scripts/download_data.py first.")
        raise SystemExit(1)

    images = gather_images(DATA_DIR)
    if not images:
        print(f"FAIL: no .bmp/.jpg images found under {DATA_DIR}")
        raise SystemExit(1)

    variant = detect_variant(images)

    per_class = Counter()
    unclassified = []
    dims = Counter()
    channels = Counter()
    corrupt = []

    for path in images:
        cls = classify(path, variant)
        if cls is None:
            unclassified.append(path)
        else:
            per_class[cls] += 1

        size, mode, err = check_image(path)
        if err:
            corrupt.append((path, err))
            continue
        dims[size] += 1
        channels[mode] += 1

    print("=" * 70)
    print(f"Detected variant: {variant}")
    print(f"Total images found: {len(images)}")
    print()
    print("Per-class counts:")
    for cls in CLASSES:
        print(f"  {cls:>16}: {per_class.get(cls, 0)}")
    if unclassified:
        print(f"  {'UNCLASSIFIED':>16}: {len(unclassified)}")
    print()
    print("Image dimensions found:")
    for size, count in dims.most_common():
        print(f"  {size}: {count}")
    print()
    print("Channel modes found:")
    for mode, count in channels.most_common():
        print(f"  {mode}: {count}")
    print()

    correct_size = bool(dims) and set(dims.keys()) == {EXPECTED_SIZE}

    if corrupt:
        print(f"Corrupt/unreadable files ({len(corrupt)}):")
        for path, err in corrupt[:20]:
            print(f"  {path}: {err}")
        if len(corrupt) > 20:
            print(f"  ... and {len(corrupt) - 20} more")
    else:
        print("No corrupt/unreadable files found.")
    print("=" * 70)

    passed = not corrupt and not unclassified and correct_size and variant != "unknown"
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
