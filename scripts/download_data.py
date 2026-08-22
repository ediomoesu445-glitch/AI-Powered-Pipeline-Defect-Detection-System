"""Download the NEU Surface Defect Database (NEU-CLS) into data/raw/."""
import shutil
import subprocess
import sys
from pathlib import Path

KAGGLE_DATASET = "kaustubhdikshit/neu-surface-defect-database"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
KAGGLE_TOKEN_PATH = Path.home() / ".kaggle" / "kaggle.json"

FALLBACK_SOURCES = [
    "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
    "https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection",
    'https://universe.roboflow.com/ (search "NEU-DET", export as folder structure)',
    "https://figshare.com/articles/dataset/NEU-CLS/28903550",
]


def kaggle_cli_available() -> bool:
    return shutil.which("kaggle") is not None


def kaggle_token_configured() -> bool:
    return KAGGLE_TOKEN_PATH.exists()


def print_manual_instructions() -> None:
    print("=" * 70)
    print("Automatic download via the Kaggle CLI is not available.")
    print()
    if not kaggle_cli_available():
        print("The 'kaggle' CLI is not installed. Install it with:")
        print("    pip install kaggle")
        print()
    if not kaggle_token_configured():
        print("No Kaggle API token found at:")
        print(f"    {KAGGLE_TOKEN_PATH}")
        print("To get one:")
        print("  1. Log in to https://www.kaggle.com")
        print("  2. Account settings -> Create New API Token")
        print(f"  3. Save the downloaded kaggle.json to {KAGGLE_TOKEN_PATH}")
        print()
    print("Once the CLI and token are set up, re-run this script:")
    print("    python scripts/download_data.py")
    print()
    print("Or download manually from one of these sources and extract the")
    print(f"images into: {DATA_DIR}")
    for src in FALLBACK_SOURCES:
        print(f"  - {src}")
    print()
    print("Note: only NEU-CLS (.bmp files, e.g. Cr_1.bmp) is the classification")
    print("variant we want. If only NEU-DET (.jpg + PASCAL VOC XML) is available,")
    print("its images can still be used for classification since the class name")
    print("is encoded in the filename.")
    print("=" * 70)


def download_via_kaggle() -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(DATA_DIR),
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return False
    return True


def main() -> None:
    if not kaggle_cli_available() or not kaggle_token_configured():
        print_manual_instructions()
        sys.exit(1)

    print(f"Downloading '{KAGGLE_DATASET}' from Kaggle into {DATA_DIR} ...")
    if not download_via_kaggle():
        print_manual_instructions()
        sys.exit(1)

    print(f"Download complete. Files are in {DATA_DIR}")


if __name__ == "__main__":
    main()
