"""Robustness testing under simulated field-imaging corruptions.

Premise: NEU is clean lab imagery, but real pipeline visual inspection happens
via drone or CCTV in poor conditions (motion, low light, weather, compression
artifacts). This measures how much TEST-set accuracy degrades under 7
corruption types at 3 severity levels, for the primary trained model
(models/best.pt) and - once available - the top 3 backbones from
reports/benchmark.csv, to find which backbone is most robust
out-of-distribution, not just most accurate in-distribution on clean images.

Usage:
    python -m scripts.robustness_test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from src.augment import get_eval_transform
from src.dataset import load_splits
from src.model import build_model
from src.train import find_project_root, load_config
from src.utils import set_seed

SEVERITIES = ["mild", "moderate", "severe"]
SEVERITY_LEVEL = {"mild": 1, "moderate": 2, "severe": 3}


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    ksize = {1: 7, 2: 15, 3: 25}[severity]
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, kernel)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    std = {1: 10, 2: 25, 3: 45}[severity]
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def low_light(img: np.ndarray, severity: int) -> np.ndarray:
    factor = {1: 0.7, 2: 0.5, 3: 0.3}[severity]
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def overexposure(img: np.ndarray, severity: int) -> np.ndarray:
    add = {1: 50, 2: 100, 3: 160}[severity]
    return np.clip(img.astype(np.float32) + add, 0, 255).astype(np.uint8)


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    quality = {1: 40, 2: 20, 3: 8}[severity]
    _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def rain_streak(img: np.ndarray, severity: int) -> np.ndarray:
    n_streaks, alpha = {1: (30, 0.25), 2: (70, 0.40), 3: (130, 0.55)}[severity]
    overlay = img.copy()
    h, w = img.shape[:2]
    for _ in range(n_streaks):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(10, 25)
        angle = np.random.uniform(70, 110)
        dx = int(length * np.cos(np.radians(angle)))
        dy = int(length * np.sin(np.radians(angle)))
        cv2.line(overlay, (x1, y1), (x1 + dx, y1 + dy), (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    ksize = {1: 5, 2: 11, 3: 19}[severity]
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


CORRUPTIONS = {
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "low_light": low_light,
    "overexposure": overexposure,
    "jpeg_compression": jpeg_compression,
    "rain_streak": rain_streak,
    "defocus_blur": defocus_blur,
}


class CorruptedNEUDataset(Dataset):
    def __init__(self, items, img_size, corruption_fn=None, severity=None):
        self.items = items
        self.corruption_fn = corruption_fn
        self.severity = severity
        self.transform = get_eval_transform(img_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        if self.corruption_fn is not None:
            rgb = self.corruption_fn(rgb, SEVERITY_LEVEL[self.severity])
        tensor = self.transform(image=rgb)["image"]
        return tensor, label


@torch.no_grad()
def evaluate_dataset(model, dataset, device, batch_size: int = 32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return accuracy, macro_f1


def load_checkpoint_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    config = checkpoint["config"]
    model = build_model(config["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config


def run_robustness_grid(model, test_items, img_size, device):
    rows = []

    clean_ds = CorruptedNEUDataset(test_items, img_size)
    clean_acc, clean_f1 = evaluate_dataset(model, clean_ds, device)
    rows.append({"corruption": "clean", "severity": "none", "accuracy": clean_acc, "macro_f1": clean_f1})
    print(f"  clean baseline: accuracy={clean_acc:.4f} macro_f1={clean_f1:.4f}")

    for corruption_name, corruption_fn in CORRUPTIONS.items():
        for severity in SEVERITIES:
            ds = CorruptedNEUDataset(test_items, img_size, corruption_fn, severity)
            acc, f1 = evaluate_dataset(model, ds, device)
            rows.append({"corruption": corruption_name, "severity": severity, "accuracy": acc, "macro_f1": f1})
            print(f"  {corruption_name} ({severity}): accuracy={acc:.4f} macro_f1={f1:.4f}")

    return rows, clean_acc


def save_examples_figure(test_items, out_path, severity: str = "severe"):
    path, _ = test_items[0]
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    names = ["original"] + list(CORRUPTIONS.keys())
    images = [rgb] + [fn(rgb.copy(), SEVERITY_LEVEL[severity]) for fn in CORRUPTIONS.values()]

    n_cols = 4
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.ravel()
    for ax, name, img in zip(axes, names, images):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.suptitle(f"Corruption examples ({severity} severity): {Path(path).name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_heatmap(df, title, out_path):
    pivot = df[df["corruption"] != "clean"].pivot(index="corruption", columns="severity", values="accuracy")
    pivot = pivot[SEVERITIES]
    plt.figure(figsize=(7, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "accuracy"})
    plt.title(f"Robustness under corruption - {title}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = find_project_root()
    config = load_config(project_root / "configs" / "resnet18.yaml")
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _train_items, _val_items, test_items = load_splits(project_root / "configs" / "splits.json")

    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_path = project_root / "reports" / "robustness.csv"

    # Which checkpoints to test: the primary trained model, plus the top 3
    # backbones from reports/benchmark.csv if that's available yet (it's
    # produced by a separate, much longer-running script).
    checkpoints_to_test = [("resnet18_primary", project_root / "models" / "best.pt")]

    benchmark_path = project_root / "reports" / "benchmark.csv"
    if benchmark_path.exists():
        benchmark_df = pd.read_csv(benchmark_path)
        top3 = benchmark_df.sort_values("test_accuracy", ascending=False).head(3)
        for _, row in top3.iterrows():
            name = row["backbone"]
            ckpt_path = project_root / "models" / f"benchmark_{name}.pt"
            if ckpt_path.exists():
                checkpoints_to_test.append((name, ckpt_path))
    else:
        print(
            "reports/benchmark.csv not found yet - testing only the primary model. "
            "Re-run this script after scripts/benchmark_backbones.py has produced "
            "results to also test the top 3 backbones."
        )

    # Resume: skip backbones whose full grid is already in robustness.csv.
    all_results = []
    done_backbones = set()
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        all_results = existing_df.to_dict("records")
        done_backbones = set(existing_df["backbone"].unique())
        if done_backbones:
            print(f"Resuming: skipping already-tested backbone(s) {sorted(done_backbones)}")

    summary = []
    for name, ckpt_path in checkpoints_to_test:
        if name in done_backbones:
            rows = [r for r in all_results if r["backbone"] == name]
            clean_rows = [r for r in rows if r["corruption"] == "clean"]
            corrupted = [r["accuracy"] for r in rows if r["corruption"] != "clean"]
            if clean_rows and corrupted:
                mean_drop = clean_rows[0]["accuracy"] - (sum(corrupted) / len(corrupted))
                summary.append((name, mean_drop))
            continue

        if not ckpt_path.exists():
            print(f"Checkpoint not found for {name}: {ckpt_path} - skipping")
            continue

        print(f"=== Robustness testing {name} ({ckpt_path.name}) ===")
        model, model_config = load_checkpoint_model(ckpt_path, device)
        rows, clean_acc = run_robustness_grid(model, test_items, model_config["img_size"], device)

        for row in rows:
            row["backbone"] = name
        all_results.extend(rows)
        pd.DataFrame(all_results).to_csv(results_path, index=False)

        corrupted_accs = [r["accuracy"] for r in rows if r["corruption"] != "clean"]
        mean_drop = clean_acc - (sum(corrupted_accs) / len(corrupted_accs))
        summary.append((name, mean_drop))
        print(f"{name}: mean corruption accuracy drop = {mean_drop:.4f}")

        backbone_df = pd.DataFrame(rows)
        save_heatmap(backbone_df, name, fig_dir / f"robustness_heatmap_{name}.png")
        if name == checkpoints_to_test[0][0]:
            save_heatmap(backbone_df, name, fig_dir / "robustness_heatmap.png")

    examples_path = fig_dir / "robustness_examples.png"
    if not examples_path.exists():
        save_examples_figure(test_items, examples_path)
        print(f"Saved {examples_path}")

    print()
    print("=== Summary: mean corruption accuracy drop per backbone (lower = more robust) ===")
    for name, drop in sorted(summary, key=lambda x: x[1]):
        print(f"  {name}: {drop:.4f}")

    if len(summary) > 1:
        most_robust = min(summary, key=lambda x: x[1])
        print(f"\nMost robust (lowest mean accuracy drop): {most_robust[0]} ({most_robust[1]:.4f})")


if __name__ == "__main__":
    main()
