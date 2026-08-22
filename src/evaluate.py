"""Evaluate a trained checkpoint on the held-out TEST split.

Usage:
    python -m src.evaluate --ckpt models/best.pt
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.augment import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform
from src.dataset import NEUDataset, load_splits
from src.model import build_model


def find_project_root(marker: str = "CLAUDE.md") -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looking for {marker})")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the TEST split")
    parser.add_argument(
        "--ckpt", type=str, default="models/best.pt", help="Path to a checkpoint saved by src.train"
    )
    return parser.parse_args()


def denormalize(tensor) -> np.ndarray:
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = tensor.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return (img.transpose(1, 2, 0) * 255).astype(np.uint8)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob, images = [], [], [], []

    for batch_images, labels in loader:
        batch_images = batch_images.to(device)
        probs = F.softmax(model(batch_images), dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
        images.extend(batch_images.cpu())

    return np.array(y_true), np.array(y_pred), np.array(y_prob), images


def save_confusion_matrix(y_true, y_pred, classes, out_path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_worst_predictions(images, y_true, y_pred, y_prob, classes, out_path, n: int = 12) -> None:
    wrong_indices = np.where(y_true != y_pred)[0]
    if len(wrong_indices) == 0:
        print("No misclassified test examples - skipping worst_predictions.png")
        return

    confidences = y_prob[wrong_indices, y_pred[wrong_indices]]
    top_indices = wrong_indices[np.argsort(-confidences)][:n]

    n_cols = 4
    n_rows = int(np.ceil(len(top_indices) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, idx in zip(axes, top_indices):
        ax.imshow(denormalize(images[idx]))
        true_cls = classes[y_true[idx]]
        pred_cls = classes[y_pred[idx]]
        conf = y_prob[idx, y_pred[idx]]
        ax.set_title(f"true: {true_cls} | pred: {pred_cls} ({conf:.2f})", fontsize=9)
        ax.axis("off")

    for ax in axes[len(top_indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    classes = checkpoint["classes"]
    config = checkpoint["config"]

    model = build_model(config["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    splits_path = project_root / "configs" / "splits.json"
    _train_items, _val_items, test_items = load_splits(splits_path)

    test_ds = NEUDataset(test_items, transform=get_eval_transform(config["img_size"]))
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    y_true, y_pred, y_prob, images = collect_predictions(model, test_loader, device)

    accuracy = (y_true == y_pred).mean()
    report_dict = classification_report(
        y_true, y_pred, target_names=classes, digits=4, output_dict=True, zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, target_names=classes, digits=4, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")

    print(f"Checkpoint: {ckpt_path}")
    print(f"Test set size: {len(test_items)}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print()
    print(report_str)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro one-vs-rest ROC-AUC: {macro_auc:.4f}")

    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cm_path = fig_dir / "confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, classes, cm_path)

    results_path = project_root / "reports" / "results.csv"
    pd.DataFrame(report_dict).transpose().to_csv(results_path)

    worst_path = fig_dir / "worst_predictions.png"
    save_worst_predictions(images, y_true, y_pred, y_prob, classes, worst_path)

    print(f"\nSaved confusion matrix to {cm_path}")
    print(f"Saved per-class metrics to {results_path}")
    print(f"Saved worst predictions to {worst_path}")


if __name__ == "__main__":
    main()
