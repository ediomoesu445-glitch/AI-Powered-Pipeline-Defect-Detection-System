"""Visualize Grad-CAM overlays for the trained classifier.

Loads models/best.pt and produces:
  - reports/figures/gradcam_by_class.png       (2 correct examples x 6 classes)
  - reports/figures/gradcam_method_comparison.png (GradCAM/++/ScoreCAM/EigenCAM x 3 images)
  - reports/figures/gradcam_failures.png       (4 misclassified examples)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.augment import get_eval_transform
from src.dataset import load_splits
from src.gradcam_utils import gradcam_overlay
from src.model import build_model, get_gradcam_target_layers

METHODS = ["gradcam", "gradcam++", "scorecam", "eigencam"]
METHOD_TITLES = ["GradCAM", "GradCAM++", "ScoreCAM", "EigenCAM"]


def find_project_root(marker: str = "CLAUDE.md") -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looking for {marker})")


def load_model_and_test_items(project_root: Path):
    ckpt_path = project_root / "models" / "best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    classes = checkpoint["classes"]
    config = checkpoint["config"]

    model = build_model(config["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    splits_path = project_root / "configs" / "splits.json"
    _train_items, _val_items, test_items = load_splits(splits_path)

    return model, config, classes, test_items, device


def get_rgb_and_input(path, img_size: int, device):
    """Return (rgb_float [H,W,3] in [0,1], input_tensor [1,3,H,W] on device)."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    rgb_uint8 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    resized_for_display = cv2.resize(rgb_uint8, (img_size, img_size))
    rgb_float = resized_for_display.astype(np.float32) / 255.0

    input_tensor = get_eval_transform(img_size)(image=rgb_uint8)["image"].unsqueeze(0).to(device)
    return rgb_float, input_tensor


@torch.no_grad()
def predict(model, input_tensor):
    probs = F.softmax(model(input_tensor), dim=1)
    conf, pred = probs.max(dim=1)
    return pred.item(), conf.item()


def find_correct_examples(model, test_items, classes, img_size, device, per_class: int = 2):
    found = {idx: [] for idx in range(len(classes))}
    needed = len(classes) * per_class

    for path, label in test_items:
        if len(found[label]) >= per_class:
            continue
        rgb_float, input_tensor = get_rgb_and_input(path, img_size, device)
        pred, conf = predict(model, input_tensor)
        if pred == label:
            found[label].append((path, label, conf, rgb_float, input_tensor))
        if sum(len(v) for v in found.values()) >= needed:
            break

    return found


def find_misclassified_examples(model, test_items, img_size, device, n: int = 4):
    found = []
    for path, label in test_items:
        rgb_float, input_tensor = get_rgb_and_input(path, img_size, device)
        pred, conf = predict(model, input_tensor)
        if pred != label:
            found.append((path, label, pred, conf, rgb_float, input_tensor))
        if len(found) >= n:
            break
    return found


def build_gradcam_by_class(model, target_layers, examples_by_class, classes, out_path):
    n_rows = len(classes)
    n_cols = 4  # orig1, cam1, orig2, cam2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for row, cls_idx in enumerate(range(len(classes))):
        examples = examples_by_class[cls_idx]
        for slot in range(2):
            ax_orig, ax_cam = axes[row, slot * 2], axes[row, slot * 2 + 1]

            if slot >= len(examples):
                ax_orig.axis("off")
                ax_cam.axis("off")
                continue

            path, label, conf, rgb_float, input_tensor = examples[slot]
            overlay = gradcam_overlay(
                model, input_tensor, rgb_float, target_layers, class_idx=label, method="gradcam"
            )

            ax_orig.imshow(rgb_float)
            ax_orig.set_title(f"{classes[label]}\n{Path(path).name}", fontsize=8)
            ax_orig.axis("off")

            ax_cam.imshow(overlay)
            ax_cam.set_title(f"Grad-CAM (conf {conf:.2f})", fontsize=8)
            ax_cam.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_method_comparison(model, target_layers, examples, classes, out_path):
    n_rows = len(examples)
    n_cols = 1 + len(METHODS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)

    for row, (path, label, conf, rgb_float, input_tensor) in enumerate(examples):
        axes[row, 0].imshow(rgb_float)
        axes[row, 0].set_title(f"{classes[label]}\n{Path(path).name}", fontsize=9)
        axes[row, 0].axis("off")

        for col, (method, title) in enumerate(zip(METHODS, METHOD_TITLES), start=1):
            overlay = gradcam_overlay(
                model, input_tensor, rgb_float, target_layers, class_idx=label, method=method
            )
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(title, fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_gradcam_failures(model, target_layers, misclassified, classes, out_path):
    n_rows = len(misclassified)
    fig, axes = plt.subplots(n_rows, 2, figsize=(8, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for row, (path, label, pred, conf, rgb_float, input_tensor) in enumerate(misclassified):
        overlay = gradcam_overlay(
            model, input_tensor, rgb_float, target_layers, class_idx=pred, method="gradcam"
        )

        axes[row, 0].imshow(rgb_float)
        axes[row, 0].set_title(f"true: {classes[label]}\n{Path(path).name}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"pred: {classes[pred]} ({conf:.2f})\nGrad-CAM for predicted class", fontsize=9)
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = find_project_root()
    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model, config, classes, test_items, device = load_model_and_test_items(project_root)
    target_layers = get_gradcam_target_layers(model, config["backbone"])
    img_size = config["img_size"]

    print("Finding 2 correctly-classified examples per class...")
    examples_by_class = find_correct_examples(model, test_items, classes, img_size, device, per_class=2)
    for idx, cls in enumerate(classes):
        print(f"  {cls}: found {len(examples_by_class[idx])}/2")

    out_path = fig_dir / "gradcam_by_class.png"
    build_gradcam_by_class(model, target_layers, examples_by_class, classes, out_path)
    print(f"Saved {out_path}")

    print("Building method comparison (GradCAM / GradCAM++ / ScoreCAM / EigenCAM)...")
    comparison_examples = [examples_by_class[idx][0] for idx in (0, 1, 2) if examples_by_class[idx]]
    out_path = fig_dir / "gradcam_method_comparison.png"
    build_method_comparison(model, target_layers, comparison_examples, classes, out_path)
    print(f"Saved {out_path}")

    print("Finding 4 misclassified examples...")
    misclassified = find_misclassified_examples(model, test_items, img_size, device, n=4)
    print(f"  found {len(misclassified)}/4")
    if misclassified:
        out_path = fig_dir / "gradcam_failures.png"
        build_gradcam_failures(model, target_layers, misclassified, classes, out_path)
        print(f"Saved {out_path}")
    else:
        print("No misclassified examples found - skipping gradcam_failures.png")


if __name__ == "__main__":
    main()
