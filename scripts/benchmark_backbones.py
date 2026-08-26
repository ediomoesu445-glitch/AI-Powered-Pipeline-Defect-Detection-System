"""Benchmark timm backbones on the same frozen split with identical
hyperparameters (from configs/resnet18.yaml): test accuracy, macro-F1,
parameter count, model size, CPU inference latency, and training time.

Usage:
    python -m scripts.benchmark_backbones
"""
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from src.augment import get_eval_transform, get_train_transform
from src.dataset import CLASSES, NEUDataset, load_splits
from src.model import build_model
from src.train import evaluate, find_project_root, load_config, train_one_epoch
from src.utils import set_seed

BACKBONES = ["resnet18", "resnet50", "efficientnet_b0", "mobilenetv3_small_100", "vit_tiny_patch16_224"]
N_LATENCY_RUNS = 100
N_WARMUP_RUNS = 5


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


@torch.no_grad()
def measure_cpu_latency_ms(model, img_size: int) -> float:
    cpu_model = model.to("cpu")
    cpu_model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    for _ in range(N_WARMUP_RUNS):
        cpu_model(dummy)

    start = time.perf_counter()
    for _ in range(N_LATENCY_RUNS):
        cpu_model(dummy)
    elapsed = time.perf_counter() - start
    return (elapsed / N_LATENCY_RUNS) * 1000


@torch.no_grad()
def evaluate_test(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return accuracy, macro_f1


def train_backbone(backbone_name, config, train_loader, val_loader, device, use_amp, ckpt_path, epochs):
    model = build_model(backbone_name, num_classes=len(CLASSES), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = OneCycleLR(
        optimizer, max_lr=config["lr"], steps_per_epoch=len(train_loader), epochs=epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0

    # This environment has been killing long background jobs unpredictably, so
    # resume per-epoch within a backbone's training (not just skip a whole
    # already-finished backbone) - same tradeoff as src.train/src.cross_validate:
    # weights only, LR schedule restarts, but no completed epochs are wasted.
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        start_epoch = state["epoch"] + 1
        best_val_acc = state["best_val_acc"]
        best_state = state["model_state_dict"]
        epochs_without_improvement = state["epochs_without_improvement"]
        print(f"[{backbone_name}] Resuming from epoch {start_epoch} (best_val_acc so far={best_val_acc:.4f})")

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        print(
            f"[{backbone_name}] Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
            },
            ckpt_path,
        )

        if epochs_without_improvement >= config["patience"]:
            print(f"[{backbone_name}] Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    ckpt_path.unlink(missing_ok=True)
    return model


def main() -> None:
    project_root = find_project_root()
    config = load_config(project_root / "configs" / "resnet18.yaml")
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_items, val_items, test_items = load_splits(project_root / "configs" / "splits.json")

    train_ds = NEUDataset(train_items, transform=get_train_transform(config["img_size"]))
    val_ds = NEUDataset(val_items, transform=get_eval_transform(config["img_size"]))
    test_ds = NEUDataset(test_items, transform=get_eval_transform(config["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    results_path = project_root / "reports" / "benchmark.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip backbones already benchmarked on a prior run.
    results = []
    done_backbones = set()
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        for _, row in existing_df.iterrows():
            done_backbones.add(row["backbone"])
            results.append(row.to_dict())
        if done_backbones:
            print(f"Resuming: skipping already-benchmarked backbone(s) {sorted(done_backbones)}")

    for backbone_name in BACKBONES:
        if backbone_name in done_backbones:
            continue

        print(f"=== Benchmarking {backbone_name} ===")
        set_seed(config["seed"])
        ckpt_path = project_root / config["out_dir"] / f"benchmark_{backbone_name}_inprogress.pt"

        start_time = time.perf_counter()
        model = train_backbone(
            backbone_name, config, train_loader, val_loader, device, use_amp, ckpt_path, config["epochs"]
        )
        train_time_min = (time.perf_counter() - start_time) / 60

        test_accuracy, macro_f1 = evaluate_test(model, test_loader, device)
        param_count = count_params(model)
        size_mb = model_size_mb(model)
        latency_ms = measure_cpu_latency_ms(model, config["img_size"])

        print(
            f"[{backbone_name}] test_acc={test_accuracy:.4f} macro_f1={macro_f1:.4f} "
            f"params={param_count:,} size={size_mb:.2f}MB latency={latency_ms:.2f}ms "
            f"train_time={train_time_min:.1f}min"
        )

        results.append({
            "backbone": backbone_name,
            "test_accuracy": test_accuracy,
            "macro_f1": macro_f1,
            "param_count": param_count,
            "model_size_mb": size_mb,
            "cpu_latency_ms": latency_ms,
            "train_time_min": train_time_min,
        })

        # Save after every backbone so an interruption doesn't lose completed ones.
        pd.DataFrame(results).to_csv(results_path, index=False)

    results_df = pd.DataFrame(results)

    md_path = project_root / "reports" / "benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Backbone benchmark\n\n")
        f.write(
            "| Backbone | Test Accuracy | Macro F1 | Params | Model Size (MB) | "
            "CPU Latency (ms/img) | Train Time (min) |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for _, row in results_df.iterrows():
            f.write(
                f"| {row['backbone']} | {float(row['test_accuracy']):.4f} | {float(row['macro_f1']):.4f} | "
                f"{int(row['param_count']):,} | {float(row['model_size_mb']):.2f} | "
                f"{float(row['cpu_latency_ms']):.2f} | {float(row['train_time_min']):.1f} |\n"
            )
    print(f"Saved {md_path}")

    fig_dir = project_root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df["cpu_latency_ms"], results_df["test_accuracy"], s=80)
    for _, row in results_df.iterrows():
        ax.annotate(
            row["backbone"], (row["cpu_latency_ms"], row["test_accuracy"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )
    ax.set_xlabel("CPU inference latency (ms/image, batch=1)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs. CPU latency")
    plt.tight_layout()
    scatter_path = fig_dir / "accuracy_vs_latency.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved {scatter_path}")

    print()
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
