"""Benchmark timm backbones on the same frozen split with identical
hyperparameters (from configs/resnet18.yaml): test accuracy, macro-F1,
parameter count, model size, CPU inference latency, and training time.

Usage:
    python -m scripts.benchmark_backbones
"""
import io
import os
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
from src.train import evaluate, find_project_root, load_config
from src.utils import set_seed

# Ordered lightest-first. Each backbone trains independently from pretrained
# weights on the same frozen split with identical hyperparameters, so the order
# does not affect any reported number - but on a memory-constrained machine that
# kills jobs within a minute, the lighter models are the ones that can actually
# finish, and efficientnet_b0 is the backbone the cited robustness claim is about.
BACKBONES = ["resnet18", "mobilenetv3_small_100", "efficientnet_b0", "resnet50", "vit_tiny_patch16_224"]
N_LATENCY_RUNS = 100
N_WARMUP_RUNS = 5
# Epoch-level checkpointing is not enough here: the heavier backbones need
# 15-20 min per epoch while this machine kills background jobs in well under
# that, so a whole epoch's work was being discarded on every restart and the
# run made no progress at all. Same sub-epoch fix as src.cross_validate.
BATCH_CHECKPOINT_INTERVAL = 5


def save_checkpoint_resilient(state: dict, path) -> None:
    """Checkpoint atomically, with retry.

    Two hazards this guards against, both observed on this machine:

    - The project lives in a OneDrive-synced folder, which transiently locks a
      just-written file while uploading it (Windows sharing violation,
      WinError 32). Unretried, that surfaces as a RuntimeError from the
      zipfile writer and kills the whole run.
    - Jobs here are killed frequently and without warning. Writing straight to
      the checkpoint path means a kill landing mid-write leaves a truncated,
      unloadable file, which would then break every subsequent resume. So
      write to a temporary file first and os.replace() it into place, which is
      atomic on the same volume: the real checkpoint is either the previous
      good one or the new complete one, never a partial write.
    """
    tmp_path = Path(str(path) + ".tmp")
    last_err = None
    for attempt in range(6):
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)
            return
        except (OSError, RuntimeError) as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    tmp_path.unlink(missing_ok=True)
    raise last_err


def train_one_epoch_resumable(
    model, loader, optimizer, scheduler, criterion, device, scaler, use_amp, start_batch, checkpoint_fn
):
    """Like src.train.train_one_epoch, but skips the first `start_batch`
    batches (mid-epoch resume) and checkpoints every
    BATCH_CHECKPOINT_INTERVAL batches.
    """
    model.train()
    running_loss = 0.0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if batch_idx < start_batch:
            continue

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        if (batch_idx + 1) % BATCH_CHECKPOINT_INTERVAL == 0:
            checkpoint_fn(batch_idx)

    return running_loss / total if total else 0.0


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
    start_batch = 0
    best_val_acc = -1.0
    best_state = None
    epochs_without_improvement = 0
    steps_per_epoch = len(train_loader)

    # This environment kills long background jobs well inside a single epoch,
    # so resume mid-epoch (weights + optimizer + scheduler + scaler state),
    # not just from the last fully-completed epoch.
    state = None
    if ckpt_path.exists():
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:
            # A checkpoint truncated by a kill mid-write (possible before saves
            # became atomic) must not crash every subsequent restart. Discard it
            # and retrain this backbone from pretrained weights.
            print(f"[{backbone_name}] Ignoring unreadable checkpoint {ckpt_path.name}: {e}")
            ckpt_path.unlink(missing_ok=True)

    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        best_val_acc = state["best_val_acc"]
        best_state = state["model_state_dict"]
        epochs_without_improvement = state["epochs_without_improvement"]

        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler.load_state_dict(state["scaler_state_dict"])
            start_epoch = state["epoch"]
            start_batch = state["batch"] + 1
            if start_batch >= steps_per_epoch:
                start_epoch += 1
                start_batch = 0
        else:
            # Older epoch-only checkpoint format: resume at the next epoch
            # rather than discarding this backbone's progress entirely.
            start_epoch = state["epoch"] + 1
            start_batch = 0

        print(
            f"[{backbone_name}] Resuming from epoch {start_epoch + 1}, batch {start_batch} "
            f"(best_val_acc so far={best_val_acc:.4f})"
        )

    def save_progress(epoch: int, batch: int) -> None:
        save_checkpoint_resilient(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "batch": batch,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
            },
            ckpt_path,
        )

    for epoch in range(start_epoch, epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        train_loss = train_one_epoch_resumable(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, use_amp,
            epoch_start_batch, lambda b, e=epoch: save_progress(e, b),
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

        # Mark this epoch fully complete so a resume starts the next one.
        save_progress(epoch, steps_per_epoch - 1)

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

        # Persist the trained weights (not just the metrics) - needed to later
        # run e.g. robustness testing against these same trained backbones.
        final_ckpt_path = project_root / config["out_dir"] / f"benchmark_{backbone_name}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {**config, "backbone": backbone_name},
                "classes": CLASSES,
                "test_accuracy": test_accuracy,
            },
            final_ckpt_path,
        )
        print(f"Saved trained weights to {final_ckpt_path}")

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
