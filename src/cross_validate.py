"""5-fold stratified cross-validation over train+val (TEST split untouched).

Rationale: with only 1800 images total, a single train/val/test split can look
deceptively perfect purely by luck of which images ended up where (our own
single-split run happened to hit 100% val accuracy and 99.63% test accuracy).
Repeating training across 5 stratified folds of the combined train+val pool
and reporting the mean +/- std of accuracy and macro-F1 gives a much more
honest picture of how much that reported performance depends on the specific
split - without ever touching the held-out TEST split, which stays reserved
for the one final reported number.

Usage:
    python -m src.cross_validate
"""
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import pandas as pd

from src.augment import get_eval_transform, get_train_transform
from src.dataset import CLASSES, NEUDataset, load_splits
from src.model import build_model
from src.train import find_project_root, load_config
from src.utils import set_seed

N_FOLDS = 5
# This environment has been killing background jobs well before a single
# epoch (39 batches) can finish, repeatedly losing all of that epoch's
# progress on resume. Checkpoint every few batches (with full optimizer/
# scheduler state, not just model weights) so a kill only costs a handful
# of batches instead of a whole epoch.
BATCH_CHECKPOINT_INTERVAL = 5


def train_one_epoch_resumable(
    model, loader, optimizer, scheduler, criterion, device, scaler, use_amp, start_batch, checkpoint_fn
):
    """Like src.train.train_one_epoch, but skips the first `start_batch`
    batches (to continue a mid-epoch resume) and calls
    checkpoint_fn(batch_idx) every BATCH_CHECKPOINT_INTERVAL batches.
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


@torch.no_grad()
def evaluate_fold(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    val_loss = running_loss / len(y_true)
    val_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    val_f1 = f1_score(y_true, y_pred, average="macro")
    return val_loss, val_acc, val_f1


def run_fold(fold_train_items, fold_val_items, config, device, use_amp, epochs, fold_idx, fold_ckpt_path):
    train_ds = NEUDataset(fold_train_items, transform=get_train_transform(config["img_size"]))
    val_ds = NEUDataset(fold_val_items, transform=get_eval_transform(config["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    # Fresh pretrained model per fold - never carry fine-tuned weights across
    # folds, or a fold's validation images could be influenced by having been
    # seen (via the model's weights) during a different fold's training.
    model = build_model(config["backbone"], num_classes=len(CLASSES), pretrained=True)
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
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    steps_per_epoch = len(train_loader)

    # A fold can take well over an hour, longer than this environment has
    # reliably run background jobs without being killed, so resume mid-epoch
    # (weights + optimizer + scheduler state, not just model weights) rather
    # than only being able to resume from the last fully-completed epoch.
    if fold_ckpt_path.exists():
        state = torch.load(fold_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        best_val_acc = state["best_val_acc"]
        best_val_f1 = state["best_val_f1"]
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
            # Migrating from the older epoch-only checkpoint format (no
            # optimizer/scheduler state): resume at the start of the next
            # epoch rather than losing this fold's progress entirely.
            start_epoch = state["epoch"] + 1
            start_batch = 0

        print(
            f"[fold {fold_idx}] Resuming from epoch {start_epoch + 1}, batch {start_batch} "
            f"(best_val_acc so far={best_val_acc:.4f})"
        )

    def save_checkpoint(epoch: int, batch: int) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "batch": batch,
                "best_val_acc": best_val_acc,
                "best_val_f1": best_val_f1,
                "epochs_without_improvement": epochs_without_improvement,
            },
            fold_ckpt_path,
        )

    for epoch in range(start_epoch, epochs):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        train_loss = train_one_epoch_resumable(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, use_amp,
            epoch_start_batch, lambda b, e=epoch: save_checkpoint(e, b),
        )
        val_loss, val_acc, val_f1 = evaluate_fold(model, val_loader, criterion, device, use_amp)

        print(
            f"[fold {fold_idx}] Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Mark this epoch fully complete so a resume starts the next epoch.
        save_checkpoint(epoch, steps_per_epoch - 1)

        if epochs_without_improvement >= config["patience"]:
            print(f"[fold {fold_idx}] Early stopping at epoch {epoch + 1}")
            break

    fold_ckpt_path.unlink(missing_ok=True)
    return best_val_acc, best_val_f1


def main() -> None:
    project_root = find_project_root()
    config = load_config(project_root / "configs" / "resnet18.yaml")
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_items, val_items, test_items = load_splits(project_root / "configs" / "splits.json")
    combined = train_items + val_items
    labels = [label for _, label in combined]
    print(f"Combined train+val pool: {len(combined)} images (TEST split of {len(test_items)} left untouched)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=config["seed"])

    results_path = project_root / "reports" / "cv_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: this can be a multi-hour run, so if it was interrupted
    # and restarted, skip folds whose results are already on disk instead of
    # redoing them.
    results = []
    completed_folds = set()
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        existing_df = existing_df[existing_df["fold"].apply(lambda x: str(x).isdigit())]
        for _, row in existing_df.iterrows():
            completed_folds.add(int(row["fold"]))
            results.append(
                {"fold": int(row["fold"]), "val_accuracy": row["val_accuracy"], "macro_f1": row["macro_f1"]}
            )
        if completed_folds:
            print(f"Resuming: skipping already-completed fold(s) {sorted(completed_folds)}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(combined, labels), start=1):
        if fold_idx in completed_folds:
            continue

        fold_train_items = [combined[i] for i in train_idx]
        fold_val_items = [combined[i] for i in val_idx]

        print(f"=== Fold {fold_idx}/{N_FOLDS}: train={len(fold_train_items)} val={len(fold_val_items)} ===")
        fold_ckpt_path = project_root / config["out_dir"] / f"cv_fold{fold_idx}_inprogress.pt"
        val_acc, val_f1 = run_fold(
            fold_train_items, fold_val_items, config, device, use_amp, config["epochs"], fold_idx, fold_ckpt_path
        )
        results.append({"fold": fold_idx, "val_accuracy": val_acc, "macro_f1": val_f1})

        # Save after every fold so an interruption doesn't lose completed folds.
        pd.DataFrame(results).to_csv(results_path, index=False)
        print(f"Fold {fold_idx} done: val_accuracy={val_acc:.4f} macro_f1={val_f1:.4f}")

    results_df = pd.DataFrame(results)
    acc_mean, acc_std = results_df["val_accuracy"].mean(), results_df["val_accuracy"].std()
    f1_mean, f1_std = results_df["macro_f1"].mean(), results_df["macro_f1"].std()

    print()
    print(f"Accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
    print(f"Macro F1: {f1_mean:.4f} +/- {f1_std:.4f}")

    summary_row = pd.DataFrame([{
        "fold": "mean+-std",
        "val_accuracy": f"{acc_mean:.4f}+-{acc_std:.4f}",
        "macro_f1": f"{f1_mean:.4f}+-{f1_std:.4f}",
    }])
    pd.concat([results_df, summary_row], ignore_index=True).to_csv(results_path, index=False)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
