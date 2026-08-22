"""Config-driven training script for the NEU steel defect classifier.

Usage:
    python -m src.train --config configs/resnet18.yaml
    python -m src.train --config configs/resnet18.yaml --smoke-test
"""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.augment import get_eval_transform, get_train_transform
from src.dataset import CLASSES, NEUDataset, load_splits
from src.model import build_model
from src.utils import set_seed

SMOKE_TEST_TRAIN_SIZE = 80
SMOKE_TEST_VAL_SIZE = 20


def find_project_root(marker: str = "CLAUDE.md") -> Path:
    p = Path.cwd().resolve()
    for parent in [p, *p.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looking for {marker})")


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloaders(project_root: Path, config: dict, smoke_test: bool):
    splits_path = project_root / "configs" / "splits.json"
    train_items, val_items, _test_items = load_splits(splits_path)

    if smoke_test:
        train_items = train_items[:SMOKE_TEST_TRAIN_SIZE]
        val_items = val_items[:SMOKE_TEST_VAL_SIZE]

    train_ds = NEUDataset(train_items, transform=get_train_transform(config["img_size"]))
    val_ds = NEUDataset(val_items, transform=get_eval_transform(config["img_size"]))

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, use_amp) -> float:
    model.train()
    running_loss = 0.0
    total = 0

    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
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
        progress.set_postfix(loss=loss.item())

    return running_loss / total


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="val", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return running_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train the NEU steel defect classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Run 1 epoch on ~{SMOKE_TEST_TRAIN_SIZE + SMOKE_TEST_VAL_SIZE} images "
        "to catch shape/normalization bugs in seconds, without downloading "
        "pretrained weights or writing to models/best.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    config = load_config(args.config)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    epochs = 1 if args.smoke_test else config["epochs"]

    print(f"Device: {device} (mixed precision: {use_amp})")

    train_loader, val_loader = build_dataloaders(project_root, config, args.smoke_test)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Smoke-test runs are for catching shape/normalization bugs quickly, not for
    # validating pretrained weights, so skip the (network-dependent) download.
    model = build_model(
        config["backbone"], num_classes=len(CLASSES), pretrained=not args.smoke_test
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = OneCycleLR(
        optimizer, max_lr=config["lr"], steps_per_epoch=len(train_loader), epochs=epochs
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = f"{'smoke_' if args.smoke_test else ''}{config['backbone']}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=str(project_root / "runs" / run_name))

    best_val_acc = -1.0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler, use_amp
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.2e}"
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "config": config,
                "classes": CLASSES,
                "val_accuracy": best_val_acc,
                "epoch": epoch,
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1} (patience={config['patience']})")
                break

    writer.close()

    out_dir = project_root / config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = "smoke_test.pt" if args.smoke_test else "best.pt"
    ckpt_path = out_dir / ckpt_name
    torch.save(best_state, ckpt_path)
    print(f"Saved checkpoint (val_accuracy={best_state['val_accuracy']:.4f}) to {ckpt_path}")


if __name__ == "__main__":
    main()
