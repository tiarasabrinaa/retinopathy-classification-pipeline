"""Fine-tune one model (ConvNeXt or DeiT) end to end and log everything to MLflow.

    python train.py --config configs/convnext_tiny.yaml
    python train.py --config configs/deit_small.yaml

Two-phase fine-tuning (freeze backbone -> unfreeze everything) follows the
standard transfer-learning recipe from the official PyTorch tutorial:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Train/eval loop structure (zero_grad -> forward -> backward -> step) follows
the official PyTorch ImageNet example:
https://github.com/pytorch/examples/blob/main/imagenet/main.py

LR schedule for the fine-tune phase uses plain cosine annealing:
https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html

Metrics (macro-F1, confusion matrix) via scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

MLflow run/param/metric logging API:
https://mlflow.org/docs/latest/python_api/mlflow.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    RetinaDataset,
    build_model,
    build_transforms,
    get_device,
    init_mlflow,
    log_confusion_matrix,
    set_backbone_trainable,
    set_seed,
)


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over `loader`. Training when `optimizer` is given, eval otherwise."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, y_true, y_pred = 0.0, [], []

    with torch.set_grad_enabled(is_train):
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            y_pred += outputs.argmax(dim=1).cpu().tolist()
            y_true += targets.cpu().tolist()

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, macro_f1, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg["seed"])
    device = get_device()
    class_names = json.loads(Path(cfg["class_names_json"]).read_text())

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"]).to(device)
    train_tf = build_transforms(model, cfg["image_size"], is_train=True)
    eval_tf = build_transforms(model, cfg["image_size"], is_train=False)

    train_loader = DataLoader(
        RetinaDataset(cfg["train_csv"], ".", train_tf),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        RetinaDataset(cfg["val_csv"], ".", eval_tf),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        RetinaDataset(cfg["test_csv"], ".", eval_tf),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    checkpoint_dir = Path(cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_val_f1 = -1.0

    def maybe_save_checkpoint(val_f1: float, epoch: int) -> None:
        nonlocal best_val_f1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1}, best_path)

    init_mlflow()
    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params(cfg)
        global_epoch = 0

        # --- Phase 1: linear probe (backbone frozen, only the head trains) ---
        set_backbone_trainable(model, trainable=False)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["linear_probe_lr"], weight_decay=cfg["weight_decay"],
        )
        for epoch in range(cfg["linear_probe_epochs"]):
            train_loss, train_f1, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_f1, _, _ = run_epoch(model, val_loader, device, criterion)
            mlflow.log_metrics(
                {"train_loss": train_loss, "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1},
                step=global_epoch,
            )
            print(f"[linear_probe] epoch {epoch}: val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
            maybe_save_checkpoint(val_f1, global_epoch)
            global_epoch += 1

        # --- Phase 2: full fine-tune (every parameter trains) ---
        set_backbone_trainable(model, trainable=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["finetune_lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["finetune_epochs"])
        for epoch in range(cfg["finetune_epochs"]):
            train_loss, train_f1, _, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_f1, y_true, y_pred = run_epoch(model, val_loader, device, criterion)
            scheduler.step()
            mlflow.log_metrics(
                {"train_loss": train_loss, "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1,
                 "lr": scheduler.get_last_lr()[0]},
                step=global_epoch,
            )
            print(f"[finetune] epoch {epoch}: val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
            maybe_save_checkpoint(val_f1, global_epoch)
            global_epoch += 1

        # --- Final test-set evaluation using the best checkpoint ---
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
        test_loss, test_f1, y_true, y_pred = run_epoch(model, test_loader, device, criterion)
        mlflow.log_metrics({"test_loss": test_loss, "test_macro_f1": test_f1})
        log_confusion_matrix(confusion_matrix(y_true, y_pred), class_names, "test")
        print(f"[test] loss={test_loss:.4f} macro_f1={test_f1:.4f} (best checkpoint: {best_path})")


if __name__ == "__main__":
    main()
