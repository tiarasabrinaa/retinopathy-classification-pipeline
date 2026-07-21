"""Shared building blocks used by both make_splits.py and train.py.

Kept as one flat module on purpose — everything here is a handful of lines,
so splitting it into a data/ + models/ + utils/ package would just add
navigation overhead for a 2-model research pipeline.

Sources:
- set_seed: https://pytorch.org/docs/stable/notes/randomness.html
- get_device (CUDA/MPS/CPU): https://pytorch.org/docs/stable/notes/mps.html
- build_model via timm.create_model:
  https://huggingface.co/docs/timm/quickstart#load-a-pretrained-model
- build_transforms via timm.data.resolve_data_config (matches whatever
  normalization the pretrained checkpoint expects, instead of hardcoding
  ImageNet mean/std):
  https://huggingface.co/docs/timm/quickstart#image-augmentation
- MLflow -> Databricks Community Edition tracking (tracking_uri="databricks",
  auth via DATABRICKS_HOST/DATABRICKS_TOKEN env vars):
  https://docs.databricks.com/en/mlflow/access-hosted-tracking-server.html
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import mlflow
except ImportError:
    mlflow = None  # only required when actually running train.py, not for make_splits.py


# --- reproducibility / device -------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- data ----------------------------------------------------------------------

class RetinaDataset(Dataset):
    """Reads a manifest CSV produced by make_splits.py (filepath, label, label_idx)."""

    def __init__(self, csv_path: str | Path, project_root: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.project_root = Path(project_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(self.project_root / row["filepath"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(int(row["label_idx"]), dtype=torch.long)


def build_transforms(model: torch.nn.Module, image_size: int, is_train: bool):
    """`model` is used only to resolve the mean/std it was pretrained with."""
    data_config = timm.data.resolve_data_config({}, model=model)
    mean, std = data_config["mean"], data_config["std"]

    if is_train:
        # Mild augmentation on purpose: fundus images encode diagnostic signal in
        # color/texture (hemorrhages, exudates), so we avoid aggressive color jitter.
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size / 0.9)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# --- model -----------------------------------------------------------------------

def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """False = linear-probe (only classifier head trains), True = full fine-tune.
    Head is located via timm's model.get_classifier(), implemented by every
    timm model regardless of architecture (CNN or ViT-family)."""
    head_params = set(model.get_classifier().parameters())
    for param in model.parameters():
        param.requires_grad = trainable or (param in head_params)


# --- mlflow ------------------------------------------------------------------------

def init_mlflow(experiment_name: str | None = None) -> None:
    load_dotenv()  # reads .env into os.environ if present, no-op otherwise
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment_name or os.environ["MLFLOW_EXPERIMENT_NAME"])


def log_confusion_matrix(cm: np.ndarray, class_names: list[str], step_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    mlflow.log_figure(fig, f"{step_name}_confusion_matrix.png")
    plt.close(fig)
