"""Runs Grad-CAM (ConvNeXt) or Attention Rollout (DeiT) plus deletion/insertion
AUC faithfulness scoring for one already-trained model, over a random sample
of the test set.

    python run_explainability.py --config configs/convnext_tiny.yaml
    python run_explainability.py --config configs/deit_small.yaml

Requires `checkpoint_dir/best.pt` from a prior `train.py` run. Saves one
heatmap-overlay PNG per sampled image to `checkpoint_dir/explanations/`
(kept off MLflow -- see the storage-quota note in README.md, same reason
checkpoints stay local). Only the mean deletion/insertion AUC (overall and
per class) is logged to MLflow, under a `<run_name>_explain` run.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import timm
import torch
import yaml

from common import RetinaDataset, build_model, build_transforms, get_device, init_mlflow, set_seed
from explain import PATCH_GRID, attention_rollout, blur_baseline, deletion_auc, grad_cam, insertion_auc

XAI_METHODS = {"grad_cam": grad_cam, "attention_rollout": attention_rollout}


def _denormalize(image: torch.Tensor, mean, std) -> np.ndarray:
    """(1, 3, H, W) normalized tensor -> (H, W, 3) uint8 array for display."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    pixels = (image[0].detach().cpu() * std + mean).clamp(0, 1)
    return (pixels.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_overlay(path: Path, image: torch.Tensor, patch_map: np.ndarray, mean, std, title: str) -> None:
    rgb = _denormalize(image, mean, std)
    heatmap = np.kron(patch_map, np.ones((rgb.shape[0] // PATCH_GRID, rgb.shape[1] // PATCH_GRID)))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(rgb)
    axes[0].set_title("input")
    axes[1].imshow(rgb)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=40, help="images sampled from the test set")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    set_seed(cfg["seed"])
    device = get_device()
    class_names = json.loads(Path(cfg["class_names_json"]).read_text())
    explain_fn = XAI_METHODS[cfg["xai_method"]]

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"]).to(device)
    checkpoint = torch.load(Path(cfg["checkpoint_dir"]) / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eval_tf = build_transforms(model, cfg["image_size"], is_train=False)
    data_config = timm.data.resolve_data_config({}, model=model)
    mean, std = data_config["mean"], data_config["std"]

    dataset = RetinaDataset(cfg["test_csv"], ".", eval_tf)
    sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    out_dir = Path(cfg["checkpoint_dir"]) / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    deletion_scores, insertion_scores = [], []
    per_class_deletion = {name: [] for name in class_names}
    per_class_insertion = {name: [] for name in class_names}

    for i in sample_indices:
        image, true_label = dataset[i]
        image = image.unsqueeze(0).to(device)
        baseline = blur_baseline(image)

        patch_map, target_class = explain_fn(model, image)
        d_auc = deletion_auc(model, image, patch_map, target_class, baseline)
        i_auc = insertion_auc(model, image, patch_map, target_class, baseline)

        class_name = class_names[target_class]
        deletion_scores.append(d_auc)
        insertion_scores.append(i_auc)
        per_class_deletion[class_name].append(d_auc)
        per_class_insertion[class_name].append(i_auc)

        title = f"{cfg['xai_method']} (pred={class_name}, del={d_auc:.3f}, ins={i_auc:.3f})"
        save_overlay(out_dir / f"{i:05d}_{class_name}.png", image, patch_map, mean, std, title)

    summary = {
        "explain_deletion_auc_mean": float(np.mean(deletion_scores)),
        "explain_insertion_auc_mean": float(np.mean(insertion_scores)),
    }
    for name in class_names:
        if per_class_deletion[name]:
            summary[f"explain_deletion_auc_{name}"] = float(np.mean(per_class_deletion[name]))
            summary[f"explain_insertion_auc_{name}"] = float(np.mean(per_class_insertion[name]))

    init_mlflow()
    with mlflow.start_run(run_name=f"{cfg['run_name']}_explain"):
        mlflow.log_params({"xai_method": cfg["xai_method"], "num_samples": len(sample_indices)})
        mlflow.log_metrics(summary)

    print(f"[{cfg['xai_method']}] deletion_auc={summary['explain_deletion_auc_mean']:.4f} "
          f"insertion_auc={summary['explain_insertion_auc_mean']:.4f} "
          f"(overlays saved to {out_dir})")


if __name__ == "__main__":
    main()
