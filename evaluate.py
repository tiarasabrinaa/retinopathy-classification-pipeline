"""Full test-set evaluation for one already-trained model: accuracy,
macro/weighted precision-recall-F1, the full per-class classification
report, confusion matrices (raw counts + row-normalized), and multiclass
ROC-AUC (one-vs-rest, per class + macro).

    python evaluate.py --config configs/convnext_tiny.yaml
    python evaluate.py --config configs/deit_small.yaml

train.py's own final test-set pass only logs macro-F1 + a raw-count
confusion matrix (enough to pick the better model). This script produces
everything needed to write up the Results section: per-class
precision/recall/F1, ROC-AUC, and both confusion matrix views. Saved
locally under `checkpoint_dir/eval/` (JSON + text + 2 PNGs) and the scalar
metrics logged to MLflow under a `<run_name>_eval` run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from common import RetinaDataset, build_model, build_transforms, confusion_matrix_figure, get_device, init_mlflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    device = get_device()
    class_names = json.loads(Path(cfg["class_names_json"]).read_text())

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"]).to(device)
    checkpoint = torch.load(Path(cfg["checkpoint_dir"]) / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eval_tf = build_transforms(model, cfg["image_size"], is_train=False)
    test_loader = DataLoader(
        RetinaDataset(cfg["test_csv"], ".", eval_tf),
        batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
    )

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, targets in test_loader:
            probs = F.softmax(model(images.to(device)), dim=1)
            y_prob += probs.cpu().tolist()
            y_pred += probs.argmax(dim=1).cpu().tolist()
            y_true += targets.tolist()

    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_text)

    auc_per_class = roc_auc_score(y_true, y_prob, multi_class="ovr", average=None)
    auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    print(f"ROC-AUC (macro, one-vs-rest): {auc_macro:.4f}")
    for name, auc in zip(class_names, auc_per_class):
        print(f"  {name}: {auc:.4f}")
        report_dict[name]["roc_auc"] = float(auc)
    report_dict["roc_auc_macro"] = float(auc_macro)

    out_dir = Path(cfg["checkpoint_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2))
    (out_dir / "classification_report.txt").write_text(
        report_text + f"\nROC-AUC (macro, one-vs-rest): {auc_macro:.4f}\n" +
        "".join(f"  {name}: {auc:.4f}\n" for name, auc in zip(class_names, auc_per_class))
    )

    cm = confusion_matrix(y_true, y_pred)
    for normalize, filename in [(False, "confusion_matrix.png"), (True, "confusion_matrix_normalized.png")]:
        fig = confusion_matrix_figure(cm, class_names, normalize=normalize)
        fig.savefig(out_dir / filename, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)

    init_mlflow()
    with mlflow.start_run(run_name=f"{cfg['run_name']}_eval"):
        mlflow.log_metrics({
            "eval_accuracy": report_dict["accuracy"],
            "eval_macro_precision": report_dict["macro avg"]["precision"],
            "eval_macro_recall": report_dict["macro avg"]["recall"],
            "eval_macro_f1": report_dict["macro avg"]["f1-score"],
            "eval_weighted_f1": report_dict["weighted avg"]["f1-score"],
            "eval_roc_auc_macro": auc_macro,
            **{f"eval_{name}_precision": report_dict[name]["precision"] for name in class_names},
            **{f"eval_{name}_recall": report_dict[name]["recall"] for name in class_names},
            **{f"eval_{name}_f1": report_dict[name]["f1-score"] for name in class_names},
            **{f"eval_{name}_roc_auc": report_dict[name]["roc_auc"] for name in class_names},
        })
        mlflow.log_text(report_text, "classification_report.txt")
        mlflow.log_figure(confusion_matrix_figure(cm, class_names), "confusion_matrix.png")
        mlflow.log_figure(confusion_matrix_figure(cm, class_names, normalize=True), "confusion_matrix_normalized.png")

    print(f"\nSaved to {out_dir}/ and logged to MLflow run '{cfg['run_name']}_eval'")


if __name__ == "__main__":
    main()
