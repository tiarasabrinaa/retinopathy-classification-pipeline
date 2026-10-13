"""Deletion/insertion AUC for the trained models, plus one figure per model
showing what deletion and insertion actually do to an image.

    python faithfulness.py
    python faithfulness.py --configs configs/_xai_convnext_winner.yaml configs/_xai_deit_winner.yaml

For each config (each needs `checkpoint_dir/best.pt` from a prior `train.py`
run) this:
1. scores every test image with deletion and insertion AUC (or, with
   `--num-samples N`, a random sample of N of them -- same sampling as
   eval.ipynb: `random.Random(seed).sample`),
2. draws one example per class: input, heatmap, the image after its top
   `--fraction` of patches is deleted, the blurred baseline after the same
   patches are inserted, and both confidence curves.

The example images are the same for every model -- see `explain.pick_examples`
for the rule, or pass `--indices` (test.csv row numbers, one per class) to
choose them yourself. Everything is saved to `--out-dir`: `faithfulness.json`
(means, per-class means, and the score of every single image) and
`faithfulness_<model_name>.png`. Nothing is logged to MLflow.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import yaml

from common import RetinaDataset, build_model, build_transforms, get_device
from explain import (MODEL_LABELS, XAI_METHOD_LABELS, XAI_METHODS, _trapezoid_auc, blur_baseline, delete_patches,
                     deletion_curve, denormalize, insert_patches, insertion_curve, pick_examples, show_overlay)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+",
                        default=[Path("configs/_xai_convnext_winner.yaml"), Path("configs/_xai_deit_winner.yaml")])
    parser.add_argument("--num-samples", type=int, default=None,
                        help="score only a random sample of this many test images (default: the whole test set)")
    parser.add_argument("--seed", type=int, default=0, help="seeds the example choice and the --num-samples sample")
    parser.add_argument("--fraction", type=float, default=0.25, help="share of patches deleted/inserted in the figure")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="test.csv rows to draw, one per class")
    parser.add_argument("--out-dir", type=Path, default=Path("../results"))
    args = parser.parse_args()

    device = get_device()
    loaded = []
    for config_path in args.configs:
        cfg = yaml.safe_load(config_path.read_text())
        model = build_model(cfg["model_name"], num_classes=cfg["num_classes"]).to(device)
        checkpoint = torch.load(Path(cfg["checkpoint_dir"]) / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        data_config = timm.data.resolve_data_config({}, model=model)
        dataset = RetinaDataset(cfg["test_csv"], ".", build_transforms(model, cfg["image_size"], is_train=False))
        loaded.append((cfg, model, dataset, data_config["mean"], data_config["std"]))

    class_names = json.loads(Path(loaded[0][0]["class_names_json"]).read_text())
    examples = args.indices or pick_examples(
        [model for _, model, _, _, _ in loaded], [dataset for _, _, dataset, _, _ in loaded],
        len(class_names), args.seed,
    )
    print(f"example images (test.csv rows): {examples}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for cfg, model, dataset, mean, std in loaded:
        explain_fn = XAI_METHODS[cfg["xai_method"]]
        method_label = XAI_METHOD_LABELS[cfg["xai_method"]]
        model_label = MODEL_LABELS.get(cfg["model_name"], cfg["model_name"])

        def score(idx: int):
            """Heatmap + both confidence curves for test image `idx`."""
            image, true_label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            baseline = blur_baseline(image)
            patch_map, target_class = explain_fn(model, image)
            deletion = deletion_curve(model, image, patch_map, target_class, baseline)
            insertion = insertion_curve(model, image, patch_map, target_class, baseline)
            record = {
                "index": idx,
                "filepath": dataset.df.iloc[idx]["filepath"],
                "true": class_names[int(true_label)],
                "pred": class_names[target_class],
                "deletion_auc": _trapezoid_auc(deletion),
                "insertion_auc": _trapezoid_auc(insertion),
            }
            return record, image, baseline, patch_map, deletion, insertion

        # --- 1. deletion/insertion AUC over the test set ---
        sample = list(range(len(dataset)))
        if args.num_samples is not None:
            sample = random.Random(args.seed).sample(sample, min(args.num_samples, len(dataset)))
        records = []
        for done, idx in enumerate(sample, start=1):
            records.append(score(idx)[0])
            if done % 50 == 0 or done == len(sample):
                print(f"[{model_label}] scored {done}/{len(sample)} images")

        per_class = {}
        for name in class_names:
            rows = [r for r in records if r["pred"] == name]
            if rows:
                per_class[name] = {
                    "n": len(rows),
                    "deletion_auc_mean": float(np.mean([r["deletion_auc"] for r in rows])),
                    "insertion_auc_mean": float(np.mean([r["insertion_auc"] for r in rows])),
                }
        deletion_mean = float(np.mean([r["deletion_auc"] for r in records]))
        insertion_mean = float(np.mean([r["insertion_auc"] for r in records]))
        print(f"[{model_label} / {method_label}] deletion_auc={deletion_mean:.4f} insertion_auc={insertion_mean:.4f}")

        # --- 2. figure: what deletion and insertion do to one image per class ---
        percent = round(args.fraction * 100)
        fig, axes = plt.subplots(len(examples), 5, figsize=(13, 2.7 * len(examples)),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.5]}, squeeze=False)
        example_records = []
        for row, idx in enumerate(examples):
            record, image, baseline, patch_map, deletion, insertion = score(idx)
            example_records.append(record)
            n = round(args.fraction * patch_map.size)
            rgb = denormalize(image, mean, std)

            axes[row, 0].imshow(rgb)
            axes[row, 0].set_xlabel(f"confidence {deletion[0]:.2f}")
            axes[row, 0].set_ylabel(record["true"].replace("_", " "))
            show_overlay(axes[row, 1], rgb, patch_map)
            axes[row, 1].set_xlabel(f"pred: {record['pred'].replace('_', ' ')}")
            axes[row, 2].imshow(denormalize(delete_patches(image, patch_map, baseline, n), mean, std))
            axes[row, 2].set_xlabel(f"confidence {deletion[n]:.2f}")
            axes[row, 3].imshow(denormalize(insert_patches(image, patch_map, baseline, n), mean, std))
            axes[row, 3].set_xlabel(f"confidence {insertion[n]:.2f}")
            for ax in axes[row, :4]:
                ax.set_xticks([])
                ax.set_yticks([])

            x = np.linspace(0, 1, len(deletion))
            axes[row, 4].plot(x, deletion, color="tab:red", label=f"deletion (AUC {record['deletion_auc']:.2f})")
            axes[row, 4].plot(x, insertion, color="tab:blue", label=f"insertion (AUC {record['insertion_auc']:.2f})")
            axes[row, 4].axvline(args.fraction, color="gray", linestyle=":", linewidth=1)
            axes[row, 4].set_ylim(0, 1.02)
            axes[row, 4].set_xlim(0, 1)
            axes[row, 4].set_ylabel("confidence")
            axes[row, 4].legend(fontsize=8, loc="best")
        axes[-1, 4].set_xlabel("fraction of patches deleted / inserted")
        for ax, title in zip(axes[0], ["Input", method_label, f"Top {percent}% deleted",
                                       f"Top {percent}% inserted", "Confidence curves"]):
            ax.set_title(title)
        fig.suptitle(f"{method_label} on {model_label}")
        fig.tight_layout()
        figure_path = args.out_dir / f"faithfulness_{cfg['model_name']}.png"
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)
        print(f"[{model_label}] figure saved to {figure_path}")

        results[cfg["model_name"]] = {
            "xai_method": cfg["xai_method"],
            "checkpoint": str(Path(cfg["checkpoint_dir"]) / "best.pt"),
            "seed": args.seed,
            "num_images": len(records),
            "deletion_auc_mean": deletion_mean,
            "insertion_auc_mean": insertion_mean,
            "per_predicted_class": per_class,
            "examples": example_records,
            "per_image": records,
        }

    (args.out_dir / "faithfulness.json").write_text(json.dumps(results, indent=2))
    print(f"scores saved to {args.out_dir / 'faithfulness.json'}")


if __name__ == "__main__":
    main()
