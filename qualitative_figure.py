"""Qualitative comparison figure: the same test image per class, explained
by every trained model's own XAI method (Grad-CAM for ConvNeXt, Attention
Rollout for DeiT), side by side.

    python qualitative_figure.py
    python qualitative_figure.py --indices 12 345 67 89

One row per class, one column for the input plus one per config. The images
are chosen by `explain.pick_examples` (same rule and seed as faithfulness.py,
so both figures show the same images), or by `--indices` (test.csv row
numbers, one per class). Saves `xai_qualitative_comparison.png` and
`xai_qualitative_examples.json` (which images were drawn) to `--out-dir`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import timm
import torch
import yaml

from common import RetinaDataset, build_model, build_transforms, get_device
from explain import MODEL_LABELS, XAI_METHOD_LABELS, XAI_METHODS, denormalize, pick_examples, show_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="+",
                        default=[Path("configs/_xai_convnext_winner.yaml"), Path("configs/_xai_deit_winner.yaml")])
    parser.add_argument("--seed", type=int, default=0, help="seeds the example choice")
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

    num_cols = 1 + len(loaded)
    fig, axes = plt.subplots(len(examples), num_cols, figsize=(2.5 * num_cols, 2.5 * len(examples)), squeeze=False)
    example_records = []
    for row, idx in enumerate(examples):
        record = {"index": idx, "filepath": loaded[0][2].df.iloc[idx]["filepath"]}
        for col, (cfg, model, dataset, mean, std) in enumerate(loaded, start=1):
            image, true_label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            patch_map, target_class = XAI_METHODS[cfg["xai_method"]](model, image)
            rgb = denormalize(image, mean, std)
            if col == 1:
                axes[row, 0].imshow(rgb)
                axes[row, 0].set_ylabel(class_names[int(true_label)].replace("_", " "))
                record["true"] = class_names[int(true_label)]
            show_overlay(axes[row, col], rgb, patch_map)
            record[f"pred_{cfg['model_name']}"] = class_names[target_class]
        example_records.append(record)

    axes[0, 0].set_title("Input")
    for col, (cfg, _, _, _, _) in enumerate(loaded, start=1):
        model_label = MODEL_LABELS.get(cfg["model_name"], cfg["model_name"])
        axes[0, col].set_title(f"{XAI_METHOD_LABELS[cfg['xai_method']]}\n({model_label})")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_dir / "xai_qualitative_comparison.png", dpi=200)
    plt.close(fig)
    (args.out_dir / "xai_qualitative_examples.json").write_text(json.dumps(example_records, indent=2))
    print(f"saved to {args.out_dir}/ (test.csv rows: {examples})")


if __name__ == "__main__":
    main()
