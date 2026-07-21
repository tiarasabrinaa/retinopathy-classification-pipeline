"""Build ONE fixed stratified train/val/test split, shared by every model.

Run this once and commit the resulting CSVs. Both ConvNeXt and DeiT read from
the same dataset_splits/*.csv so the head-to-head comparison isn't confounded
by different splits.

Two-step stratified split (sklearn has no single-call 3-way stratified split,
so this is the standard train_test_split-twice recipe):
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

Usage:
    python make_splits.py --val-size 0.15 --test-size 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_dataset(data_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Assumes torchvision ImageFolder layout: dataset/<class_name>/*.jpg"""
    rows = []
    class_names = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    for label_idx, class_name in enumerate(class_names):
        for img_path in sorted((data_root / class_name).iterdir()):
            if img_path.suffix.lower() in IMG_EXTENSIONS:
                rows.append(
                    {
                        "filepath": str(img_path.relative_to(data_root.parent)),
                        "label": class_name,
                        "label_idx": label_idx,
                    }
                )
    return pd.DataFrame(rows), class_names


def stratified_split(df: pd.DataFrame, val_size: float, test_size: float, seed: int):
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["label"],
        random_state=seed,
    )
    return train_df, val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("dataset_splits"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df, class_names = scan_dataset(args.data_root)
    train_df, val_df, test_df = stratified_split(df, args.val_size, args.test_size, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out_dir / "train.csv", index=False)
    val_df.to_csv(args.out_dir / "val.csv", index=False)
    test_df.to_csv(args.out_dir / "test.csv", index=False)
    (args.out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))

    print(f"classes: {class_names}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"{name}: {len(split_df)} images -> {split_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
