#!/usr/bin/env bash
# Full grid search: 3 finetune_lr candidates x 5 finetune_epochs budgets,
# for both architectures (linear_probe_epochs stays fixed at 5). 30 runs
# total, each a complete run with its own properly-annealed cosine LR
# schedule (T_max = finetune_epochs) -- not a cheap proxy search.
#
# Pick the winner per architecture by val_macro_f1 / val_classification_report.txt
# in MLflow (never by test_macro_f1 -- see README "Comparing learning rates").
#
#   bash start.sh
set -euo pipefail

cd "$(dirname "$0")"

LRS=(1e-4 1e-5 5e-5)
EPOCHS=(10 20 30 40 50)

for model in convnext_tiny deit_small; do
    for lr in "${LRS[@]}"; do
        for epochs in "${EPOCHS[@]}"; do
            echo "=== ${model}  finetune_lr=${lr}  finetune_epochs=${epochs} ==="
            python train.py \
                --config "configs/${model}.yaml" \
                --finetune-lr "${lr}" \
                --finetune-epochs "${epochs}"
        done
    done
done
