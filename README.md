# Retinopathy Disease Classification — ConvNeXt vs DeiT

Fine-tunes two pretrained vision backbones — **ConvNeXt-Tiny** (CNN) and
**DeiT-Small** (ViT) — on a 4-class fundus image dataset (`cataract`,
`diabetic_retinopathy`, `glaucoma`, `normal`), tracks every run in
MLflow, and picks the better model for the explainability phase.

## Project structure

```
dataset/                 # ImageFolder layout: dataset/<class_name>/*.jpg (gitignored, ~1000 img/class)
dataset_splits/          # committed CSV manifests, generated once by make_splits.py
configs/
  convnext_tiny.yaml
  deit_small.yaml
common.py                # Dataset, transforms, model builder, MLflow helpers — shared by the scripts below
explain.py                 # Grad-CAM, Attention Rollout, deletion/insertion AUC — shared by run_explainability.py
make_splits.py            # run once: builds the stratified train/val/test split
train.py                  # fine-tunes ONE model per run, logs to MLflow
start.sh                    # runs the full finetune_lr x finetune_epochs grid (see "Comparing learning rates and epoch budgets")
evaluate.py                  # full classification report + ROC-AUC + confusion matrices for one trained model
run_explainability.py       # explainability + faithfulness scoring for one trained model
```

No `src/` package, no framework-y layers — this is a 2-model research
pipeline, not a service, so everything reusable fits in one `common.py` and
each script is readable top to bottom.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATABRICKS_HOST / DATABRICKS_TOKEN / MLFLOW_EXPERIMENT_NAME
```

**MLflow tracking (Databricks Community Edition):**
1. Sign up / log in at https://community.cloud.databricks.com
2. Settings → Developer → Access tokens → generate a Personal Access Token
   ([docs](https://docs.databricks.com/en/dev-tools/auth/pat.html))
3. Put the token + workspace URL in `.env`. Experiment name must be an
   absolute path under your user folder, e.g.
   `/Users/you@example.com/retinopathy-disease-classification`.
4. `train.py` authenticates via `mlflow.set_tracking_uri("databricks")`,
   which reads `DATABRICKS_HOST`/`DATABRICKS_TOKEN` from the environment —
   no notebook context needed, works fine from a plain RunPod pod.

> **Storage quota note:** Databricks CE has a small free artifact quota.
> Model checkpoints and full-res heatmaps are saved to local/network-volume
> disk (`checkpoint_dir` in the configs), **not** logged as MLflow
> artifacts. Only params, scalar metrics, and small plots (confusion
> matrix) go to MLflow.

## Running

```bash
# 1. one-time: build the shared stratified split (already committed under dataset_splits/,
#    re-run only if the dataset changes)
python make_splits.py --val-size 0.15 --test-size 0.15 --seed 42

# 2. fine-tune each model (run on a RunPod GPU pod for real training)
python train.py --config configs/convnext_tiny.yaml
python train.py --config configs/deit_small.yaml
```

Both configs share the exact same split/seed/augmentation/optimizer/schedule
so the ConvNeXt-vs-DeiT comparison isn't confounded by different setups —
only the architecture-specific fields (`model_name`, `checkpoint_dir`,
`run_name`) differ between the two YAMLs.

Each run does:
1. **Phase 1 — linear probe**: backbone frozen, only the classifier head
   trains (`linear_probe_epochs`, higher LR).
2. **Phase 2 — full fine-tune**: every parameter trains (`finetune_epochs`,
   low LR, cosine decay).
3. Best checkpoint (by validation macro-F1) is evaluated once on the held-out
   test set; test metrics + confusion matrix are logged to MLflow. A
   validation-set classification report (per-class precision/recall/F1) is
   also logged as an MLflow text artifact (`val_classification_report.txt`)
   on every run — safe to check any time since it never touches the test set.

Comparing the two finished models' `test_macro_f1` (with per-class recall on
`diabetic_retinopathy` as a tie-breaker, since missing DR is the costlier
error) is the final head-to-head result. Both models move on to the
explainability phase below — the comparison isn't just "which one wins,"
it's ConvNeXt vs DeiT on both accuracy and explainability faithfulness.

## Comparing learning rates and epoch budgets

`train.py` accepts `--finetune-lr` / `--finetune-epochs` to override a
config's values for a grid search, instead of hand-maintaining one YAML
file per combination. Each override run gets a unique `run_name` /
`checkpoint_dir` (suffixed `_lr<value>_ep<epochs>`) so nothing collides with
the main run. `linear_probe_epochs` stays fixed (from the base config);
`--finetune-epochs` also reshapes that run's cosine LR schedule (`T_max`),
so each budget gets a properly-annealed schedule of its own rather than
being read off a single longer run.

`start.sh` runs the full grid: 3 `finetune_lr` candidates
(`1e-4`, `1e-5`, `5e-5`) x 5 `finetune_epochs` budgets (`10, 20, 30, 40, 50`),
for both architectures -- 30 complete runs:

```bash
bash start.sh
```

Compare runs by **`val_macro_f1`** (or the fuller
`val_classification_report.txt` artifact) in MLflow to pick a winner per
architecture -- **never** by `test_macro_f1` / `test_classification_report.txt`,
even though every run logs those too (for record-keeping only). Using test
results to choose between configs is exactly the leakage the paper's own
methodology (Sec. 3.1.3) warns against. Once a `(finetune_lr, finetune_epochs)`
combo wins for an architecture, update `configs/convnext_tiny.yaml` /
`configs/deit_small.yaml` to match, and copy that winning run's checkpoint
(`checkpoints/<model>_lr<...>_ep<...>/best.pt`) to `checkpoints/<model>/best.pt`
instead of re-training, since it was already a full run.

## Explainability phase

Once a model has a `checkpoint_dir/best.pt` from `train.py`, run:

```bash
python run_explainability.py --config configs/convnext_tiny.yaml
python run_explainability.py --config configs/deit_small.yaml
```

Each config's `xai_method` field picks the method to use — `grad_cam` for
ConvNeXt (CNN-native, uses the last conv feature map) or
`attention_rollout` for DeiT (ViT-native, uses the self-attention matrices).
Both produce a 14x14 patch-importance heatmap (16px patches, matching
DeiT's native patch grid) so faithfulness scores stay comparable across
architectures.

For each of `--num-samples` (default 40) randomly sampled test images, the
script:
1. generates the heatmap via the configured method,
2. scores it with **deletion AUC** and **insertion AUC** (RISE-style,
   patch-based, blurred-image baseline) — see `explain.py` for the
   implementation and paper `documents/paper/sections/03_methodology.tex`
   for the write-up,
3. saves a heatmap-overlay PNG to `checkpoint_dir/explanations/`.

Overlay PNGs stay local (same storage-quota reasoning as checkpoints, see
below); only the mean deletion/insertion AUC (overall + per class) is
logged to MLflow, under a `<run_name>_explain` run.

## Sources

Code patterns here are adapted from, and cited inline as comments where used:
- [timm](https://github.com/huggingface/pytorch-image-models) — pretrained model loading (`create_model`), transform/normalization resolution
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) (Liu et al., 2022) — fine-tuning recipe (AdamW, weight decay, label smoothing)
- [DeiT](https://github.com/facebookresearch/deit) (Touvron et al., 2021) — fine-tuning recipe
- [Grad-CAM](https://arxiv.org/abs/1610.02391) (Selvaraju et al., 2017) — explanation method for `explain.grad_cam`
- [Attention Rollout](https://arxiv.org/abs/2005.00928) (Abnar & Zuidema, 2020), [reference implementation](https://github.com/samiraabnar/attention_flow) — explanation method for `explain.attention_rollout`
- [RISE](https://arxiv.org/abs/1806.07421) (Petsiuk et al., 2018) — deletion/insertion AUC faithfulness metric
- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — freeze/unfreeze two-phase pattern
- [PyTorch ImageNet example](https://github.com/pytorch/examples/blob/main/imagenet/main.py) — train/eval loop structure
- [MLflow tracking docs](https://mlflow.org/docs/latest/tracking.html) / [Databricks-hosted tracking server](https://docs.databricks.com/en/mlflow/access-hosted-tracking-server.html)
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — stratified splitting, macro-F1/confusion matrix
