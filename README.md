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
make_splits.py            # run once: builds the stratified train/val/test split
train.py                  # fine-tunes ONE model per run, logs to MLflow
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
   test set; test metrics + confusion matrix are logged to MLflow.

The model with the better test macro-F1 (with per-class recall on
`diabetic_retinopathy` as a tie-breaker, since missing DR is the costlier
error) is the one that moves on to the explainability phase.

## Not yet implemented (next phase)

Grad-CAM (ConvNeXt), Attention Rollout (DeiT), and deletion/insertion AUC
faithfulness scoring are planned but not built yet — this repo currently
only covers data prep + fine-tuning + tracking.

## Sources

Code patterns here are adapted from, and cited inline as comments where used:
- [timm](https://github.com/huggingface/pytorch-image-models) — pretrained model loading (`create_model`), transform/normalization resolution
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) (Liu et al., 2022) — fine-tuning recipe (AdamW, weight decay, label smoothing)
- [DeiT](https://github.com/facebookresearch/deit) (Touvron et al., 2021) — fine-tuning recipe
- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — freeze/unfreeze two-phase pattern
- [PyTorch ImageNet example](https://github.com/pytorch/examples/blob/main/imagenet/main.py) — train/eval loop structure
- [MLflow tracking docs](https://mlflow.org/docs/latest/tracking.html) / [Databricks-hosted tracking server](https://docs.databricks.com/en/mlflow/access-hosted-tracking-server.html)
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — stratified splitting, macro-F1/confusion matrix
