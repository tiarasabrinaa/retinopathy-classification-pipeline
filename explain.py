"""Grad-CAM (ConvNeXt), Attention Rollout (DeiT), and deletion/insertion
faithfulness scoring, shared by run_explainability.py.

Heatmaps are produced as a 14x14 patch-importance grid (16x16 px patches
over a 224x224 image) rather than per-pixel, matching DeiT's native patch
grid. This keeps deletion/insertion AUC comparable across both
architectures, since the same masking granularity is used regardless of
which model/method produced the heatmap.

Sources:
- Grad-CAM: https://arxiv.org/abs/1610.02391 (Selvaraju et al., 2017)
- Attention Rollout: https://arxiv.org/abs/2005.00928 (Abnar & Zuidema, 2020)
  Identity-matrix addition before renormalizing (to account for the
  transformer's residual connections) follows the reference implementation
  linked from that paper: https://github.com/samiraabnar/attention_flow
- Deletion/Insertion AUC (RISE): https://arxiv.org/abs/1806.07421 (Petsiuk et al., 2018)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

PATCH_GRID = 14  # 224 / 16, matches DeiT's patch16 grid


def _normalize(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    denom = x.max()
    return x / denom if denom > 0 else x


# --- Grad-CAM (ConvNeXt) -----------------------------------------------------

def grad_cam(model: torch.nn.Module, image: torch.Tensor, target_class: int | None = None):
    """`image`: normalized image tensor, shape (1, 3, H, W).
    Returns (patch_map, target_class): patch_map is a (14, 14) np.ndarray in
    [0, 1], target_class is the class the CAM was computed for.
    """
    model.eval()
    image = image.clone().requires_grad_(True)

    activation = {}

    def save_activation(_module, _inp, out):
        out.retain_grad()
        activation["value"] = out

    handle = model.stages[-1].register_forward_hook(save_activation)
    logits = model(image)
    handle.remove()

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    model.zero_grad(set_to_none=True)
    logits[0, target_class].backward()

    feature_map = activation["value"].detach()             # (1, C, h, w)
    weights = activation["value"].grad.mean(dim=(2, 3), keepdim=True)  # Eq. 1: GAP of gradients
    cam = F.relu((weights * feature_map).sum(dim=1, keepdim=True))     # Eq. 2

    patch_map = F.interpolate(cam, size=(PATCH_GRID, PATCH_GRID), mode="bilinear", align_corners=False)
    return _normalize(patch_map[0, 0]).detach().cpu().numpy(), target_class


# --- Attention Rollout (DeiT) ------------------------------------------------

def attention_rollout(model: torch.nn.Module, image: torch.Tensor, target_class: int | None = None):
    """`image`: normalized image tensor, shape (1, 3, H, W).
    Returns (patch_map, target_class) like `grad_cam`.
    """
    model.eval()
    attn_maps = []

    def save_attn(_module, _inp, out):
        attn_maps.append(out.detach())

    # timm's Attention only materializes softmax weights on the manual path
    # (the fused scaled_dot_product_attention path never exposes them), so
    # force it off for the duration of this forward pass.
    handles, was_fused = [], []
    for block in model.blocks:
        was_fused.append(block.attn.fused_attn)
        block.attn.fused_attn = False
        handles.append(block.attn.attn_drop.register_forward_hook(save_attn))

    with torch.no_grad():
        logits = model(image)

    for block, fused in zip(model.blocks, was_fused):
        block.attn.fused_attn = fused
    for h in handles:
        h.remove()

    if target_class is None:
        target_class = int(logits.argmax(dim=1).item())

    num_tokens = attn_maps[0].shape[-1]
    rollout = torch.eye(num_tokens, device=attn_maps[0].device)
    for attn in attn_maps:                                   # Eq. 3, recursively composed
        attn = attn.mean(dim=1)[0]                            # average over heads -> (N, N)
        attn = attn + torch.eye(num_tokens, device=attn.device)  # account for the residual connection
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn @ rollout

    cls_attention = rollout[0, 1:]                           # CLS row, patch tokens only
    patch_map = _normalize(cls_attention.reshape(PATCH_GRID, PATCH_GRID))
    return patch_map.cpu().numpy(), target_class


# --- Deletion / Insertion AUC (faithfulness) ---------------------------------

def blur_baseline(image: torch.Tensor) -> torch.Tensor:
    """The 'uninformative' image used by both metrics: a heavily blurred
    version of the same input, so replaced/inserted regions stay in-distribution
    (no sharp black/gray patch edges the model has never seen)."""
    return gaussian_blur(image, kernel_size=51, sigma=41.0)


def _patch_order(patch_map: np.ndarray) -> np.ndarray:
    """Patch indices (row-major into the 14x14 grid), most important first."""
    return np.argsort(patch_map.ravel())[::-1]


def _trapezoid_auc(scores: list[float]) -> float:
    """Trapezoidal AUC over `scores` sampled at evenly spaced x in [0, 1].
    Written by hand instead of calling numpy's trapz/trapezoid because that
    function was renamed between numpy 1.x and 2.x."""
    step = 1.0 / (len(scores) - 1)
    return sum((scores[i] + scores[i + 1]) / 2 * step for i in range(len(scores) - 1))


def _replace_patches(image: torch.Tensor, patch_indices, baseline: torch.Tensor) -> torch.Tensor:
    replaced = image.clone()
    patch_size = image.shape[-1] // PATCH_GRID
    for idx in patch_indices:
        r, c = divmod(int(idx), PATCH_GRID)
        rs, cs = r * patch_size, c * patch_size
        replaced[..., rs:rs + patch_size, cs:cs + patch_size] = \
            baseline[..., rs:rs + patch_size, cs:cs + patch_size]
    return replaced


@torch.no_grad()
def deletion_auc(model: torch.nn.Module, image: torch.Tensor, patch_map: np.ndarray,
                  target_class: int, baseline: torch.Tensor) -> float:
    """Starts from the full image, progressively replaces the most important
    patches with `baseline` first. Lower AUC = more faithful explanation."""
    model.eval()
    order = _patch_order(patch_map)
    scores = [
        F.softmax(model(_replace_patches(image, order[:n], baseline)), dim=1)[0, target_class].item()
        for n in range(len(order) + 1)
    ]
    return _trapezoid_auc(scores)


@torch.no_grad()
def insertion_auc(model: torch.nn.Module, image: torch.Tensor, patch_map: np.ndarray,
                   target_class: int, baseline: torch.Tensor) -> float:
    """Starts from `baseline`, progressively inserts the most important
    original patches first. Higher AUC = more faithful explanation."""
    model.eval()
    order = _patch_order(patch_map)
    scores = [
        F.softmax(model(_replace_patches(image, order[n:], baseline)), dim=1)[0, target_class].item()
        for n in range(len(order) + 1)
    ]
    return _trapezoid_auc(scores)
