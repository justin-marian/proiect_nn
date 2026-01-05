from __future__ import annotations
from __future__ import print_function

import os
from typing import Callable, Optional

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data

import matplotlib.pyplot as plt
import matplotlib.figure as fig
from loguru import logger


def plot_dists(
    values: dict[str, np.ndarray],
    xlabel: Optional[str],
    bins: int = 30,
    density: bool = False,
    out_dir: str = "output",
    file_name: str = "distributions.png",
) -> fig.Figure:
    os.makedirs(out_dir, exist_ok=True)

    keys = sorted(values.keys())
    ncols = max(1, len(keys))
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.2, 2.6))
    if ncols == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        arr = np.asarray(values[key]).reshape(-1)
        if arr.size == 0:
            ax.set_title(f"{key} (empty)")
            ax.axis("off")
            continue

        ax.hist(arr, bins=bins, density=density)
        ax.set_title(f"{key} ({arr.size} vals)")
        if xlabel is not None:
            ax.set_xlabel(xlabel)

    fig.subplots_adjust(wspace=0.35)
    save_path = os.path.join(out_dir, file_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    logger.info(f"Saved plot to {save_path}")
    return fig


def visualize_weight_distribution(
    model: nn.Module, *,
    out_dir: str = "output",
    file_name: str = "weights.png",
) -> None:
    weights: dict[str, np.ndarray] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias"):
            continue
        weights[name] = p.detach().cpu().view(-1).numpy()

    if not weights:
        logger.warning("No non-bias trainable parameters found.")
        return

    plt.ioff()
    fig = plot_dists(weights, xlabel="Weight value", out_dir=out_dir, file_name=file_name)
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.close(fig)


def visualize_gradients(
    model: nn.Module,
    train_set: data.Dataset, *,
    device: Optional[torch.device],
    batch_size: int = 256,
    out_dir: str = "output",
    file_name: str = "gradients.png",
) -> None:
    if train_set is None:
        raise ValueError("Train set must be provided to visualize gradients.")

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)

    model.zero_grad(set_to_none=True)
    preds = model(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()

    grads: dict[str, np.ndarray] = {}
    for name, p in model.named_parameters():
        if "weight" not in name:
            continue
        if p.grad is None:
            continue
        # Take absolute value of gradients to analyze magnitude distribution
        grads[name] = p.grad.detach().abs().cpu().view(-1).numpy()

    model.zero_grad(set_to_none=True)

    if not grads:
        logger.warning("No gradients collected (no 'weight' grads found).")
        return

    plt.ioff()
    fig = plot_dists(grads, xlabel="|grad|", out_dir=out_dir, file_name=file_name)
    fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
    plt.close(fig)


def visualize_activations(
    model: nn.Module,
    train_set: data.Dataset, *,
    device: Optional[torch.device],
    batch_size: int = 256,
    max_samples_per_layer: int = 100_000,
    out_dir: str = "output",
    file_name: str = "activations.png",
    print_variance: bool = False,
) -> None:
    if train_set is None:
        logger.warning("No train set provided, skipping activation visualization.")
        return

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)

    activations: dict[str, np.ndarray] = {}
    hooks = []

    def register_hook(layer_name: str) -> Callable:
        def hook(_, __, output):
            out = output[0] if isinstance(output, tuple) else output
            flat = out.detach().reshape(-1)

            # Subsample if too many activations to avoid memory issues
            # Take random subset to get representative distribution
            if flat.numel() > max_samples_per_layer:
                idx = torch.randperm(flat.numel(), device=flat.device)[:max_samples_per_layer]
                flat = flat[idx]

            activations[layer_name] = flat.cpu().numpy()
        return hook

    # Register hooks on Linear and Conv2d layers to capture activations
    # during forward pass through the model
    layer_idx = 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            name = f"{module.__class__.__name__} {layer_idx}"
            hooks.append(module.register_forward_hook(register_hook(name)))
            layer_idx += 1

    with torch.no_grad():
        _ = model(imgs)

    for h in hooks:
        h.remove()

    if not activations:
        logger.warning("No activations collected (no Linear/Conv2d layers found).")
        return

    if print_variance:
        for k in sorted(activations.keys()):
            logger.info(f"{k} variance: {np.var(activations[k])}")

    plt.ioff()
    fig = plot_dists(
        activations,
        xlabel="Activation value",
        density=True,
        out_dir=out_dir,
        file_name=file_name,
    )
    fig.suptitle("Activation distribution", fontsize=14, y=1.05)
    plt.close(fig)
