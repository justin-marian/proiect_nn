import os
from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def plot_dists(
    val_dict: dict[str, np.ndarray],
    color: str = "C0",
    xlabel: str | None = None,
    stat: str = "count",
    use_kde: bool = False,
    path: str = "output",
    name: str = "distributions.png"
):
    """
    Plot distributions for a collection of arrays (one subplot per entry).
    Create a single-row figure with one subplot for each key. 
    Each subplot displays a histogram of the corresponding array using
    seaborn.histplot. Intended to be used for quick inspection of weight,
    gradient or activation distributions in neural networks.
    """
    os.makedirs(path, exist_ok=True)
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))
    fig_index = 0

    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index % columns] if columns > 1 else ax
        sns.histplot(
            val_dict[key], ax=key_ax, color=color, bins=50, stat=stat,
            kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8)
        )

        val_counts = val_dict[key].shape[0]
        conditional = val_dict[key].shape[1] if len(val_dict[key].shape) > 1 else 1
        key_ax.set_title(
            f"{key} " + (r"(%i $\to$ %i)" % conditional)
            if conditional > 1 else "(%i vals)" % val_counts
        )

        if xlabel is not None:
            key_ax.set_xlabel(xlabel)
        fig_index += 1

    fig.subplots_adjust(wspace=0.4)
    save_path = os.path.join(path, name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    logger.info(f"Saved plot to {save_path}")
    return fig


def visualize_weight_distribution(
    model: nn.Module, 
    color: str = "C0",
    path: str = "output",
    name: str = "weights"
):
    """
    Visualize the distribution of trainable (non-bias) weights in a model.
    This function collects all parameters from except those whose names end with ".bias",
    flattens each parameter tensor to a 1-D NumPy array, and plots their distributions. 
    Each parameter tensor is labeled by the layer index inferred from the parameter name
    (e.g., "Layer 0", "Layer 1", ...). The resulting figure is displayed and then closed.
    """
    weights = {}
    for pname, param in model.named_parameters():
        if pname.endswith(".bias"):
            continue
        key_name = f"Layer {pname.split('.')[1]}"
        weights[key_name] = param.detach().view(-1).cpu().numpy()

    plt.ioff()
    fig = plot_dists(weights, color=color, xlabel="Weight vals", path=path, name=name + ".png")
    fig.suptitle("Weight distribution", fontsize=14, y=1.05)
    plt.close(fig)


def visualize_gradients(
    model: nn.Module,
    color: str = "C0",
    train_set: data.Dataset = None,
    device: torch.device = None,
    batch_size: int = 256,
    path: str = "output/gradients",
    name: str = "gradients"
):
    """
    Visualize the distribution of gradient magnitudes across model weights after one backward pass.
    Perform a single forward and backward step using a small batch of samples
    from the provided training dataset. It collects gradients of all parameters whose names contain
    "weight", flattens them to 1-D arrays, and plots their magnitude distributions.
    Each layer's gradients are represented as a separate distribution curve in the resulting plot.
    Optionally, the variance of each layer's gradients can be printed to the console.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    small_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    imgs, labels = next(iter(small_loader))
    imgs, labels = imgs.to(device), labels.to(device)

    model.zero_grad()
    preds = model(imgs)
    loss = F.cross_entropy(preds, labels)
    loss.backward()

    grads = {
        name: param.grad.detach().view(-1).abs().cpu().numpy()
        for name, param in model.named_parameters()
        if "weight" in name and param.grad is not None
    }
    model.zero_grad()

    plt.ioff()
    fig = plot_dists(grads, color=color, xlabel="Grad magnitude", path=path, name=name + ".png")
    fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
    plt.close(fig)


def visualize_activations(
    model: nn.Module,
    color: str = "C0",
    print_variance: bool = False,
    train_set: data.Dataset = None,
    device: torch.device = None,
    batch_size: int = 256,
    max_samples_per_layer: int = 100_000,
    path: str = "output/activations",
    name: str = "activations"
):
    """
    Visualize the distribution of neuron activations across linear/conv layers in a model.
    Uses a single (small) batch to avoid OOM. Each layer's activations are subsampled if
    they exceed `max_samples_per_layer` to keep CPU memory usage bounded.
    """
    if train_set is None:
        logger.warning("No train_set provided to visualize_activations; skipping.")
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

    def make_hook(label: str) -> Callable:
        def hook(model, input, output):
            tensor_out = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                flat = tensor_out.detach().view(-1)
                n = flat.numel()
                if n > max_samples_per_layer:
                    idx = torch.randperm(n, device=flat.device)[:max_samples_per_layer]
                    flat = flat[idx]
                activations[label] = flat.cpu().numpy()
        return hook

    idx = 0
    for _, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            label = f"{module.__class__.__name__} {idx}"
            hooks.append(module.register_forward_hook(make_hook(label)))
            idx += 1

    with torch.no_grad():
        _ = model(imgs)
    for h in hooks:
        h.remove()

    if not activations:
        logger.warning("No activations were collected (no Linear/Conv2d layers found).")
        return

    if print_variance:
        for key in sorted(activations.keys()):
            logger.info(f"{key} - Variance: {np.var(activations[key])}")

    plt.ioff()
    fig = plot_dists(
        activations,
        color=color,
        stat="density",
        use_kde=False,
        xlabel="Activation vals",
        path=path,
        name=name + ".png"
    )
    fig.suptitle("Activation distribution", fontsize=14, y=1.05)
    plt.close(fig)
