import os
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torchviz import make_dot


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_tensor_for_viz(output: Any) -> torch.Tensor:
    """
    Convert common model outputs (tensor / dict / list-of-dicts)
    into a single tensor for make_dot.
    This does NOT change the model; it's only for visualization.
    """
    if torch.is_tensor(output):
        return output

    # Dict of tensors for outputs
    if isinstance(output, dict):
        tensors = [v for v in output.values() if torch.is_tensor(v)]
        if not tensors:
            raise ValueError("No tensor values found in dict output for visualization.")
        return sum(tensors)

    # List or tuple of tensors / dicts
    if isinstance(output, (list, tuple)) and len(output) > 0:
        parts = []
        for elem in output:
            if torch.is_tensor(elem):
                parts.append(elem.sum())
            elif isinstance(elem, dict):
                for v in elem.values():
                    if torch.is_tensor(v):
                        parts.append(v.sum())
        if not parts:
            raise ValueError("No tensor values found in list/tuple output for visualization.")
        return sum(parts)

    raise TypeError(
        f"Unsupported output type {type(output)} for visualization. "
        "Expected tensor, dict-of-tensors, or list/tuple containing tensors/dicts."
    )


def visualize_model(
    model: nn.Module,
    arch: str = "",
    experiment: str = "",
    path: str = "output",
    *forward_args: Any,
    **forward_kwargs: Any,
):
    """
    Build and save a torchviz graph for the given model.
    You MUST pass the same arguments you normally give to `model(...)`:
        visualize_model(model, 'fasterrcnn', 'exp1', images, targets)
        visualize_model(model, 'clfnet', 'exp1', x)
    - Works for models that return:
        * a tensor
        * a dict of tensors
        * a list/tuple of tensors or dicts (typical in detection)
    - Saves to: {path}/{arch}_{experiment}.png
    """
    model = model.to("cpu")
    model.eval()  # use whatever mode you prefer (usually eval)

    # Forward with user-provided inputs (no dummy creation here)
    output = model(*forward_args, **forward_kwargs)

    y = to_tensor_for_viz(output)
    dot = make_dot(y, params=dict(model.named_parameters()))

    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"{arch}_{experiment}")

    dot.format = "png"
    dot.render(out_path, cleanup=True)
    logger.info(f"Model graph saved to {out_path}.png")
    return dot


def save_model(path: str, name: str, net: nn.Module):
    os.makedirs(path, exist_ok=True)
    model_file = os.path.join(path, name + ".pth")
    torch.save(net.state_dict(), model_file)


def load_model(path: str, name: str, net: nn.Module) -> nn.Module:
    model_file = os.path.join(path, name + ".pth")

    assert os.path.isfile(model_file), f'Could not find the model file "{model_file}".'
    assert net is not None, "Model instance must be provided to load the state dict."

    net.load_state_dict(torch.load(model_file, map_location=device))
    return net
