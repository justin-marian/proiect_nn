from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from data.datasets.voc import VOC_CLASSES


@dataclass(frozen=True)
class ClassColor:
    """For image visualization purposes when see the detected boxes with colors/labels."""
    name: str       # e.g. "aeroplane"
    color: str      # e.g. "#FF6B6B" (hex color code)


@dataclass(frozen=True)
class VOCColor:
    # inherit from ClassColor for VOC dataset specific colors
    aeroplane: ClassColor = ClassColor("aeroplane", "#FF6B6B")
    bicycle: ClassColor = ClassColor("bicycle", "#FFB86C")
    bird: ClassColor = ClassColor("bird", "#FFFF6B")
    boat: ClassColor = ClassColor("boat", "#6BFF6B")
    bottle: ClassColor = ClassColor("bottle", "#6BFFFF")
    bus: ClassColor = ClassColor("bus", "#6B6BFF")
    car: ClassColor = ClassColor("car", "#FF6BFF")
    cat: ClassColor = ClassColor("cat", "#FF8F6B")
    chair: ClassColor = ClassColor("chair", "#D36BFF")
    cow: ClassColor = ClassColor("cow", "#6BD3FF")
    diningtable: ClassColor = ClassColor("diningtable", "#FF6BD3")
    dog: ClassColor = ClassColor("dog", "#6BFF8F")
    horse: ClassColor = ClassColor("horse", "#8FFF6B")
    motorbike: ClassColor = ClassColor("motorbike", "#D3D36B")
    person: ClassColor = ClassColor("person", "#FFB86B")
    pottedplant: ClassColor = ClassColor("pottedplant", "#6BD36B")
    sheep: ClassColor = ClassColor("sheep", "#6BD3D3")
    sofa: ClassColor = ClassColor("sofa", "#6B6BD3")
    train: ClassColor = ClassColor("train", "#D36B6B")
    tvmonitor: ClassColor = ClassColor("tvmonitor", "#D3D36B")


@dataclass(frozen=True)
class COCOColor:
    

def to_numpy_boxes(boxes: list | np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert boxes to numpy array of shape (N, 4)."""
    arr = np.asarray(boxes, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("`boxes` must have shape (N, 4) with [xmin, ymin, xmax, ymax].")
    return arr


def get_voc_info(cid: int) -> ClassColor:
    """Get ClassColor info for a given class ID, with fallback for unknown IDs."""
    cid = int(cid)
    info = VOC_CLASSES.get(cid)
    if info is not None:
        return info
    raise ValueError(f"Unknown VOC class ID: {cid}")


def legend_handles_from_used(used: set[int]) -> list[Patch]:
    """Create legend handles for used class IDs, using VOC_CLASSES colors."""
    handles = []
    for cid in sorted(set(used)):
        info = get_voc_info(cid)
        handles.append(
            Patch(
                facecolor=info.color,   # color associated with the class
                edgecolor=info.color,   # color associated with the class
                alpha=0.8,
                label=info.name,        # class name
                linewidth=2,
            )
        )
    return handles


def draw_bbox(
    image: np.ndarray,
    boxes: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    scores: list | np.ndarray | torch.Tensor | None = None,
    conf_thr: float = 0.5,
    ax: Axes | None = None,
    title: str = "Bounding Boxs Visualization",
    figsize: tuple[int, int] = (12, 8),
) -> tuple[Figure, Axes, set[int]]:
    """
    Draw bounding boxes on the image with labels and optional scores.
    Returns the figure, axes, and set of used category IDs for legend.
    --------------------------------------------------------------------------
    The `boxes` should be in (N, 4) format with (xmin, ymin, xmax, ymax) coordinates.
    The `labels` should be of length N with class IDs corresponding to VOC_CLASSES.
    The `scores` (if provided) should be of length N with confidence scores.
    """    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()

    # Ensure image is in the correct format for display
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)

    ax.imshow(img)
    ax.axis("off")
    H, W = image.shape[:2]

    boxes_np = to_numpy_boxes(boxes)
    if len(labels) != len(boxes_np):
        raise ValueError("`labels` length must match number of `boxes`.")
    if scores is not None and len(scores) != len(boxes_np):
        raise ValueError("`scores` length must match number of `boxes` when provided.")

    # Draw each box with label and score
    used_categories = set()
    for i, (box, lab) in enumerate(zip(boxes_np, labels)):
        # Skip boxes below confidence threshold
        if scores is not None and float(scores[i]) < conf_thr:
            print(f"Skipping box {i} with score {scores[i]:.3f} below threshold {conf_thr:.3f}")
            continue

        cid = int(lab)
        info = get_voc_info(cid)

        # Extract box coordinates and convert to float
        xmin, ymin, xmax, ymax = [float(v) for v in box.tolist()]
        w, h = xmax - xmin, ymax - ymin

        # Clamp box coordinates to image boundaries
        xmin = max(0.0, min(W - 1.0, xmin))
        ymin = max(0.0, min(H - 1.0, ymin))
        w = max(1.0, min(W - xmin, w))
        h = max(1.0, min(H - ymin, h))

        color = info.color
        name = info.name

        # Draw rectangle for bounding box
        rect = Rectangle(
            (xmin, ymin),
            w, h, fill=False,
            linewidth=2.5,
            edgecolor=color,
            alpha=0.9,
        )
        ax.add_patch(rect)

        # Prepare label text with optional score
        score_txt = f" {scores[i]:.2f}" if scores is not None else ""
        label_text = f"{name}{score_txt}"

        # Draw background rectangle for text label and add text
        text_height = 14
        text_width = max(36, int(len(label_text) * 6.5))
        text_bg = Rectangle(
            (xmin, max(0.0, ymin - text_height)),
            text_width, text_height,
            fill=True, alpha=0.85,
            edgecolor=color, 
            facecolor=color,
        )
        ax.add_patch(text_bg)

        # Add text label above the box, with slight offset
        ax.text(
            xmin + 3,
            max(2, int(ymin) - 4),
            label_text, fontsize=9,
            color="white", weight="bold",
            alpha=0.95, va="top"
        )
        used_categories.add(cid)

    if used_categories:
        ax.legend(
            handles=legend_handles_from_used(used_categories),
            loc="upper right", fontsize=10, framealpha=0.9,
            fancybox=True, shadow=True
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=20, color="#2C3E50")
    ax.grid(False)
    plt.tight_layout()
    return fig, ax, used_categories


def show_bbox(
    images: list[np.ndarray],
    boxes: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    scores: list | None = None,
    titles: list[str] | None = None,
    cols: int = 4,
) -> tuple[Figure, np.ndarray]:
    """
    Show bounding boxes on multiple images in a grid.
    Returns the figure and array of axes.
    """
    # Determine grid size based on number of images and columns
    # it puts them into a grid with given number of columns and enough rows
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    fig, axes_grid = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.atleast_1d(axes_grid).ravel()

    all_used = set()
    last_idx = -1
    for i, (img, bxs, labs) in enumerate(zip(images, boxes, labels)):
        # Stop if we exceed available axes
        if i >= len(axes):
            break

        # Draw on the specific axis
        ax = axes[i]
        sc = scores[i] if scores is not None else None
        title = titles[i] if titles is not None else f"Image {i + 1}"

        # Draw bounding boxes on the image
        _, _, used = draw_bbox(
            image=img,
            boxes=bxs,
            labels=labs,
            scores=sc,
            ax=ax,
            title=title,
            figsize=(6, 4),
        )
        all_used.update(used)
        last_idx = i

    # No necessary axes turned off to see pixel units
    for j in range(last_idx + 1, len(axes)):
        axes[j].axis("off")

    # Add legend to the last used axis if there are any used categories
    if all_used and last_idx >= 0:
        handles = legend_handles_from_used(all_used)
        axes[last_idx].legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(handles), 5),
            framealpha=0.95,
            fontsize=10,
            fancybox=True,
            shadow=True,
        )

    fig.suptitle(
        f"Bboxes {num_images} image(s)",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    return fig, axes


def save_bbox_visualization(
    image: np.ndarray,
    boxes: list | np.ndarray | torch.Tensor,
    labels: list | np.ndarray | torch.Tensor,
    scores: list | np.ndarray | torch.Tensor | None = None,
    out_pth: str = "xxx.png",
    dpi: int = 300,
    **kwargs,
) -> None:
    """Make and save bounding box visualization to file."""
    fig, _, _ = draw_bbox(image, boxes, labels, scores=scores, **kwargs)
    fig.savefig(out_pth, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
