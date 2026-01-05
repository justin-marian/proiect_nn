import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def imshow(
    img: torch.Tensor,
    title: str = "Image",
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    ax=None,
) -> None:
    """
    Show a single image tensor, with optional de-normalization.
    The image tensor should be in (C, H, W) format.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    img = img.clone().cpu()

    # Optional de-normalization
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
        img = img * std_t + mean_t

    img = img.clamp(0, 1)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # (H, W, C)

    ax.imshow(npimg)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def make_grid(
    images: torch.Tensor,
    labels: list[str],
    boxes: list,                    # list of (Ni, 4) [xmin, ymin, xmax, ymax] per image - required
    box_labels: list | None = None,    # list of list[str] per image - optional
    box_scores: list | None = None,    # list of (Ni,) scores per image - optional
    title: str = "Detection Grid",
    out_path: str = "output/xxx.png",
    mean: list[float] | None = None,   # from stats_mean_std
    std: list[float] | None = None,    # from stats_mean_std
    pred_status: list[bool] | None = None
) -> None:
    """
    Create and save a grid of images for *object detection*
    (with optional classification check) from a batch of images.
    Each image shows its boxes, optional per-box labels/scores, and an overall title.
    If `pred_status` is given (list of bool per image), the title and border
    colors are set to green (correct) or red (incorrect) accordingly.
    ----------------------------------------------------------------------------
    It shows is a grid the given batch of images with their predicted boxes.
    Each image shows its boxes, optional per-box labels/scores, and an overall title.
    for object detection, with optional classification check.
    ----------------------------------------------------------------------------
    """
    def color_pred(idx: int) -> tuple[str, str]:
        """
        Returns (title_color, border_color) based on pred_status.
        Defaults to black if pred_status is None.
        """
        if pred_status is not None and idx < len(pred_status):
            ok = bool(pred_status[idx])
            color = "green" if ok else "red"
            return color, color
        return "black", "black"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    no_images = images.size(0)
    if no_images == 0:
        return

    # Basic sanity check: boxes length must match number of images
    if len(boxes) != no_images:
        raise ValueError(
            f"`boxes` must have length {no_images} (one entry per image), "
            f"but got {len(boxes)}."
        )

    # De-normalize images for display if mean/std are given
    imgs_disp = images.clone().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(1, 3, 1, 1)
        std_t = torch.tensor(std).view(1, 3, 1, 1)
        imgs_disp = imgs_disp * std_t + mean_t
    imgs_disp = imgs_disp.clamp(0, 1)

    # Determine grid size
    ncols = min(4, no_images)
    nrows = (no_images + ncols - 1) // ncols

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if isinstance(axes_grid, np.ndarray):
        axes = axes_grid.ravel()
    else:
        axes = np.array([axes_grid])

    for img_idx in range(nrows * ncols):
        ax = axes[img_idx]
        ax.axis("off")

        # Skip if exceeding available images
        if img_idx >= no_images:
            continue

        # Show image
        img = imgs_disp[img_idx].numpy()
        img = np.transpose(img, (1, 2, 0))
        ax.imshow(img)

        title_color, border_color = color_pred(img_idx)

        # Draw border based on pred_status if given
        if pred_status is not None and img_idx < len(pred_status):
            for spine in ax.spines.values():
                spine.set_color(border_color)
                spine.set_linewidth(3)

        # Object detection: draw boxes, labels, scores
        img_boxes = boxes[img_idx]
        img_boxes = np.asarray(img_boxes, dtype=float)
        H, W = img.shape[:2]

        # Optional per-box labels/scores for this image
        per_box_labels = (
            box_labels[img_idx]
            if box_labels is not None and img_idx < len(box_labels)
            else None
        )
        per_box_scores = (
            box_scores[img_idx]
            if box_scores is not None and img_idx < len(box_scores)
            else None
        )

        for b_idx, box in enumerate(img_boxes):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            x1 = max(0.0, min(W - 1.0, x1))
            y1 = max(0.0, min(H - 1.0, y1))
            x2 = max(0.0, min(W - 1.0, x2))
            y2 = max(0.0, min(H - 1.0, y2))
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)

            # Light blue box for detections
            rect = Rectangle(
                (x1, y1), w, h,
                fill=False, linewidth=2,
                edgecolor="#00BFFF",
                alpha=0.9,
            )
            ax.add_patch(rect)

            # Build caption "label score"
            if per_box_labels is not None and b_idx < len(per_box_labels):
                txt = str(per_box_labels[b_idx])
                if per_box_scores is not None and b_idx < len(per_box_scores):
                    txt += f" {float(per_box_scores[b_idx]):.2f}"

                # Rounded box with text, used as background
                ax.text(
                    x1 + 2, max(10, y1 + 2),
                    txt, fontsize=8,
                    color="white", weight="bold",
                    bbox=dict(
                        facecolor="#00BFFF",
                        edgecolor="none",
                        boxstyle="round,pad=0.15",
                        alpha=0.8,
                    ),
                )

        # Per-image title (classification / summary)
        this_label = labels[img_idx] if img_idx < len(labels) else f"Image {img_idx}"
        ax.set_title(this_label, fontsize=10, color=title_color)

    # Turn off any extra axes (if grid is bigger than number of images)
    for img_idx in range(no_images, nrows * ncols):
        axes[img_idx].axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
