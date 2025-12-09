import torch
from dataclasses import dataclass


@dataclass
class Anchors:
    """
    Anchors for a single image.
    ------------------------------------------------------------------------------
    The correlation between spatial locations in the feature map and the anchors
    is determined by the anchor generator, the correlation between anchors and
    boxes is determined by the box regression head of the model.
    ------------------------------------------------------------------------------
    Anchors are transformed into boxes during training/inference for matching
    with ground-truth boxes / predicted boxes, The purpose of anchors is to
    provide a fixed set of reference boxes at each spatial location in the
    feature map, so that the model can predict offsets relative to these anchors.
    ------------------------------------------------------------------------------
    Thus, there are two key mappings:
    feature map locations <-> anchors: via anchor generation (grid + sizes/ratios)
    anchors <-> boxes: via box transformations (offsets) predicted by the model
    """
    boxes: torch.Tensor             # (N, 4) in (x1, y1, x2, y2) format in image coords
    image_size: tuple[int, int]     # (H, W) of the image these anchors correspond to
    num_anchors_per_location: int   # no. of anchors per spatial location

    def to(self, device) -> "Anchors":
        return Anchors(
            boxes=self.boxes.to(device),
            image_size=self.image_size,
            num_anchors_per_location=self.num_anchors_per_location,
        )


class AnchorGenerator:
    def __init__(
        self,
        sizes: list[list[float]],
        aspect_ratios: list[list[float]],
        strides: list[int],
    ) -> None:
        assert len(sizes) == len(aspect_ratios) == len(strides), \
            "sizes, aspect_ratios, strides must have the same length"
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.base_anchors = self.generate_base_anchors()

    def generate_base_anchors(self) -> list[torch.Tensor]:
        """
        Generate base anchors centered at (0, 0) for each FPN level.
        Returns a list of tensors, each of shape (A, 4) where A is the
        number of anchors per location for that level - total no. of (size, ratio) combinations.
        """
        base_anchors: list[torch.Tensor] = []
        # For each FPN level (feature pyramid network level - corresponds to a certain stride)
        for sizes_per_level, ratios_per_level in zip(self.sizes, self.aspect_ratios):
            # Generate anchors for this level (centered at origin)
            anchors_per_level = []
            for size in sizes_per_level:
                # For each size and aspect ratio, compute anchor box coordinates
                area = float(size * size)
                for ratio in ratios_per_level:
                    # width and height based on area and aspect ratio, centered at (0,0)
                    w = torch.sqrt(torch.tensor(area / ratio))
                    h = w * ratio
                    # (x1, y1, x2, y2) format centered at (0,0)
                    x1, y1 = -0.5 * w, -0.5 * h
                    x2, y2 = +0.5 * w, +0.5 * h
                    anchors_per_level.append(torch.tensor([x1, y1, x2, y2]))
            base_anchors.append(torch.stack(anchors_per_level, dim=0))  # (A, 4)
        return base_anchors

    def num_anchors_per_location(self) -> list[int]:
        """Number of anchors per spatial location for each FPN level."""
        return [base.shape[0] for base in self.base_anchors]

    def generate_anchors(
        self,
        feature_shapes: list[tuple[int, int]],
        device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        """
        Generate anchors for each feature map level.
        feature_shapes: list of (H, W) for each level.
        """
        if device is None:
            device = torch.device("cpu")

        # Stop proceeding if mismatch in number of levels
        assert len(feature_shapes) == len(self.base_anchors), \
            "feature_shapes length must match number of anchor levels for the generator"

        # Generate anchors for each level based on feature map size and stride
        # Then combine with base anchors to get final anchor boxes in image coords
        anchors_per_level: list[torch.Tensor] = []
        for (feat_h, feat_w), stride, base_anchors in zip(feature_shapes, self.strides, self.base_anchors):
            base_anchors = base_anchors.to(device)
            A = base_anchors.size(0)

            # Grid of center points on feature map (scaled to image space)
            # Each point corresponds to the center of an anchor box on the image
            shifts_x = (torch.arange(0, feat_w, device=device) + 0.5) * stride
            shifts_y = (torch.arange(0, feat_h, device=device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

            # Flatten the grid of shifts to create anchor centers
            # Each shift corresponds to the center of an anchor box on the image
            shift_x = shift_x.reshape(-1)  # (H*W,)
            shift_y = shift_y.reshape(-1)  # (H*W,)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # Combine base anchors with shifts to get final anchor boxes
            # Add the shifts to the base anchors to move them to their correct positions on the image
            anchors = base_anchors.reshape(1, A, 4) + shifts.reshape(-1, 1, 4)  # (H*W, A, 4)
            anchors = anchors.reshape(-1, 4)  # (H*W*A, 4)
            anchors_per_level.append(anchors)

        return anchors_per_level
