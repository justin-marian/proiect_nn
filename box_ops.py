import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING


# only for type hints to avoid circular imports
# these imports are used in type annotations only
if TYPE_CHECKING:
    from anchors import Anchors


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """Box area: product of width and height."""
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU = intersection over union of two sets of boxes."""
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    # intersection boxes (left top and right bottom)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # (N, M, 2)
    # width and height of intersection boxes
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]
    # union = area1 + area2 - intersection
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou


def clip_boxes_to_image(boxes: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Clip boxes to image boundaries, in-place, in case boxes go out of bounds."""
    x1 = boxes[:, 0].clamp(0, w - 1)
    y1 = boxes[:, 1].clamp(0, h - 1)
    x2 = boxes[:, 2].clamp(0, w - 1)
    y2 = boxes[:, 3].clamp(0, h - 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


@dataclass
class BoxList:
    """
    Container for bounding boxes and associated info.
    This makes it easier to pass boxes, scores, and labels through the
    WSOD -> PGE/PGA -> Pseudo Labels -> Student-Teacher + EMA pipeline.
    Here are the fields:
        boxes:  (N, 4) tensor of are different from anchors they have (x1, y1, x2, y2)
        format compared to anchors which are (cx, cy, w, h) format
        scores: (N,) tensor of confidence scores
        labels: (N,) tensor of class indices (int64)
        image_size: (H, W) tuple of image size
    """
    boxes: torch.Tensor          # (N, 4) boxes in (x1, y1, x2, y2) format
    labels: torch.Tensor         # (N,) int64, class index
    scores: torch.Tensor | None  # (N,) it can be None for unlabeled boxes or pseudo-GT
    image_size: tuple[int, int]  # (H, W) image size

    def to(self, device) -> "BoxList":
        return BoxList(
            boxes=self.boxes.to(device),
            scores=self.scores.to(device) if self.scores is not None else None,
            labels=self.labels.to(device),
            image_size=self.image_size,
        )


def encode_boxes(
    anchors: torch.Tensor,  # (N, 4)
    gt_boxes: torch.Tensor,  # (N, 4)
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """From (x1, y1, x2, y2) boxes to encoded deltas w.r.t. anchors."""
    wx, wy, ww, wh = weights
    # anchor centers, widths, heights
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    # gt centers, widths, heights
    gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) * 0.5
    gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) * 0.5
    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    # encode deltas
    dx = wx * (gx - ax) / (aw + 1e-6)
    dy = wy * (gy - ay) / (ah + 1e-6)
    dw = ww * torch.log(gw / (aw + 1e-6) + 1e-6)
    dh = wh * torch.log(gh / (ah + 1e-6) + 1e-6)
    # stack is for to gether in one tensor all deltas
    # deltas from anchors to gt boxes defined as (dx, dy, dw, dh)
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(
    anchors: torch.Tensor,  # (N, 4)
    deltas: torch.Tensor,   # (N, 4)
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """From encoded deltas back to (x1, y1, x2, y2) boxes."""
    wx, wy, ww, wh = weights
    # anchor centers, widths, heights
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    # decode deltas
    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh
    # predicted box centers, widths, heights
    px = dx * aw + ax
    py = dy * ah + ay
    pw = torch.exp(dw) * aw
    ph = torch.exp(dh) * ah
    # convert to (x1, y1, x2, y2)
    x1, y1 = px - 0.5 * pw, py - 0.5 * ph
    x2, y2 = px + 0.5 * pw, py + 0.5 * ph
    # stack to get all boxes in one tensor
    # return boxes as (x1, y1, x2, y2)
    return torch.stack([x1, y1, x2, y2], dim=1)


def match_anchors_to_gt(
    anchors: torch.Tensor,          # (N_a, 4)
    gt: BoxList,                    # pseudo-GT boxes
    high_thresh: float = 0.7,       # IoU >= high_thresh -> positive
    low_thresh: float = 0.3,        # IoU < low_thresh -> negative
    allow_low_quality_matches: bool = True,  # there must be at least one match per GT
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Match anchors to ground-truth (pseudo-GT) boxes using IoU thresholds.
    Returns:
        labels:          (N_a,) int64, -1 ignored, 0 bg, >0 class index
        bbox_targets:    (N_a, 4) encoded deltas (only valid for labels > 0)
        matched_gt_inds: (N_a,) index into gt.boxes for positives, -1 for bg/ignored
    """
    device = anchors.device
    N_a = anchors.size(0)

    # no objects: all background, zero targets
    if gt.boxes.numel() == 0:
        raise ValueError("No ground-truth boxes provided for matching.")

    # IoU between anchors and GT boxes
    ious = box_iou(anchors, gt.boxes)  # (N_a, N_gt)
    max_iou, argmax_iou = ious.max(dim=1)  # per anchor

    # Assign labels based on IoU thresholds
    labels = torch.full((N_a,), -1, dtype=torch.long, device=device)
    labels[max_iou < low_thresh] = 0
    labels[max_iou >= high_thresh] = gt.labels[argmax_iou[max_iou >= high_thresh]]

    # Each GT box should have at least one matched anchor
    if allow_low_quality_matches:
        # ensure each GT has at least one anchor (the best one)
        _, gt_argmax_iou = ious.max(dim=0)
        for gt_idx in range(gt.boxes.size(0)):
            anchor_idx = gt_argmax_iou[gt_idx]
            labels[anchor_idx] = gt.labels[gt_idx]
            argmax_iou[anchor_idx] = gt_idx

    # Regression targets (only meaningful for positives)
    matched_gt_inds = argmax_iou.clone()
    matched_gt_inds[labels <= 0] = -1

    # Regression targets (only meaningful for positives)
    bbox_targets = torch.zeros(N_a, 4, device=device)
    pos_mask = labels > 0
    if pos_mask.any():
        pos_anchors = anchors[pos_mask]
        pos_gt = gt.boxes[matched_gt_inds[pos_mask]]
        bbox_targets[pos_mask] = encode_boxes(pos_anchors, pos_gt)

    return labels, bbox_targets, matched_gt_inds


def anchors_to_boxlist(
    anchors: "Anchors",
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> BoxList:
    """
    Wrap an Anchors object into a BoxList with given scores and labels.
    The purpose of this is to convert from Anchors to BoxList for further processing.
    """
    return BoxList(
        boxes=anchors.boxes,
        scores=scores,
        labels=labels,
        image_size=anchors.image_size,
    )


def match_anchors_object_to_gt(
    anchors: "Anchors",
    gt: BoxList,
    high_thresh: float = 0.7,
    low_thresh: float = 0.3,
    allow_low_quality_matches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Wrap an Anchors object to match it to ground-truth BoxList.
    The purpose of this is to convert from Anchors to BoxList for further processing.
    """
    return match_anchors_to_gt(
        anchors.boxes, gt,
        high_thresh=high_thresh,
        low_thresh=low_thresh,
        allow_low_quality_matches=allow_low_quality_matches,
    )


def filter_by_image_labels(
    boxlist: BoxList,
    image_labels: torch.Tensor,
) -> BoxList:
    """
    Keep only boxes whose class is present in the image-level label set.
    This is the key WSOD step: we know which classes are present in the image
    (multi-hot vector), but we do not know locations. Any predicted box whose
    label is not present in `image_labels` is discarded.
    """
    # Present classes in the image - multi-hot vector
    present = torch.nonzero(image_labels > 0).squeeze(1)  # (K,)
    # Mask is just to filter boxes whose labels are in `present` set
    # Maybe not all classes are present in the image - we only keep boxes of present classes
    mask = torch.isin(boxlist.labels, present.to(boxlist.labels.device))
    return BoxList(
        boxes=boxlist.boxes[mask],
        scores=boxlist.scores[mask] if boxlist.scores is not None else None,
        labels=boxlist.labels[mask],
        image_size=boxlist.image_size,
    )


def aggregate_boxes(
    boxlist: BoxList,
    iou_merge_thresh: float = 0.5,
    min_group_size: int = 1,
) -> BoxList:
    """
    Merge highly-overlapping boxes (of the same class) into one pseudo-GT box
    by weighted averaging with their scores.
    "PGE + PGA -> less noisy labels -> LA"
    """
    boxes = boxlist.boxes
    labels = boxlist.labels
    scores = boxlist.scores.to(dtype=torch.float32) if boxlist.scores is not None else None

    # Nothing to aggregate, it didn't make sense to proceed
    # it detected no boxes at all
    if boxes.numel() == 0:
        print("No boxes to aggregate; returning empty BoxList.")
        return boxlist

    # If we aggregate boxes class-wise, we can process each class separately
    # then combine results at the end, so we avoid merging boxes of different classes
    agg_boxes = []
    agg_scores = []
    agg_labels = []

    for c in labels.unique():
        # Process each class separately and aggregate boxes of that class
        # get indices of boxes of class c in the original boxlist
        cls_idx = torch.nonzero(labels == c).squeeze(1)
        cls_boxes = boxes[cls_idx]  # (N_c, 4)
        cls_scores = scores[cls_idx] if scores is not None else None  # (N_c,)

        # No boxes of this class, skip it
        # it is a rare case, but possible if no boxes of class c were detected, problem
        if cls_boxes.size(0) == 0:
            print(f"No boxes of class {c.item()} to aggregate; skipping.")
            continue

        # Keep track of which boxes have been assigned to a cluster already, to avoid double counting
        # we will iterate over boxes, and for each unassigned box, find all boxes that overlap with it enough
        assigned = torch.zeros(cls_boxes.size(0), dtype=torch.bool, device=cls_boxes.device)
        for i in range(cls_boxes.size(0)):
            # Already assigned to a cluster, skip it
            if assigned[i]:
                # print(f"Box {i} of class {c.item()} already assigned; skipping.")
                continue

            # Find all boxes that have high IoU with box i (to form a cluster)
            ref_box = cls_boxes[i].unsqueeze(0)  # (1,4)
            ious = box_iou(ref_box, cls_boxes)[0]  # (N,)

            # Determine which boxes form a cluster with box i based on IoU threshold
            cluster_mask = ious >= iou_merge_thresh
            cluster_idx = torch.nonzero(cluster_mask).squeeze(1)
            assigned[cluster_idx] = True

            # treat as noisy, drop this tiny cluster
            if cluster_idx.numel() < min_group_size:
                # print(f"Cluster of class {c.item()} too small ({cluster_idx.numel()} boxes); skipping.")
                continue

            # Aggregate boxes in the cluster by weighted averaging with their scores
            cluster_boxes = cls_boxes[cluster_idx]
            if cls_scores is not None:
                cluster_scores = cls_scores[cluster_idx]
            else: 
                cluster_scores = torch.ones(cluster_boxes.size(0), device=cluster_boxes.device)

            # score-weighted average of corners
            w = cluster_scores / (cluster_scores.sum() + 1e-6)
            x1 = (cluster_boxes[:, 0] * w).sum()
            y1 = (cluster_boxes[:, 1] * w).sum()
            x2 = (cluster_boxes[:, 2] * w).sum()
            y2 = (cluster_boxes[:, 3] * w).sum()

            # Append aggregated box at the end of the lists for this class
            # It finishes processing all clusters of this class, then moves to next class
            agg_boxes.append(torch.stack([x1, y1, x2, y2]))
            agg_scores.append(cluster_scores.mean())
            agg_labels.append(c)

    # All considered too noisy - drop everything itself is rare, but possible
    # It didn't make sense to proceed further, always there should be at least one box kept
    if not agg_boxes:
        raise ValueError("No boxes kept after aggregation... something went wrong.")

    agg_boxes = torch.stack(agg_boxes, dim=0)
    agg_labels = torch.stack(agg_labels, dim=0)
    agg_scores = torch.stack(agg_scores, dim=0)
    order = agg_scores.argsort(descending=True)

    return BoxList(
        boxes=agg_boxes[order],
        scores=agg_scores[order],
        labels=agg_labels[order],
        image_size=boxlist.image_size,
    )
