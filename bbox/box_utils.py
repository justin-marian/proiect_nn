from __future__ import annotations

import torch

EPS = 1e-6


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # (N,M,2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # (N,M,2)

    wh = (rb - lt).clamp(min=0)  # (N,M,2)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + EPS)
