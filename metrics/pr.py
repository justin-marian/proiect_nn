import torch


def avg_prec(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """
    Average Precision (AP):
    - area under the P-R curve.
    - using a number of interpolation points.
    - it is assumed that recall and precision are 1D tensors of the same length.
    (the scores associated with the points are not needed for AP computation)
    """
    order = torch.argsort(recall)
    recall = recall[order]
    precision = precision[order]

    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

    idx = (mrec[1:] != mrec[:-1]).nonzero(as_tuple=False).squeeze()
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)
