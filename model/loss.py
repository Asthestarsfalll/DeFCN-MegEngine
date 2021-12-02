import megengine.distributed as dist
import megengine.functional as F
from megengine import Tensor


def focal_loss(
        logits: Tensor, targets: Tensor, alpha: float = -1, gamma: float = 2, with_logits=True
) -> Tensor:
    ce_loss = F.nn.binary_cross_entropy(
        logits, targets, with_logits=with_logits, reduction='none')
    p_t = logits * targets + (1 - logits) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss *= (targets * alpha + (1 - targets) * (1 - alpha))

    return loss.mean()


def iou_loss(
        pred: Tensor, target: Tensor, box_mode: str = "xyxy", loss_type: str = "iou", eps: float = 1e-8,
) -> Tensor:
    if box_mode == "ltrb":
        pred = F.concat([-pred[..., :2], pred[..., 2:]], axis=-1)
        target = F.concat([-target[..., :2], target[..., 2:]], axis=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    pred_area = F.maximum(pred[..., 2] - pred[..., 0], 0) * F.maximum(
        pred[..., 3] - pred[..., 1], 0
    )
    target_area = F.maximum(target[..., 2] - target[..., 0], 0) * F.maximum(
        target[..., 3] - target[..., 1], 0
    )

    w_intersect = F.maximum(
        F.minimum(pred[..., 2], target[..., 2]) -
        F.maximum(pred[..., 0], target[..., 0]), 0
    )
    h_intersect = F.maximum(
        F.minimum(pred[..., 3], target[..., 3]) -
        F.maximum(pred[..., 1], target[..., 1]), 0
    )

    area_intersect = w_intersect * h_intersect
    area_union = pred_area + target_area - area_intersect
    ious = area_intersect / F.maximum(area_union, eps)

    if loss_type == "iou":
        loss = -F.log(F.maximum(ious, eps))
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = F.maximum(pred[..., 2], target[..., 2]) - F.minimum(
            pred[..., 0], target[..., 0]
        )
        g_h_intersect = F.maximum(pred[..., 3], target[..., 3]) - F.minimum(
            pred[..., 1], target[..., 1]
        )
        ac_union = g_w_intersect * g_h_intersect
        gious = ious - (ac_union - area_union) / F.maximum(ac_union, eps)
        loss = 1 - gious
    return loss


def pairwise_iou(boxes1: Tensor, boxes2: Tensor, return_ioa=False) -> Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1 (Tensor): boxes tensor with shape (N, 4)
        boxes2 (Tensor): boxes tensor with shape (M, 4)
        return_ioa (Bool): wheather return Intersection over Boxes1 or not, default: False

    Returns:
        iou (Tensor): IoU matrix, shape (N,M).
    """
    b_box1 = F.expand_dims(boxes1, axis=1)
    b_box2 = F.expand_dims(boxes2, axis=0)

    iw = F.minimum(b_box1[:, :, 2], b_box2[:, :, 2]) - F.maximum(
        b_box1[:, :, 0], b_box2[:, :, 0]
    )
    ih = F.minimum(b_box1[:, :, 3], b_box2[:, :, 3]) - F.maximum(
        b_box1[:, :, 1], b_box2[:, :, 1]
    )
    inter = F.maximum(iw, 0) * F.maximum(ih, 0)

    area_box1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_box2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = F.expand_dims(area_box1, axis=1) + \
        F.expand_dims(area_box2, axis=0) - inter
    overlaps = F.maximum(inter / union, 0)

    if return_ioa:
        ioa = F.maximum(inter / area_box1, 0)
        return overlaps, ioa

    return overlaps
