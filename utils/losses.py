"""Segmentation losses for BRMSNet."""

from typing import Optional

import torch
import torch.nn.functional as F


def _loss_inputs(logits, mask, valid_region):
    # Pixel sums at 512 x 512 can overflow FP16 under autocast.
    logits, mask = logits.float(), mask.float()
    valid_region = (
        torch.ones_like(mask) if valid_region is None else valid_region.float()
    )
    return logits, mask, valid_region


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    per_image = values.flatten(1).mean(dim=1)
    if reduction == "none":
        return per_image
    if reduction == "mean":
        return per_image.mean()
    raise ValueError("reduction must be 'none' or 'mean'")


def structure_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 1.0,
    valid_region: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Weighted BCE and IoU, with local averages excluding padding."""
    logits, mask, valid_region = _loss_inputs(logits, mask, valid_region)
    local_foreground = F.avg_pool2d(mask * valid_region, 31, stride=1, padding=15)
    local_valid = F.avg_pool2d(valid_region, 31, stride=1, padding=15)
    local_mean = local_foreground / (local_valid + 1e-6)
    weights = valid_region * (1 + 5 * (local_mean - mask).abs())

    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    weighted_bce = (weights * bce).sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + 1e-6)
    probability = logits.sigmoid()
    intersection = (weights * probability * mask).sum(dim=(2, 3))
    union = (weights * (probability + mask - probability * mask)).sum(dim=(2, 3))
    weighted_iou = 1 - (intersection + 1) / (union + 1)
    return _reduce(weight * (weighted_bce + weighted_iou), reduction)


def _soft_boundary(values: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    padding = kernel_size // 2
    dilation = F.max_pool2d(values, kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-values, kernel_size, stride=1, padding=padding)
    return (dilation - erosion).clamp(0, 1)


def boundary_dice_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid_region: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    logits, mask, valid_region = _loss_inputs(logits, mask, valid_region)
    interior = -F.max_pool2d(-valid_region, 5, stride=1, padding=2)
    prediction = _soft_boundary(logits.sigmoid() * valid_region) * interior
    target = _soft_boundary(mask * valid_region) * interior
    intersection = (prediction * target).sum(dim=(2, 3))
    denominator = prediction.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + 1) / (denominator + 1)
    return _reduce(1 - dice.clamp(0, 1), reduction)


def focal_tversky_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    valid_region: Optional[torch.Tensor] = None,
    false_positive_weight: float = 0.3,
    false_negative_weight: float = 0.7,
    gamma: float = 4.0 / 3.0,
    reduction: str = "mean",
) -> torch.Tensor:
    logits, mask, valid_region = _loss_inputs(logits, mask, valid_region)
    probability = logits.sigmoid()
    true_positive = (probability * mask * valid_region).sum(dim=(2, 3))
    false_positive = (probability * (1 - mask) * valid_region).sum(dim=(2, 3))
    false_negative = ((1 - probability) * mask * valid_region).sum(dim=(2, 3))
    denominator = (
        true_positive
        + false_positive_weight * false_positive
        + false_negative_weight * false_negative
        + 1
    )
    tversky = ((true_positive + 1) / denominator).clamp(0, 1)
    return _reduce((1 - tversky).pow(gamma), reduction)


def segmentation_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    boundary_weight: float = 0.5,
    tversky_weight: float = 0.3,
    valid_region: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    losses = (
        structure_loss(logits, mask, valid_region=valid_region, reduction="none")
        + boundary_weight * boundary_dice_loss(logits, mask, valid_region, "none")
        + tversky_weight
        * focal_tversky_loss(logits, mask, valid_region, reduction="none")
    )
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction must be 'none' or 'mean'")
