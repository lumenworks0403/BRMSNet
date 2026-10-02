"""Shared post-processing and binary-mask metrics for RSOD."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ModelOutput = Union[torch.Tensor, Sequence[torch.Tensor]]


def primary_logits(model_output: ModelOutput) -> torch.Tensor:
    """Return the full-resolution logits from a model output contract."""

    if isinstance(model_output, torch.Tensor):
        return model_output
    if not model_output:
        raise ValueError("Model returned an empty output sequence")
    return model_output[0]


def normalized_saliency_map(
    logits: torch.Tensor,
    output_size: Tuple[int, int],
    content_box: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Resize one CHW/BCHW logit map and normalize its sigmoid to [0, 1]."""

    probabilities = resized_probability_map(logits, output_size, content_box)
    value_range = probabilities.max() - probabilities.min()
    return (probabilities - probabilities.min()) / (value_range + 1e-8)


def resized_probability_map(
    logits: torch.Tensor,
    output_size: Tuple[int, int],
    content_box: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Crop letterbox padding, resize logits, and return sigmoid probabilities."""

    if logits.ndim == 3:
        logits = logits.unsqueeze(0)
    if logits.ndim != 4:
        raise ValueError(
            f"Expected CHW or BCHW logits, got shape {tuple(logits.shape)}"
        )

    if content_box is not None:
        top, left, bottom, right = (int(value) for value in content_box)
        if not (0 <= top < bottom <= logits.shape[-2]):
            raise ValueError(f"Invalid vertical content box: {content_box}")
        if not (0 <= left < right <= logits.shape[-1]):
            raise ValueError(f"Invalid horizontal content box: {content_box}")
        logits = logits[..., top:bottom, left:right]

    resized = F.interpolate(
        logits,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.sigmoid()[0, 0]


def boundary_f1_score(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int = 2,
) -> float:
    """Boundary F1 with a two-pixel matching tolerance."""

    prediction = prediction.astype(np.uint8)
    target = target.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    prediction_edge = prediction - cv2.erode(prediction, kernel, iterations=1)
    target_edge = target - cv2.erode(target, kernel, iterations=1)
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * tolerance + 1, 2 * tolerance + 1),
    )
    target_dilated = cv2.dilate(target_edge, dilation_kernel)
    prediction_dilated = cv2.dilate(prediction_edge, dilation_kernel)

    predicted_count = int(prediction_edge.sum())
    target_count = int(target_edge.sum())
    if predicted_count == 0 and target_count == 0:
        return 1.0
    if predicted_count == 0 or target_count == 0:
        return 0.0
    precision = float((prediction_edge * target_dilated).sum()) / predicted_count
    recall = float((target_edge * prediction_dilated).sum()) / target_count
    return 2.0 * precision * recall / (precision + recall + 1e-8)


def read_binary_mask(mask_path: Union[str, Path]) -> np.ndarray:
    """Read a 0/1 or 0/255 mask and return a uint8 binary array."""

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read ground-truth mask: {mask_path}")
    threshold = 0 if int(mask.max()) <= 1 else 128
    return (mask > threshold).astype(np.uint8)


def dice_score(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the binary Dice coefficient over all pixels."""

    prediction = prediction.float()
    target = target.to(device=prediction.device, dtype=torch.float32)
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (prediction_flat * target_flat).sum()
    return (2.0 * intersection + 1e-6) / (
        prediction_flat.sum() + target_flat.sum() + 1e-6
    )


def intersection_over_union(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute binary intersection over union over all pixels."""

    prediction = prediction.float()
    target = target.to(device=prediction.device, dtype=torch.float32)
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (prediction_flat * target_flat).sum()
    union = prediction_flat.sum() + target_flat.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)
