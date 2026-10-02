"""Training-only quality weighting and gated cross-scale guidance."""

from typing import NamedTuple, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from utils.losses import segmentation_loss


class SupervisionResult(NamedTuple):
    loss: torch.Tensor
    weights: torch.Tensor
    gates: torch.Tensor


class QAMWS(nn.Module):
    """Quality-aware supervision; the first prediction is the main head."""

    def __init__(
        self,
        temperature: float = 0.2,
        weight_floor: float = 0.2,
        multi_scale_weight: float = 0.6,
        mixture_weight: float = 0.1,
        warmup_epochs: int = 10,
        boundary_weight: float = 0.5,
        tversky_weight: float = 0.3,
    ) -> None:
        super().__init__()
        if temperature <= 0 or not 0 < weight_floor < 1:
            raise ValueError("temperature must be positive and weight_floor in (0, 1)")
        if (
            warmup_epochs < 0
            or min(multi_scale_weight, mixture_weight, boundary_weight, tversky_weight)
            < 0
        ):
            raise ValueError("warmup_epochs and loss weights cannot be negative")
        self.temperature = temperature
        self.weight_floor = weight_floor
        self.multi_scale_weight = multi_scale_weight
        self.mixture_weight = mixture_weight
        self.warmup_epochs = warmup_epochs
        self.boundary_weight = boundary_weight
        self.tversky_weight = tversky_weight

    def _segmentation_loss(self, logits, mask, valid_region):
        return segmentation_loss(
            logits,
            mask,
            self.boundary_weight,
            self.tversky_weight,
            valid_region,
            reduction="none",
        )

    def forward(
        self,
        head_logits: Sequence[torch.Tensor],
        mask: torch.Tensor,
        valid_region: Optional[torch.Tensor] = None,
        epoch: int = 1,
    ) -> SupervisionResult:
        if len(head_logits) != 4:
            raise ValueError("QAMWS requires four heads, with the main head first")
        if epoch < 1:
            raise ValueError("epoch is one-based")
        valid_region = (
            torch.ones_like(mask) if valid_region is None else valid_region.float()
        )
        aligned = [
            logits.float()
            if logits.shape[-2:] == mask.shape[-2:]
            else F.interpolate(
                logits.float(),
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for logits in head_logits
        ]
        losses = torch.stack(
            [self._segmentation_loss(logits, mask, valid_region) for logits in aligned],
            dim=1,
        )
        quality = torch.exp(-losses.detach())
        if epoch <= self.warmup_epochs:
            weights = torch.full_like(losses, 0.25)
        else:
            weights = self.weight_floor / 4 + (1 - self.weight_floor) * torch.softmax(
                quality / self.temperature, dim=1
            )
        total = losses[:, 0] + self.multi_scale_weight * (weights * losses).sum(dim=1)
        gates = torch.zeros_like(weights)

        if epoch > self.warmup_epochs and self.mixture_weight > 0:
            probability = torch.stack([logits.sigmoid() for logits in aligned], dim=1)
            # Quality, mixture targets and gates never participate in backpropagation.
            with torch.no_grad():
                teacher = (weights[:, :, None, None, None] * probability).sum(dim=1)
                teacher = teacher.clamp(1e-6, 1 - 1e-6)
                teacher_logits = teacher.log() - torch.log1p(-teacher)
                teacher_quality = torch.exp(
                    -self._segmentation_loss(teacher_logits, mask, valid_region)
                )
                gates = (teacher_quality[:, None] - quality).clamp_min(0)
            student = probability.clamp(1e-6, 1 - 1e-6)
            teacher = teacher[:, None]
            divergence = teacher * (teacher.log() - student.log()) + (1 - teacher) * (
                torch.log1p(-teacher) - torch.log1p(-student)
            )
            valid_pixels = valid_region.sum(dim=(1, 2, 3)) + 1e-6
            per_head_kl = (divergence * valid_region[:, None]).sum(dim=(2, 3, 4))
            per_head_kl = (per_head_kl / valid_pixels[:, None]).clamp_min(0)
            total = total + self.mixture_weight * (gates * per_head_kl).mean(dim=1)

        return SupervisionResult(total.mean(), weights.detach(), gates.detach())
