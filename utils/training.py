"""Small training utilities for the RSOD pipeline."""

from __future__ import annotations

from collections import deque
from logging import Logger
from typing import Deque, Dict, Tuple, Union

import torch

Scalar = Union[float, torch.Tensor]


def clip_gradients(
    optimizer: torch.optim.Optimizer,
    max_value: float,
) -> Tuple[bool, float]:
    """Clip finite gradients and report overflow without modifying bad grads."""

    if max_value <= 0:
        raise ValueError("max_value must be positive")
    parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
        if parameter.grad is not None
    ]
    if not parameters:
        return True, 0.0

    parameter_norms = [
        parameter.grad.detach().float().norm(2) for parameter in parameters
    ]
    total_norm = torch.stack(parameter_norms).norm(2)
    if not torch.isfinite(total_norm).item():
        return False, float(total_norm.detach().cpu())

    torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=max_value,
        error_if_nonfinite=False,
    )
    return True, float(total_norm.detach().cpu())


class AverageMeter:
    """Track a weighted average and a short moving window."""

    def __init__(self, window_size: int = 40) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        self.value: Scalar = 0.0
        self.average: Scalar = 0.0
        self.total: Scalar = 0.0
        self.count = 0
        self._recent: Deque[torch.Tensor] = deque(maxlen=self.window_size)

    def update(self, value: Scalar, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be at least 1")
        detached = (
            value.detach() if isinstance(value, torch.Tensor) else torch.tensor(value)
        )
        self.value = detached
        self.total = self.total + detached * count
        self.count += count
        self.average = self.total / self.count
        self._recent.append(detached)

    def moving_average(self) -> torch.Tensor:
        if not self._recent:
            raise RuntimeError("AverageMeter has no observations")
        return torch.stack(list(self._recent)).mean()

    # Original API retained for older call sites.
    def show(self) -> torch.Tensor:
        return self.moving_average()


def profile_model(
    model: torch.nn.Module, image_size: int, logger: Logger
) -> Dict[str, float]:
    """Profile inference with THOP; count one MAC as two FLOPs."""

    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Model profiling requires THOP: pip install thop") from exc

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            operations, profiled_parameters = profile(
                model, inputs=(sample,), verbose=False
            )
    finally:
        model.train(was_training)
        # THOP analysis buffers must not leak into saved checkpoints.
        for module in model.modules():
            module._buffers.pop("total_ops", None)
            module._buffers.pop("total_params", None)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    summary = {
        "gmacs": operations / 1e9,
        "gflops": 2 * operations / 1e9,
        "profiled_million_parameters": profiled_parameters / 1e6,
        "million_parameters": total_parameters / 1e6,
    }
    logger.info(
        "Model profile | GFLOPs: %.4f | Params (THOP): %.4fM | Params: %.4fM",
        summary["gflops"],
        summary["profiled_million_parameters"],
        summary["million_parameters"],
    )
    print(
        "Model profile | "
        f"GFLOPs: {summary['gflops']:.4f} | "
        f"Params: {summary['million_parameters']:.4f}M"
    )
    return summary


# Backward-compatible names used by the original scripts.
clip_gradient = clip_gradients
AvgMeter = AverageMeter
cal_params_flops = profile_model
