"""Saliency metrics for RSOD/SOD evaluation.

The implementation is delegated to PySODMetrics. Predictions and masks are accumulated
over the whole dataset before max/mean F-measure and E-measure are reduced.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
except ImportError as exc:  # Keep the error actionable on a new training server.
    Emeasure = Fmeasure = MAE = Smeasure = WeightedFmeasure = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


METRIC_KEYS = (
    "MAE",
    "S_measure",
    "maxF",
    "meanF",
    "adpF",
    "maxE",
    "meanE",
    "adpE",
    "weighted_F",
)


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D saliency map, got shape {array.shape}")
    return array


def prepare_metric_data(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a probability map and mask to the uint8 format PySODMetrics expects.

    ``pred`` may be float in [0, 1] or uint8 in [0, 255].  ``gt`` may be a
    boolean/0-1 mask or uint8 mask.  Ground truth is binarized at 0.5 (or 128
    for uint8), which avoids the non-standard 0.2 threshold used previously.
    """
    pred = _as_2d(pred, "pred")
    gt = _as_2d(gt, "gt")
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}")

    if pred.dtype == np.uint8:
        pred_u8 = pred.copy()
    else:
        pred_float = np.nan_to_num(
            pred.astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0
        )
        pred_u8 = np.rint(np.clip(pred_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    if gt.dtype == bool:
        gt_bool = gt
    else:
        gt_float = np.nan_to_num(gt.astype(np.float64), nan=0.0)
        gt_max = float(gt_float.max()) if gt_float.size else 0.0
        # RSOD masks may be encoded as 0/1 or 0/255. Detect the encoding
        # before thresholding; otherwise uint8 0/1 masks become all background.
        if gt_max <= 1.0:
            gt_bool = gt_float >= 0.5
        else:
            gt_bool = gt_float > 128.0
    gt_u8 = gt_bool.astype(np.uint8) * 255

    return np.ascontiguousarray(pred_u8), np.ascontiguousarray(gt_u8)


class RSODMetricTracker:
    """Accumulate standard SOD metrics over one complete dataset.

    Set ``full=False`` during training validation to compute only MAE and
    S-measure.  This keeps validation fast while retaining the exact same
    definitions used by the final test script.
    """

    def __init__(self, full: bool = True):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "Standard RSOD metrics require PySODMetrics. "
                "Install it with: pip install pysodmetrics"
            ) from _IMPORT_ERROR

        self.full = full
        self.count = 0
        self._mae = MAE()
        self._sm = Smeasure()
        if full:
            self._fm = Fmeasure(beta=0.3)
            self._em = Emeasure()
            self._wfm = WeightedFmeasure(beta=1)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
        _, gt_u8 = prepare_metric_data(pred, gt)
        probability = _as_2d(pred, "pred").astype(np.float64)
        if pred.dtype == np.uint8:
            probability /= 255.0
        probability = np.clip(probability, 0, 1)
        target = gt_u8.astype(bool)
        self._mae.step(pred=probability, gt=target, normalize=False)
        self._sm.step(pred=probability, gt=target, normalize=False)
        if self.full:
            self._fm.step(pred=probability, gt=target, normalize=False)
            self._em.step(pred=probability, gt=target, normalize=False)
            self._wfm.step(pred=probability, gt=target, normalize=False)
        self.count += 1

        # PySODMetrics stores every sample result. Returning the newest one
        # lets evaluate_rsod.py create a detailed report without evaluating each
        # image twice. Final reported scores still come from averages().
        current = {
            "MAE": float(self._mae.maes[-1]),
            "S_measure": float(self._sm.sms[-1]),
        }
        if self.full:
            fm_curve = self._fm.changeable_fms[-1]
            em_curve = self._em.changeable_ems[-1]
            current.update(
                {
                    "maxF": float(np.max(fm_curve)),
                    "meanF": float(np.mean(fm_curve)),
                    "adpF": float(self._fm.adaptive_fms[-1]),
                    "maxE": float(np.max(em_curve)),
                    "meanE": float(np.mean(em_curve)),
                    "adpE": float(self._em.adaptive_ems[-1]),
                    "weighted_F": float(self._wfm.weighted_fms[-1]),
                }
            )
        return current

    def averages(self) -> Dict[str, float]:
        if self.count == 0:
            raise RuntimeError(
                "No prediction/ground-truth pairs were added to the metric tracker"
            )

        results = {
            "MAE": float(self._mae.get_results()["mae"]),
            "S_measure": float(self._sm.get_results()["sm"]),
        }
        if self.full:
            fm = self._fm.get_results()["fm"]
            em = self._em.get_results()["em"]
            results.update(
                {
                    # Correct dataset-level reduction: reduce the curve only
                    # after PySODMetrics has averaged it across all images.
                    "maxF": float(np.max(fm["curve"])),
                    "meanF": float(np.mean(fm["curve"])),
                    "adpF": float(fm["adp"]),
                    "maxE": float(np.max(em["curve"])),
                    "meanE": float(np.mean(em["curve"])),
                    "adpE": float(em["adp"]),
                    "weighted_F": float(self._wfm.get_results()["wfm"]),
                }
            )
        return results

    def format_string(self) -> str:
        metrics = self.averages()
        lines = [
            "=" * 50,
            "RSOD Evaluation Results (PySODMetrics)",
            "=" * 50,
            f"MAE:                 {metrics['MAE']:.4f}",
            f"S-measure:           {metrics['S_measure']:.4f}",
        ]
        if self.full:
            lines.extend(
                [
                    f"maxF-measure:        {metrics['maxF']:.4f}",
                    f"meanF-measure:       {metrics['meanF']:.4f}",
                    f"adaptive F-measure:  {metrics['adpF']:.4f}",
                    f"maxE-measure:        {metrics['maxE']:.4f}",
                    f"meanE-measure:       {metrics['meanE']:.4f}",
                    f"adaptive E-measure:  {metrics['adpE']:.4f}",
                    f"weighted F-measure:  {metrics['weighted_F']:.4f}",
                ]
            )
        lines.append("=" * 50)
        return "\n".join(lines)


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute standard metrics for one image (for detailed reports only)."""
    tracker = RSODMetricTracker(full=True)
    tracker.update(pred, gt)
    return tracker.averages()


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    tracker = RSODMetricTracker(full=False)
    tracker.update(pred, gt)
    return tracker.averages()["MAE"]


def s_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    tracker = RSODMetricTracker(full=False)
    tracker.update(pred, gt)
    return tracker.averages()["S_measure"]
