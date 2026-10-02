"""Evaluate an RSOD checkpoint and write predictions plus metric reports."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from models import MODEL_NAME, BRMSNet
from utils.dataloader_rsod import (
    create_dataloader,
    pair_image_mask_paths,
    training_split,
)
from utils.rsod_metrics import RSODMetricTracker
from utils.saliency import (
    boundary_f1_score,
    dice_score,
    intersection_over_union,
    primary_logits,
    resized_probability_map,
)


def _load_checkpoint_file(checkpoint_path: Path, map_location):
    """Load trusted local checkpoints across PyTorch weights-only variants."""

    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        # PyTorch releases predating the weights_only argument.
        return torch.load(checkpoint_path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError) as error:
        print(
            "Warning: safe weights-only loading could not parse this trusted "
            f"training checkpoint ({error}); retrying with weights_only=False."
        )
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )


def find_checkpoint(run_directory: Path, run_id: str) -> Path:
    """Resolve the best available checkpoint using a documented priority."""

    candidates = (
        run_directory / f"{run_id}-best_score.pth",
        run_directory / f"{run_id}-best_Smeasure.pth",
        run_directory / f"{run_id}-best.pth",  # legacy name
        run_directory / f"{run_id}-last.pth",
    )
    for checkpoint_path in candidates:
        if checkpoint_path.is_file():
            return checkpoint_path
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No checkpoint found; attempted: {attempted}")


def load_state_dict(
    checkpoint_path: Path, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Load a weights-only checkpoint across supported PyTorch versions."""

    checkpoint = _load_checkpoint_file(checkpoint_path, device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")

    # Support checkpoints produced by nn.DataParallel.
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module.") :]: value for key, value in checkpoint.items()}

    # Older checkpoints were saved after THOP profiling.  THOP registered
    # non-model analysis buffers on many modules; discard only those known keys.
    return {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith(("total_ops", "total_params"))
    }


def load_checkpoint_threshold(checkpoint_path: Path) -> Optional[float]:
    """Read the validation-calibrated threshold from a new-format checkpoint."""

    checkpoint = _load_checkpoint_file(checkpoint_path, "cpu")
    if not isinstance(checkpoint, dict) or "threshold" not in checkpoint:
        return None
    return float(checkpoint["threshold"])


def evaluate_dataset(
    model: nn.Module,
    dataset_directory: Path,
    split: str,
    image_size: int,
    batch_size: int,
    color_image: bool,
    num_workers: int,
    device: torch.device,
    prediction_directory: Optional[Path] = None,
    threshold: float = 0.5,
    samples=None,
    full_metrics: bool = True,
) -> Tuple[RSODMetricTracker, List[Dict[str, object]], Dict[str, float]]:
    """Measure on the unpadded input grid; save maps at the original size."""

    split_directory = dataset_directory / split
    if split == "val" and samples is None and not split_directory.exists():
        _, samples = training_split(dataset_directory)
    data_loader = create_dataloader(
        image_root=split_directory / "images",
        mask_root=split_directory / "masks",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        num_workers=num_workers,
        split=split,
        color_image=color_image,
        pin_memory=device.type == "cuda",
        samples=samples,
    )
    if prediction_directory is not None:
        saliency_directory = prediction_directory / "saliency"
        binary_directory = prediction_directory / "binary"
        saliency_directory.mkdir(parents=True, exist_ok=True)
        binary_directory.mkdir(parents=True, exist_ok=True)

    tracker = RSODMetricTracker(full=full_metrics)
    binary_totals = {"Dice": 0.0, "IoU": 0.0, "Boundary_F1": 0.0}
    per_image_results: List[Dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for images, masks, original_sizes, content_boxes, names in tqdm(
            data_loader, desc=f"Inference: {split}"
        ):
            images = images.to(device, non_blocking=True)
            batch_logits = primary_logits(model(images))

            for sample_index, name in enumerate(names):
                output_size = tuple(
                    int(value) for value in original_sizes[sample_index]
                )
                content_box = tuple(int(value) for value in content_boxes[sample_index])
                top, left, bottom, right = content_box
                probability_array = (
                    batch_logits[sample_index, 0, top:bottom, left:right]
                    .float()
                    .sigmoid()
                    .cpu()
                    .numpy()
                )
                ground_truth = (
                    masks[sample_index, 0, top:bottom, left:right]
                    .numpy()
                    .astype(np.uint8)
                )
                binary_prediction = (probability_array >= threshold).astype(np.uint8)
                prediction_tensor = torch.from_numpy(binary_prediction)
                target_tensor = torch.from_numpy(ground_truth)
                binary_metrics = {
                    "Dice": dice_score(prediction_tensor, target_tensor).item(),
                    "IoU": intersection_over_union(
                        prediction_tensor, target_tensor
                    ).item(),
                    "Boundary_F1": boundary_f1_score(binary_prediction, ground_truth),
                }
                for metric, value in binary_metrics.items():
                    binary_totals[metric] += value
                sample_metrics = tracker.update(probability_array, ground_truth)
                per_image_results.append(
                    {
                        "Name": name,
                        **{
                            metric: round(value, 4)
                            for metric, value in sample_metrics.items()
                        },
                        **{
                            metric: round(value, 4)
                            for metric, value in binary_metrics.items()
                        },
                    }
                )

                if prediction_directory is not None:
                    probability = (
                        resized_probability_map(
                            batch_logits[sample_index], output_size, content_box
                        )
                        .cpu()
                        .numpy()
                    )
                    output = np.rint(probability * 255).astype(np.uint8)
                    output_path = saliency_directory / name
                    if not cv2.imwrite(str(output_path), output):
                        raise RuntimeError(f"Failed to save prediction: {output_path}")
                    binary_output = (probability >= threshold).astype(np.uint8) * 255
                    binary_path = binary_directory / name
                    if not cv2.imwrite(str(binary_path), binary_output):
                        raise RuntimeError(f"Failed to save prediction: {binary_path}")

    if tracker.count == 0:
        raise RuntimeError(f"Evaluation split is empty: {split_directory}")
    binary_averages = {
        metric: total / tracker.count for metric, total in binary_totals.items()
    }
    return tracker, per_image_results, binary_averages


def write_detailed_report(
    report_path: Path,
    per_image_results: List[Dict[str, object]],
    averages: Dict[str, float],
) -> None:
    """Write per-image scores followed by a dataset-average row."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(per_image_results)
    average_row: Dict[str, object] = {
        "Name": "AVERAGE",
        **{metric: round(value, 4) for metric, value in averages.items()},
    }
    result_frame = pd.concat(
        [result_frame, pd.DataFrame([average_row])],
        ignore_index=True,
    )
    result_frame.to_excel(report_path, index=False)


def append_summary(
    summary_path: Path,
    run_metadata: Dict[str, object],
    averages: Dict[str, float],
) -> None:
    """Append one dataset-level result row to the cross-run summary."""

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{**run_metadata, **averages}])
    if summary_path.is_file():
        existing = pd.read_excel(summary_path)
        new_row = pd.concat([existing, new_row], ignore_index=True)
    new_row.to_excel(summary_path, index=False)


def run_evaluation(args: argparse.Namespace) -> None:
    if args.image_size <= 0 or args.image_size % 32 != 0:
        raise ValueError("--image-size must be a positive multiple of 32")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else find_checkpoint(Path(args.model_root) / args.run_id, args.run_id)
    )
    model = BRMSNet(pretrained=False).to(device)
    model.load_state_dict(load_state_dict(checkpoint_path, device), strict=True)
    threshold = args.threshold
    validation_samples = None
    if args.split == "val":
        manifest_path = (
            Path(args.split_file)
            if args.split_file
            else checkpoint_path.parent / "split.json"
        )
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_directory = Path(args.data_root) / args.dataset
            source = dataset_directory / (
                "val" if (dataset_directory / "val").exists() else "train"
            )
            pairs = pair_image_mask_paths(source / "images", source / "masks")
            index = {image.stem: (image, mask) for image, mask in pairs}
            validation_samples = [index[name] for name in manifest["val"]]
        elif args.split_file:
            raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")

    prediction_directory = (
        Path(args.prediction_root) / args.run_id / args.dataset / args.split
    )
    tracker, per_image_results, binary_averages = evaluate_dataset(
        model=model,
        dataset_directory=Path(args.data_root) / args.dataset,
        split=args.split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        color_image=args.color_image,
        num_workers=args.num_workers,
        device=device,
        prediction_directory=prediction_directory,
        threshold=threshold,
        samples=validation_samples,
    )
    averages = {
        **tracker.averages(),
        **binary_averages,
        "threshold": threshold,
    }

    report_path = (
        Path(args.result_root)
        / f"Results_{args.run_id}_{args.dataset}_{args.split}.xlsx"
    )
    write_detailed_report(report_path, per_image_results, averages)
    append_summary(
        Path(args.summary_file),
        {
            "run_id": args.run_id,
            "model": MODEL_NAME,
            "dataset": args.dataset,
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "image_size": args.image_size,
            "threshold": threshold,
            "boundary_tolerance": 2,
        },
        averages,
    )

    print(tracker.format_string())
    print(
        f"Binary | Dice: {binary_averages['Dice']:.4f} | "
        f"IoU: {binary_averages['IoU']:.4f} | "
        f"Boundary-F1: {binary_averages['Boundary_F1']:.4f}"
    )
    print(f"Predictions: {prediction_directory}")
    print(f"Binary threshold: {threshold:.2f}")
    print(f"Evaluation grid: {args.image_size} x {args.image_size}, excluding padding")
    print(f"Detailed report: {report_path}")
    print(f"Summary: {args.summary_file}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", default="ORSSD")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--data-root", default="./data/rsod")
    parser.add_argument("--model-root", default="./model_pth")
    parser.add_argument("--prediction-root", default="./predictions_rsod")
    parser.add_argument("--result-root", default="./results_rsod")
    parser.add_argument("--summary-file", default="./All_RSOD_Runs_Summary.xlsx")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--split-file",
        default=None,
        help="Validation split manifest; defaults to checkpoint directory/split.json",
    )
    parser.add_argument("--device", default=None, help="Examples: cuda, cuda:1, cpu")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary threshold (default: 0.5)",
    )
    parser.set_defaults(color_image=True)
    parser.add_argument("--color-image", dest="color_image", action="store_true")
    parser.add_argument("--grayscale", dest="color_image", action="store_false")
    return parser


def main() -> None:
    run_evaluation(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
