"""Train BRMSNet with quality-aware multi-scale supervision."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from evaluate_rsod import evaluate_dataset
from models import MODEL_NAME, QAMWS, BRMSNet
from utils.dataloader_rsod import create_dataloader, training_split
from utils.losses import segmentation_loss as segmentation_loss
from utils.training import AverageMeter, clip_gradients, profile_model


def validate(
    model: nn.Module,
    dataset_directory: Path,
    args: argparse.Namespace,
    device: torch.device,
    validation_samples=None,
) -> Dict[str, float]:
    tracker, _, binary_metrics = evaluate_dataset(
        model=model,
        dataset_directory=dataset_directory,
        split="val",
        image_size=args.image_size,
        batch_size=args.eval_batch_size,
        color_image=args.color_image,
        num_workers=args.num_workers,
        device=device,
        threshold=0.5,
        samples=validation_samples,
        full_metrics=False,
    )
    return {**tracker.averages(), **binary_metrics, "threshold": 0.5}


def _training_scales(multi_scale: bool) -> Sequence[float]:
    return (0.75, 1.0, 1.25) if multi_scale else (1.0,)


def train_one_epoch(
    train_loader,
    model: nn.Module,
    supervision: QAMWS,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor]:
    """Return elapsed seconds, average loss and image-averaged scale weights."""

    model.train()
    start_time = time.time()
    loss_meter = AverageMeter()
    scale_rates = _training_scales(args.multi_scale)
    weight_sum = torch.zeros(4, device=device)
    sample_count = 0
    consecutive_amp_overflows = 0

    for step, (images, masks, valid_regions) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.float().to(device, non_blocking=True)
        valid_regions = valid_regions.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        for scale_rate in scale_rates:
            if scale_rate == 1.0:
                scaled_images, scaled_masks = images, masks
                scaled_valid_regions = valid_regions
            else:
                scaled_size = int(round(args.image_size * scale_rate / 32) * 32)
                scaled_images = F.interpolate(
                    images,
                    size=(scaled_size, scaled_size),
                    mode="bilinear",
                    align_corners=True,
                )
                scaled_masks = F.interpolate(
                    masks,
                    size=(scaled_size, scaled_size),
                    mode="nearest",
                )
                scaled_valid_regions = F.interpolate(
                    valid_regions,
                    size=(scaled_size, scaled_size),
                    mode="nearest",
                )

            with autocast(enabled=args.use_amp):
                head_logits = model(scaled_images, return_all=True)
                result = supervision(
                    head_logits, scaled_masks, scaled_valid_regions, epoch
                )
                loss = result.loss
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Non-finite loss: epoch={epoch}, step={step}, scale={scale_rate}, "
                        f"weights={result.weights.cpu().tolist()}"
                    )
                if scale_rate == 1.0:
                    weight_sum += result.weights.sum(dim=0)
                    sample_count += images.shape[0]
                averaged_scale_loss = loss / len(scale_rates)

            scaler.scale(averaged_scale_loss).backward()
            if scale_rate == 1.0:
                loss_meter.update(loss, count=images.shape[0])

        scaler.unscale_(optimizer)
        gradients_finite, gradient_norm = clip_gradients(optimizer, args.gradient_clip)
        if not gradients_finite:
            if not args.use_amp:
                raise FloatingPointError(
                    "Non-finite gradients in FP32 training; "
                    f"epoch={epoch}, step={step}, norm={gradient_norm}"
                )
            previous_scale = scaler.get_scale()
            # unscale_ has already registered the overflow.  GradScaler will
            # skip this optimizer step and reduce its scale during update().
            scaler.step(optimizer)
            scaler.update()
            consecutive_amp_overflows += 1
            print(
                "AMP gradient overflow: optimizer step skipped | "
                f"epoch {epoch} | step {step} | norm {gradient_norm} | "
                f"scale {previous_scale:.0f} -> {scaler.get_scale():.0f}",
                flush=True,
            )
            if consecutive_amp_overflows >= args.max_consecutive_amp_overflows:
                raise FloatingPointError(
                    "Too many consecutive AMP gradient overflows; "
                    f"epoch={epoch}, step={step}, "
                    f"count={consecutive_amp_overflows}"
                )
            continue

        consecutive_amp_overflows = 0
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_every == 0 or step == len(train_loader):
            print(
                f"{datetime.now()} | epoch {epoch:03d}/{args.epochs:03d} | "
                f"step {step:04d}/{len(train_loader):04d} | "
                f"lr {optimizer.param_groups[1]['lr']:.6f} | "
                f"loss {loss_meter.moving_average().item():.4f}"
            )

    elapsed = time.time() - start_time
    if sample_count == 0:
        raise RuntimeError("Training loader is empty")
    return elapsed, loss_meter.average.item(), weight_sum / sample_count


def _create_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)
    return logger


def _write_weight_header(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(["epoch", "main", "eighth", "quarter", "half"])


def _append_scale_weights(
    csv_path: Path,
    epoch: int,
    weights: torch.Tensor,
) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([epoch, *weights.cpu().tolist()])


def load_warm_start(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    """Reuse compatible weights, including checkpoints predating BRMSNet."""

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as error:
        print(
            "Warning: safe weights-only loading could not parse this trusted "
            f"training checkpoint ({error}); retrying with weights_only=False."
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    if checkpoint and all(key.startswith("module.") for key in checkpoint):
        checkpoint = {key[len("module.") :]: value for key, value in checkpoint.items()}
    checkpoint = {
        key: value
        for key, value in checkpoint.items()
        if not key.endswith(("total_ops", "total_params"))
    }
    current = model.state_dict()
    resized_attention = {
        key
        for key, value in checkpoint.items()
        if key in current
        and value.shape != current[key].shape
        and key.startswith(("CA4.fc", "CA5.fc"))
    }
    checkpoint = {
        key: value for key, value in checkpoint.items() if key not in resized_attention
    }
    incompatible = model.load_state_dict(checkpoint, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("full_resolution_refinement.")
        and key not in resized_attention
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Warm-start checkpoint is incompatible; "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    print(
        f"Warm-started from {checkpoint_path} | "
        f"new parameters: {len(incompatible.missing_keys)}"
    )


def run_training(args: argparse.Namespace) -> None:
    """Run all requested independent training repetitions."""

    if args.image_size <= 0 or args.image_size % 32 != 0:
        raise ValueError("--image-size must be a positive multiple of 32")
    if not 0 < args.encoder_lr_multiplier <= 1:
        raise ValueError("--encoder-lr-multiplier must be in (0, 1]")
    if (
        min(
            args.multi_scale_loss_weight,
            args.mixture_loss_weight,
            args.boundary_loss_weight,
            args.tversky_loss_weight,
        )
        < 0
    ):
        raise ValueError("loss weights cannot be negative")
    if args.amp_initial_scale <= 0 or args.amp_growth_interval < 1:
        raise ValueError("AMP scale must be positive and growth interval at least 1")
    if args.max_consecutive_amp_overflows < 1:
        raise ValueError("--max-consecutive-amp-overflows must be at least 1")

    if args.runs < 1 or args.epochs < 1:
        raise ValueError("--runs and --epochs must be positive")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_directory = Path(args.data_root) / args.dataset
    train_directory = dataset_directory / "train"
    training_samples, validation_samples = training_split(
        dataset_directory, args.split_seed
    )
    output_root = Path(args.output_root)
    log_root = Path(args.log_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    for run_number in range(1, args.runs + 1):
        seed = args.seed + run_number - 1
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        timestamp = time.strftime("%H%M%S")
        run_id = (
            f"{args.dataset}_{MODEL_NAME}_bs{args.batch_size}_lr{args.learning_rate}_"
            f"e{args.epochs}_aug{args.augmentation}_seed{seed}_t{timestamp}"
        )
        run_directory = output_root / run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        logger = _create_logger(log_root / f"train_{run_id}.log")

        model = BRMSNet(pretrained=args.pretrained and args.warm_start is None).to(
            device
        )
        if args.warm_start is not None:
            load_warm_start(model, Path(args.warm_start), device)
        supervision = QAMWS(
            temperature=args.quality_temperature,
            weight_floor=args.quality_weight_floor,
            multi_scale_weight=args.multi_scale_loss_weight,
            mixture_weight=args.mixture_loss_weight,
            warmup_epochs=args.warmup_epochs,
            boundary_weight=args.boundary_loss_weight,
            tversky_weight=args.tversky_loss_weight,
        ).to(device)
        split_manifest = {
            "split_seed": args.split_seed,
            "train": [image.stem for image, _ in training_samples],
            "val": [image.stem for image, _ in validation_samples],
        }
        (run_directory / "split.json").write_text(
            json.dumps(split_manifest, indent=2), encoding="utf-8"
        )
        print(f"Run: {run_id}")
        print(
            f"Model: {MODEL_NAME} | Backbone: pvt_v2_b1 | Supervision: QAMWS | Seed: {seed}"
        )
        print(f"Device: {device} | AMP: {args.use_amp}")
        print(f"Samples: train {len(training_samples)} | val {len(validation_samples)}")
        if args.profile:
            profile_model(model, args.image_size, logger)

        encoder_parameters = list(model.encoder.parameters())
        encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
        decoder_parameters = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in encoder_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_parameters,
                    "lr": args.learning_rate * args.encoder_lr_multiplier,
                },
                {"params": decoder_parameters, "lr": args.learning_rate},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.minimum_learning_rate,
        )
        scaler = GradScaler(
            enabled=args.use_amp,
            init_scale=args.amp_initial_scale,
            growth_interval=args.amp_growth_interval,
        )
        train_loader = create_dataloader(
            image_root=train_directory / "images",
            mask_root=train_directory / "masks",
            batch_size=args.batch_size,
            image_size=args.image_size,
            shuffle=True,
            num_workers=args.num_workers,
            augmentation=args.augmentation,
            split="train",
            color_image=args.color_image,
            pin_memory=device.type == "cuda",
            samples=training_samples,
            seed=seed,
        )

        weights_csv = run_directory / "qamws_weights.csv"
        _write_weight_header(weights_csv)

        best_selection_score = float("-inf")
        best_validation: Dict[str, float] = {}
        total_training_seconds = 0.0
        for epoch in range(1, args.epochs + 1):
            epoch_seconds, epoch_loss, mixing_weights = train_one_epoch(
                train_loader,
                model,
                supervision,
                optimizer,
                scaler,
                epoch,
                args,
                device,
            )
            total_training_seconds += epoch_seconds
            scheduler.step()
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "head_order": ["main", "eighth", "quarter", "half"],
                    "args": vars(args),
                },
                run_directory / f"{run_id}-last.pth",
            )
            _append_scale_weights(weights_csv, epoch, mixing_weights)

            message = (
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"seconds {epoch_seconds:.1f} | loss {epoch_loss:.4f} | "
                f"lr {optimizer.param_groups[1]['lr']:.6f}"
            )
            print(message)
            logger.info(message)

            validation = validate(
                model, dataset_directory, args, device, validation_samples
            )
            validation_message = (
                f"Validation | epoch {epoch} | S-measure {validation['S_measure']:.4f} | "
                f"MAE {validation['MAE']:.4f} | Dice {validation['Dice']:.4f} | "
                f"IoU {validation['IoU']:.4f} | Boundary-F1 "
                f"{validation['Boundary_F1']:.4f} | threshold {validation['threshold']:.2f}"
            )
            print(validation_message)
            logger.info(validation_message)
            selection_score = (
                0.4 * validation["S_measure"]
                + 0.3 * validation["Dice"]
                + 0.3 * validation["Boundary_F1"]
            )
            if selection_score > best_selection_score:
                previous_best = best_selection_score
                best_selection_score = selection_score
                best_validation = dict(validation)

                best_checkpoint_path = run_directory / f"{run_id}-best_score.pth"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "threshold": validation["threshold"],
                        "selection_score": selection_score,
                        "validation": validation,
                        "seed": seed,
                        "head_order": ["main", "eighth", "quarter", "half"],
                        "args": vars(args),
                    },
                    best_checkpoint_path,
                )

                print(
                    f"Best model saved | epoch {epoch} | "
                    f"score {previous_best:.4f} -> {best_selection_score:.4f} | "
                    f"{best_checkpoint_path}",
                    flush=True,
                )
                logger.info(
                    "Best model saved | epoch %03d/%03d | "
                    "selection score %.4f -> %.4f | %s",
                    epoch,
                    args.epochs,
                    previous_best,
                    best_selection_score,
                    best_checkpoint_path.resolve(),
                )

        summary = (
            f"Finished {run_id} | best selection score {best_selection_score:.4f} | "
            f"S-measure {best_validation.get('S_measure', float('nan')):.4f} | "
            f"Dice {best_validation.get('Dice', float('nan')):.4f} | "
            f"Boundary-F1 {best_validation.get('Boundary_F1', float('nan')):.4f} | "
            f"training time {total_training_seconds:.2f}s"
        )
        print(summary)
        logger.info(summary)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="RSISOD")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-multiplier", type=float, default=0.1)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--amp-initial-scale", type=float, default=4096.0)
    parser.add_argument("--amp-growth-interval", type=int, default=2000)
    parser.add_argument("--max-consecutive-amp-overflows", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--quality-temperature", type=float, default=0.2)
    parser.add_argument("--quality-weight-floor", type=float, default=0.2)
    parser.add_argument("--multi-scale-loss-weight", type=float, default=0.6)
    parser.add_argument("--mixture-loss-weight", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.5)
    parser.add_argument("--tversky-loss-weight", type=float, default=0.3)
    parser.add_argument("--data-root", default="./data/rsod")
    parser.add_argument("--output-root", default="./model_pth")
    parser.add_argument("--log-root", default="./logs")
    parser.add_argument(
        "--warm-start",
        default=None,
        help="Initialize compatible layers from a dataset-matched checkpoint",
    )
    parser.add_argument("--device", default=None, help="Examples: cuda, cuda:1, cpu")

    parser.set_defaults(
        amp=True,
        augmentation=True,
        color_image=True,
        multi_scale=False,
        pretrained=True,
        profile=False,
    )
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.add_argument("--color-image", dest="color_image", action="store_true")
    parser.add_argument("--grayscale", dest="color_image", action="store_false")
    parser.add_argument("--multi-scale", dest="multi_scale", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--profile", dest="profile", action="store_true")
    parser.add_argument("--no-profile", dest="profile", action="store_false")
    return parser


def main() -> None:
    run_training(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
