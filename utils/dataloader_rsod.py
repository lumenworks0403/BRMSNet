"""Dataset and DataLoader helpers for remote-sensing salient-object detection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

PathLike = Union[str, Path]
Sample = Tuple[Path, Path]
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _index_files_by_stem(directory: Path) -> Dict[str, Path]:
    """Index supported files by stem and reject ambiguous duplicates."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    index: Dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        if path.stem in index:
            raise ValueError(
                f"Duplicate sample stem {path.stem!r} in {directory}: "
                f"{index[path.stem].name} and {path.name}"
            )
        index[path.stem] = path
    return index


def pair_image_mask_paths(
    image_root: PathLike,
    mask_root: PathLike,
) -> List[Tuple[Path, Path]]:
    """Pair images and masks by filename stem, not by list position."""

    image_index = _index_files_by_stem(Path(image_root))
    mask_index = _index_files_by_stem(Path(mask_root))

    missing_masks = sorted(set(image_index) - set(mask_index))
    missing_images = sorted(set(mask_index) - set(image_index))
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing masks for: {missing_masks[:5]}")
        if missing_images:
            details.append(f"missing images for: {missing_images[:5]}")
        raise ValueError("Image/mask pairing failed; " + "; ".join(details))
    if not image_index:
        raise ValueError(f"No supported images found in {image_root}")

    return [(image_index[stem], mask_index[stem]) for stem in sorted(image_index)]


def training_split(
    dataset_directory: Path, seed: int = 42
) -> Tuple[List[Sample], List[Sample]]:
    """Use an existing validation split, or hold out 10% of official training data."""
    train_directory = dataset_directory / "train"
    training = pair_image_mask_paths(
        train_directory / "images", train_directory / "masks"
    )
    validation_directory = dataset_directory / "val"
    if validation_directory.exists():
        validation = pair_image_mask_paths(
            validation_directory / "images", validation_directory / "masks"
        )
        if {image.stem for image, _ in training} & {
            image.stem for image, _ in validation
        }:
            raise ValueError("Training and validation contain overlapping sample names")
        return training, validation
    validation_size = max(1, int(len(training) * 0.1))
    if len(training) < 2:
        raise ValueError(
            "At least two training samples are needed for a validation holdout"
        )
    indices = list(range(len(training)))
    random.Random(seed).shuffle(indices)
    held_out = set(indices[:validation_size])
    return (
        [sample for index, sample in enumerate(training) if index not in held_out],
        [sample for index, sample in enumerate(training) if index in held_out],
    )


def seed_worker(worker_id: int) -> None:
    """Seed transforms in each DataLoader worker from its PyTorch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    dataset = torch.utils.data.get_worker_info().dataset
    transform = dataset.augmentation_transform
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(worker_seed)


class RSODDataset(Dataset):
    """Binary salient-object dataset with synchronized image/mask transforms."""

    def __init__(
        self,
        image_root: PathLike,
        mask_root: PathLike,
        image_size: int,
        augmentation: bool = False,
        split: str = "train",
        color_image: bool = True,
        samples: Optional[Sequence[Sample]] = None,
        seed: int = 42,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        self.image_size = image_size
        self.color_image = color_image
        self.augmentation = augmentation
        self.split = split
        self.samples = (
            list(samples)
            if samples is not None
            else pair_image_mask_paths(image_root, mask_root)
        )

        self.mask_paths = [str(mask_path) for _, mask_path in self.samples]
        self.size = len(self.samples)
        self.augmentation_transform = self._build_augmentation_transform()
        if hasattr(self.augmentation_transform, "set_random_seed"):
            self.augmentation_transform.set_random_seed(seed)
        self.tensor_transform = self._build_tensor_transform()

    def _build_augmentation_transform(self) -> A.Compose:
        transforms = []
        if self.split == "train" and self.augmentation:
            transforms.extend(
                [
                    A.Rotate(limit=90, p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5),
                ]
            )
        return A.Compose(transforms)

    def _build_tensor_transform(self) -> A.Compose:
        mean: Sequence[float] = (0.485, 0.456, 0.406) if self.color_image else (0.5,)
        std: Sequence[float] = (0.229, 0.224, 0.225) if self.color_image else (0.229,)
        return A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])

    def _letterbox(self, image, mask):
        """Resize without distortion and pad image/mask to a fixed square."""

        height, width = mask.shape[:2]
        scale = self.image_size / max(height, width)
        resized_height = max(1, min(self.image_size, int(round(height * scale))))
        resized_width = max(1, min(self.image_size, int(round(width * scale))))
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )

        pad_height = self.image_size - resized_height
        pad_width = self.image_size - resized_width
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        image_pad_value = (124, 116, 104) if self.color_image else 128
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=image_pad_value,
        )
        mask = cv2.copyMakeBorder(
            mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        content_box = torch.tensor(
            [top, left, top + resized_height, left + resized_width],
            dtype=torch.long,
        )
        valid_region = torch.zeros(
            (1, self.image_size, self.image_size), dtype=torch.float32
        )
        valid_region[:, top : top + resized_height, left : left + resized_width] = 1.0
        return image, mask, content_box, valid_region

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        conversion = cv2.COLOR_BGR2RGB if self.color_image else cv2.COLOR_BGR2GRAY
        image = cv2.cvtColor(image, conversion)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        original_size = (mask.shape[1], mask.shape[0])  # PIL-compatible (width, height)

        augmented = self.augmentation_transform(image=image, mask=mask)
        image, mask, content_box, valid_region = self._letterbox(
            augmented["image"], augmented["mask"]
        )
        transformed = self.tensor_transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]

        # Support both 0/1 masks and the usual 0/255 encoding.  For 0/255
        # masks, values up to 20 are treated as compression/annotation noise.
        foreground_threshold = 20 if int(mask_tensor.max().item()) > 127 else 0
        mask_tensor = (mask_tensor > foreground_threshold).long()
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        if self.split == "train":
            return image_tensor, mask_tensor, valid_region
        prediction_name = f"{image_path.stem}.png"
        original_size_tensor = torch.tensor(
            [original_size[1], original_size[0]], dtype=torch.long
        )
        return (
            image_tensor,
            mask_tensor,
            original_size_tensor,
            content_box,
            prediction_name,
        )

    def __len__(self) -> int:
        return self.size


def create_dataloader(
    image_root: PathLike,
    mask_root: PathLike,
    batch_size: int,
    image_size: int,
    *,
    shuffle: bool = False,
    num_workers: int = 8,
    pin_memory: bool = True,
    augmentation: bool = False,
    split: str = "train",
    color_image: bool = True,
    samples: Optional[Sequence[Sample]] = None,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for one dataset split."""

    dataset = RSODDataset(
        image_root=image_root,
        mask_root=mask_root,
        image_size=image_size,
        augmentation=augmentation,
        split=split,
        color_image=color_image,
        samples=samples,
        seed=seed,
    )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
