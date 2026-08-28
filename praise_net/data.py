"""Dataset loading and augmentation for patient-level five-fold training."""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def patient_id_from_name(name: str) -> str:
    """Extract the ECPC-IDS patient identifier from a slice filename."""
    stem = Path(name).stem
    if len(stem) >= 6 and stem[:6].isdigit():
        return stem[:3]
    match = re.match(r"(.+?)[_-]?\d+$", stem)
    return match.group(1) if match else stem


class CTSegmentationDataset(Dataset):
    """Paired single-channel CT images and binary lesion masks."""

    def __init__(
        self,
        ct_dir: str | Path,
        label_dir: str | Path,
        image_size: int = 256,
        training: bool = False,
    ) -> None:
        self.ct_dir = Path(ct_dir)
        self.label_dir = Path(label_dir)
        self.image_size = int(image_size)
        self.training = bool(training)

        if not self.ct_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing CT or label directory: {self.ct_dir}, {self.label_dir}"
            )

        labels = {
            path.stem: path
            for path in self.label_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        }
        self.pairs: list[tuple[Path, Path]] = []
        for ct_path in sorted(self.ct_dir.iterdir()):
            if ct_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = labels.get(ct_path.stem)
            if label_path is None:
                raise FileNotFoundError(
                    f"No label with the same stem as CT image: {ct_path.name}"
                )
            self.pairs.append((ct_path, label_path))

        if not self.pairs:
            raise RuntimeError(f"No CT/label pairs found under {self.ct_dir.parent}")
        if len(self.pairs) != len(labels):
            raise RuntimeError(
                f"CT/label count mismatch: {len(self.pairs)} CT files and "
                f"{len(labels)} label files"
            )

        self.names = [ct_path.name for ct_path, _ in self.pairs]
        self.patient_ids = [patient_id_from_name(name) for name in self.names]
        slice_counts = Counter(self.patient_ids)
        self.patient_balanced_weights = [
            1.0 / slice_counts[patient_id] for patient_id in self.patient_ids
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _load_grayscale(path: Path) -> torch.Tensor:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32).copy()
        return torch.from_numpy(array).unsqueeze(0).div_(255.0)

    def _resize(self, image: torch.Tensor, mode: str) -> torch.Tensor:
        kwargs: dict[str, object] = {
            "size": (self.image_size, self.image_size),
            "mode": mode,
        }
        if mode != "nearest":
            kwargs["align_corners"] = False
        return F.interpolate(image.unsqueeze(0), **kwargs).squeeze(0)

    @staticmethod
    def _augment(
        image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        if random.random() < 0.5:
            image = (
                image * random.uniform(0.9, 1.1)
                + random.uniform(-0.03, 0.03)
            ).clamp(0, 1)
        if random.random() < 0.25:
            noise_scale = random.uniform(0.003, 0.015)
            image = (image + torch.randn_like(image) * noise_scale).clamp(0, 1)
        return image, mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        ct_path, label_path = self.pairs[index]
        image = self._resize(self._load_grayscale(ct_path), "bilinear")
        mask = self._resize(self._load_grayscale(label_path), "nearest")
        mask = (mask >= 0.5).float()
        if self.training:
            image, mask = self._augment(image, mask)
        return {
            "image": image,
            "mask": mask,
            "name": self.names[index],
            "patient_id": self.patient_ids[index],
        }


def fold_directories(data_root: str | Path, fold: int) -> dict[str, Path]:
    root = Path(data_root) / f"fold_{fold}"
    return {
        "train_ct": root / "train" / "ct",
        "train_label": root / "train" / "label",
        "val_ct": root / "val" / "ct",
        "val_label": root / "val" / "label",
    }
