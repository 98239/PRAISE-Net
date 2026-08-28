"""Patient-level segmentation metrics used in the PRAISE-Net experiment."""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 1.0 if denominator == 0 else float(numerator / denominator)


def _dice(tp: float, fp: float, fn: float) -> float:
    return _safe_ratio(2.0 * tp, 2.0 * tp + fp + fn)


def _iou(tp: float, fp: float, fn: float) -> float:
    return _safe_ratio(tp, tp + fp + fn)


def _recall(tp: float, _fp: float, fn: float) -> float:
    return _safe_ratio(tp, tp + fn)


def _precision(tp: float, fp: float, fn: float) -> float:
    return 0.0 if tp + fp == 0 and fn > 0 else _safe_ratio(tp, tp + fp)


def _slice_index(name: str) -> int:
    numbers = re.findall(r"\d+", name)
    return int(numbers[-1]) if numbers else 0


def _hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float(np.linalg.norm(np.asarray(prediction.shape, dtype=np.float64)))
    prediction_surface = np.logical_xor(
        prediction, binary_erosion(prediction)
    )
    target_surface = np.logical_xor(target, binary_erosion(target))
    distances = np.concatenate(
        [
            distance_transform_edt(~target_surface)[prediction_surface],
            distance_transform_edt(~prediction_surface)[target_surface],
        ]
    )
    return float(np.percentile(distances, 95))


class PatientMetricAccumulator:
    """Aggregate slices into patient volumes before macro-averaging patients."""

    def __init__(self, threshold: float = 0.5, store_volumes: bool = False) -> None:
        self.threshold = float(threshold)
        self.store_volumes = bool(store_volumes)
        self.counts = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        self.slices = defaultdict(list)

    @torch.no_grad()
    def update(
        self,
        probabilities: torch.Tensor,
        targets: torch.Tensor,
        patient_ids: list[str] | tuple[str, ...],
        names: list[str] | tuple[str, ...],
    ) -> None:
        predictions = probabilities.detach().cpu().numpy() >= self.threshold
        targets_array = targets.detach().cpu().numpy() >= 0.5
        for prediction, target, patient_id, name in zip(
            predictions, targets_array, patient_ids, names
        ):
            prediction = np.squeeze(prediction)
            target = np.squeeze(target)
            true_positive = np.logical_and(prediction, target).sum()
            false_positive = np.logical_and(prediction, ~target).sum()
            false_negative = np.logical_and(~prediction, target).sum()
            true_negative = np.logical_and(~prediction, ~target).sum()
            self.counts[str(patient_id)] += np.asarray(
                [true_positive, false_positive, false_negative, true_negative],
                dtype=np.float64,
            )
            if self.store_volumes:
                self.slices[str(patient_id)].append(
                    (_slice_index(str(name)), prediction.copy(), target.copy())
                )

    def compute(self, include_hd95: bool = False) -> dict[str, float | int | str]:
        if not self.counts:
            raise RuntimeError("No samples were accumulated")

        patient_rows: list[dict[str, float]] = []
        for patient_id in sorted(self.counts):
            true_positive, false_positive, false_negative, true_negative = (
                self.counts[patient_id]
            )
            row = {
                "dice": _dice(true_positive, false_positive, false_negative),
                "iou": _iou(true_positive, false_positive, false_negative),
                "precision": _precision(
                    true_positive, false_positive, false_negative
                ),
                "recall": _recall(
                    true_positive, false_positive, false_negative
                ),
                "accuracy": _safe_ratio(
                    true_positive + true_negative,
                    true_positive
                    + false_positive
                    + false_negative
                    + true_negative,
                ),
            }
            if include_hd95:
                ordered = sorted(
                    self.slices[patient_id], key=lambda item: item[0]
                )
                row["hd95"] = _hd95(
                    np.stack([item[1] for item in ordered]),
                    np.stack([item[2] for item in ordered]),
                )
            patient_rows.append(row)

        values: dict[str, float | int | str] = {
            "patients": len(patient_rows),
            "dice": float(np.mean([row["dice"] for row in patient_rows])),
            "iou": float(np.mean([row["iou"] for row in patient_rows])),
            "precision": float(
                np.mean([row["precision"] for row in patient_rows])
            ),
            "recall": float(np.mean([row["recall"] for row in patient_rows])),
            "accuracy": float(
                np.mean([row["accuracy"] for row in patient_rows])
            ),
        }
        total = np.sum(list(self.counts.values()), axis=0)
        true_positive, false_positive, false_negative, _true_negative = total
        values.update(
            global_dice=_dice(
                true_positive, false_positive, false_negative
            ),
            global_iou=_iou(
                true_positive, false_positive, false_negative
            ),
            global_precision=_precision(
                true_positive, false_positive, false_negative
            ),
            global_recall=_recall(
                true_positive, false_positive, false_negative
            ),
        )
        if include_hd95:
            values["hd95"] = float(
                np.mean([row["hd95"] for row in patient_rows])
            )
            values["hd95_unit"] = "voxel_on_resampled_grid"
        return values
