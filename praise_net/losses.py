"""Training objectives for the manuscript PRAISE-Net model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    dims = tuple(range(1, probabilities.ndim))
    intersection = (probabilities * targets).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.45,
    beta: float = 0.55,
    eps: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits.float())
    targets = targets.float()
    dims = tuple(range(1, probabilities.ndim))
    true_positive = (probabilities * targets).sum(dim=dims)
    false_positive = (probabilities * (1.0 - targets)).sum(dim=dims)
    false_negative = ((1.0 - probabilities) * targets).sum(dim=dims)
    tversky = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    return 1.0 - tversky.mean()


def _soft_binary_erosion(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    padded = F.pad(mask, (padding, padding, padding, padding), value=0.0)
    return -F.max_pool2d(-padded, kernel_size, stride=1)


def _soft_binary_dilation(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    padded = F.pad(mask, (padding, padding, padding, padding), value=0.0)
    return F.max_pool2d(padded, kernel_size, stride=1)


@torch.no_grad()
def boundary_target(targets: torch.Tensor) -> torch.Tensor:
    """Two-sided one-pixel morphological boundary target."""
    targets = targets.float().clamp(0.0, 1.0)
    return (
        _soft_binary_dilation(targets) - _soft_binary_erosion(targets)
    ).clamp(0.0, 1.0)


@torch.no_grad()
def truncated_signed_distance_target(
    targets: torch.Tensor, maximum_distance: int = 16
) -> torch.Tensor:
    """Multi-radius signed-distance surrogate: negative inside, positive outside."""
    maximum_distance = int(maximum_distance)
    if maximum_distance < 1:
        raise ValueError("maximum_distance must be positive")

    targets = (targets.float() >= 0.5).float()
    foreground = targets
    background = 1.0 - foreground
    inside_distance = torch.ones_like(targets)
    outside_distance = torch.ones_like(targets)
    radii = tuple(
        radius for radius in (1, 2, 4, 8) if radius <= maximum_distance
    ) or (1,)

    for radius in radii:
        kernel_size = radius * 2 + 1
        normalized_distance = min(1.0, radius / float(maximum_distance))
        value = torch.full_like(targets, normalized_distance)

        eroded = _soft_binary_erosion(foreground, kernel_size)
        inside_shell = (foreground - eroded).clamp(0.0, 1.0)
        inside_distance = torch.where(
            (inside_shell > 0.5) & (value < inside_distance),
            value,
            inside_distance,
        )

        dilated = _soft_binary_dilation(foreground, kernel_size)
        outside_shell = (dilated - foreground).clamp(0.0, 1.0) * background
        outside_distance = torch.where(
            (outside_shell > 0.5) & (value < outside_distance),
            value,
            outside_distance,
        )

    signed_distance = outside_distance * background - inside_distance * foreground
    return signed_distance.clamp(-1.0, 1.0)


def balanced_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    maximum_weight: float = 20.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    dims = tuple(range(1, targets.ndim))
    positive_fraction = targets.mean(dim=dims, keepdim=True).detach()
    positive_weight = (0.5 / positive_fraction.clamp_min(eps)).clamp(
        max=maximum_weight
    )
    negative_weight = (0.5 / (1.0 - positive_fraction).clamp_min(eps)).clamp(
        max=maximum_weight
    )
    weights = targets * positive_weight + (1.0 - targets) * negative_weight
    loss_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss_map * weights).mean()


def weighted_region_mean(
    values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    weights = weights.float()
    return (values.float() * weights).sum() / weights.sum().clamp_min(eps)


class EvaluationLoss(nn.Module):
    """Validation loss used in the reported experiment; metrics are computed separately."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
        return 0.40 * bce + 0.60 * soft_dice_loss(logits, targets)


class PRAISELoss(nn.Module):
    """Joint CPG-MTRG-RAISE objective used for the paper's main model."""

    def __init__(
        self,
        proposal_weight: float = 0.35,
        boundary_weight: float = 0.20,
        distance_weight: float = 0.10,
        relative_region_weight: float = 0.15,
        distance_boundary_weight: float = 0.05,
        remote_background_weight: float = 3.0,
        deep_foreground_weight: float = 1.5,
        maximum_distance: int = 16,
    ) -> None:
        super().__init__()
        self.proposal_weight = float(proposal_weight)
        self.boundary_weight = float(boundary_weight)
        self.distance_weight = float(distance_weight)
        self.relative_region_weight = float(relative_region_weight)
        self.distance_boundary_weight = float(distance_boundary_weight)
        self.remote_background_weight = float(remote_background_weight)
        self.deep_foreground_weight = float(deep_foreground_weight)
        self.maximum_distance = int(maximum_distance)

    @staticmethod
    def segmentation_term(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
        dice = soft_dice_loss(logits, targets)
        tversky = soft_tversky_loss(
            logits, targets, alpha=0.45, beta=0.55
        )
        return 0.30 * bce + 0.45 * dice + 0.25 * tversky

    @staticmethod
    def boundary_term(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return 0.50 * balanced_soft_bce(
            logits, targets
        ) + 0.50 * soft_dice_loss(logits, targets)

    def forward(
        self, outputs: dict[str, torch.Tensor | list[torch.Tensor]], targets: torch.Tensor
    ) -> torch.Tensor:
        required = {
            "logits",
            "proposal_logits",
            "mtrg_reliability_logits",
            "raise_boundary_logits",
            "raise_signed_distance",
        }
        missing = sorted(required - outputs.keys())
        if missing:
            raise RuntimeError(f"Missing PRAISE-Net outputs: {missing}")

        targets = targets.float()
        final_logits = outputs["logits"].float()
        proposal_logits = outputs["proposal_logits"].float()
        reliability_logits = outputs["mtrg_reliability_logits"]
        if not isinstance(reliability_logits, list) or len(reliability_logits) != 4:
            raise RuntimeError("PRAISE-Net must return four MTRG reliability fields")

        target_boundary = boundary_target(targets)
        target_signed_distance = truncated_signed_distance_target(
            targets, self.maximum_distance
        )

        final_loss = self.segmentation_term(final_logits, targets)
        proposal_loss = self.segmentation_term(proposal_logits, targets)

        raise_boundary_loss = self.boundary_term(
            outputs["raise_boundary_logits"].float(), target_boundary
        )
        mtrg_boundary_loss = targets.new_zeros(())
        for field_logits in reliability_logits:
            resized_target = F.interpolate(
                target_boundary,
                size=field_logits.shape[-2:],
                mode="nearest",
            )
            mtrg_boundary_loss = mtrg_boundary_loss + self.boundary_term(
                field_logits.float(), resized_target
            )
        mtrg_boundary_loss = mtrg_boundary_loss / len(reliability_logits)
        combined_boundary_loss = (
            0.70 * raise_boundary_loss + 0.30 * mtrg_boundary_loss
        )

        predicted_signed_distance = outputs["raise_signed_distance"].float()
        distance_loss = F.smooth_l1_loss(
            predicted_signed_distance, target_signed_distance
        )
        distance_loss = distance_loss + self.distance_boundary_weight * weighted_region_mean(
            predicted_signed_distance.abs(), target_boundary
        )

        probability = torch.sigmoid(final_logits)
        outside_distance = target_signed_distance.clamp_min(0.0)
        inside_distance = (-target_signed_distance).clamp_min(0.0)
        background_weights = (1.0 - targets) * (
            1.0 + self.remote_background_weight * outside_distance
        )
        foreground_weights = targets * (
            1.0 + self.deep_foreground_weight * inside_distance
        )
        background_penalty = weighted_region_mean(
            probability, background_weights
        )
        foreground_penalty = weighted_region_mean(
            1.0 - probability, foreground_weights
        )
        relative_region_loss = (
            0.60 * background_penalty + 0.40 * foreground_penalty
        )

        return (
            final_loss
            + self.proposal_weight * proposal_loss
            + self.boundary_weight * combined_boundary_loss
            + self.distance_weight * distance_loss
            + self.relative_region_weight * relative_region_loss
        )
