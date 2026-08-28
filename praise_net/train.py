"""Train the manuscript PRAISE-Net model with patient-level five-fold validation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import CTSegmentationDataset, fold_directories
from .losses import EvaluationLoss, PRAISELoss
from .metrics import PatientMetricAccumulator
from .model import PRAISENet


SUMMARY_FIELDS = [
    "model",
    "fold",
    "best_epoch",
    "best_val_dice",
    "train_samples",
    "val_samples",
    "patients",
    "loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "accuracy",
    "hd95",
    "hd95_unit",
    "global_dice",
    "global_iou",
    "global_precision",
    "global_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level five-fold training for PRAISE-Net"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/five_fold")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute intermediate activations to reduce GPU memory use.",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable the patient-balanced training sampler.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip a fold when its metrics.json already exists.",
    )
    parser.add_argument(
        "--amp", dest="amp", action="store_true", help="Enable CUDA AMP."
    )
    parser.add_argument(
        "--no-amp", dest="amp", action="store_false", help="Disable CUDA AMP."
    )
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Enable deterministic PyTorch/CuDNN behavior.",
    )
    parser.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow non-deterministic kernels.",
    )
    parser.set_defaults(amp=True, deterministic=True)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(False)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    data_root: Path,
    fold: int,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    balanced_sampler: bool,
) -> tuple[DataLoader, DataLoader]:
    paths = fold_directories(data_root, fold)
    train_set = CTSegmentationDataset(
        paths["train_ct"],
        paths["train_label"],
        image_size=image_size,
        training=True,
    )
    val_set = CTSegmentationDataset(
        paths["val_ct"],
        paths["val_label"],
        image_size=image_size,
        training=False,
    )

    sampler = None
    if balanced_sampler:
        sampler = WeightedRandomSampler(
            train_set.patient_balanced_weights,
            num_samples=len(train_set),
            replacement=True,
        )
    common = {
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader


def amp_context(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_or_validate_epoch(
    model: PRAISENet,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    training_loss: PRAISELoss,
    evaluation_loss: EvaluationLoss,
    optimizer: AdamW | None = None,
    scaler=None,
    amp: bool = False,
    gradient_accumulation_steps: int = 1,
) -> dict[str, float | int | str]:
    training = optimizer is not None
    model.train(training)
    accumulator = PatientMetricAccumulator(threshold)
    running_loss = 0.0
    context = torch.enable_grad if training else torch.no_grad

    if training:
        optimizer.zero_grad(set_to_none=True)
    with context():
        for step, batch in enumerate(loader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with amp_context(amp):
                outputs = model(images)
                if training:
                    loss = training_loss(outputs, masks)
                else:
                    loss = evaluation_loss(outputs["logits"], masks)
                probabilities = torch.sigmoid(outputs["logits"].float())

            if training:
                group_start = (
                    step // gradient_accumulation_steps
                ) * gradient_accumulation_steps
                group_end = min(
                    group_start + gradient_accumulation_steps, len(loader)
                )
                group_size = group_end - group_start
                scaler.scale(loss / group_size).backward()
                if step + 1 == group_end:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=5.0
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.detach())
            accumulator.update(
                probabilities,
                masks,
                batch["patient_id"],
                batch["name"],
            )

    values = accumulator.compute()
    values["loss"] = running_loss / max(1, len(loader))
    return values


@torch.no_grad()
def final_evaluation(
    model: PRAISENet,
    loader: DataLoader,
    criterion: EvaluationLoss,
    device: torch.device,
    threshold: float,
    amp: bool,
) -> dict[str, float | int | str]:
    model.eval()
    accumulator = PatientMetricAccumulator(threshold, store_volumes=True)
    running_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with amp_context(amp):
            outputs = model(images)
            loss = criterion(outputs["logits"], masks)
            probabilities = torch.sigmoid(outputs["logits"].float())
        running_loss += float(loss.detach())
        accumulator.update(
            probabilities,
            masks,
            batch["patient_id"],
            batch["name"],
        )
    values = accumulator.compute(include_hd95=True)
    values["loss"] = running_loss / max(1, len(loader))
    return values


def checkpoint_config(
    args: argparse.Namespace,
    fold: int,
    parameter_count: int,
) -> dict[str, object]:
    return {
        "model": "PRAISE-Net",
        "full_name": (
            "Proposal-guided Relative Anatomy and Intra-lesion Semantics "
            "Enhancement Network"
        ),
        "modules": {
            "CPG": "Context-Preserving Lesion Proposal Generator",
            "MTRG": "Multi-scale Transition Reliability Guidance",
            "RAISE": "Relative Anatomy and Intra-lesion Semantics Enhancer",
        },
        "fold": fold,
        "data_root": str(args.data_root.resolve()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "image_size": args.image_size,
        "input_channels": 1,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "threshold": args.threshold,
        "amp": args.amp,
        "deterministic": args.deterministic,
        "patient_balanced_sampler": not args.no_balanced_sampler,
        "best_model_metric": "validation patient-macro Dice (maximum)",
        "parameter_count": parameter_count,
        "training_loss": {
            "final_segmentation": 1.00,
            "proposal_segmentation": 0.35,
            "boundary": 0.20,
            "signed_distance": 0.10,
            "relative_region": 0.15,
        },
        "inference": (
            "single output from the unified proposal head; "
            "RAISE is training-only"
        ),
    }


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_fold(
    args: argparse.Namespace, fold: int, device: torch.device
) -> dict[str, object]:
    fold_dir = args.output_root / "praise_net" / f"fold_{fold}"
    metrics_path = fold_dir / "metrics.json"
    if args.skip_completed and metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        print(f"Skipping completed fold {fold}: {metrics_path}")
        return result

    set_seed(args.seed, args.deterministic)
    train_loader, val_loader = make_loaders(
        args.data_root,
        fold,
        args.image_size,
        args.batch_size,
        args.workers,
        device,
        balanced_sampler=not args.no_balanced_sampler,
    )
    model = PRAISENet(
        gradient_checkpointing=args.gradient_checkpointing
    ).to(device)
    training_loss = PRAISELoss().to(device)
    evaluation_loss = EvaluationLoss().to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=1e-6
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_dir / "best.pth"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config = checkpoint_config(args, fold, parameter_count)
    best_dice = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    print(
        f"PRAISE-Net fold={fold}; train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)}; "
        f"parameters={parameter_count / 1e6:.3f}M"
    )
    for epoch in range(1, args.epochs + 1):
        train_values = train_or_validate_epoch(
            model,
            train_loader,
            device,
            args.threshold,
            training_loss,
            evaluation_loss,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp_enabled,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        val_values = train_or_validate_epoch(
            model,
            val_loader,
            device,
            args.threshold,
            training_loss,
            evaluation_loss,
            amp=amp_enabled,
        )
        scheduler.step()

        saved = ""
        if float(val_values["dice"]) > best_dice:
            best_dice = float(val_values["dice"])
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": "PRAISE-Net",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_dice": best_dice,
                    "config": config,
                },
                checkpoint_path,
            )
            saved = f" saved-best -> {checkpoint_path}"
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": float(train_values["loss"]),
                "train_dice": float(train_values["dice"]),
                "train_iou": float(train_values["iou"]),
                "val_loss": float(val_values["loss"]),
                "val_dice": float(val_values["dice"]),
                "val_iou": float(val_values["iou"]),
            }
        )
        write_history(fold_dir / "history.csv", history)
        print(
            f"fold={fold} epoch={epoch:03d}/{args.epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.7f} "
            f"train_loss={train_values['loss']:.4f} "
            f"val_loss={val_values['loss']:.4f} "
            f"val_dice={val_values['dice']:.4f} "
            f"val_iou={val_values['iou']:.4f}{saved}"
        )
        if stale_epochs >= args.patience:
            print(
                f"Early stopping fold={fold} at epoch={epoch}; "
                f"best_epoch={best_epoch}, best_val_dice={best_dice:.6f}"
            )
            break

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    final_values = final_evaluation(
        model,
        val_loader,
        evaluation_loss,
        device,
        args.threshold,
        amp_enabled,
    )
    result: dict[str, object] = {
        "model": "PRAISE-Net",
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        **final_values,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return result


def update_summary(output_root: Path, new_results: list[dict[str, object]]) -> Path:
    summary_path = output_root / "praise_net" / "summary.csv"
    rows: dict[int, dict[str, object]] = {}
    if summary_path.is_file():
        with summary_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                rows[int(row["fold"])] = row
    for result in new_results:
        rows[int(result["fold"])] = result

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for fold in sorted(rows):
            writer.writerow(
                {field: rows[fold].get(field, "") for field in SUMMARY_FIELDS}
            )
    return summary_path


def write_five_fold_summary(
    output_root: Path, results: list[dict[str, object]]
) -> Path | None:
    by_fold = {int(result["fold"]): result for result in results}
    if set(by_fold) != {1, 2, 3, 4, 5}:
        return None
    metrics = ("dice", "iou", "precision", "recall", "accuracy", "hd95")
    summary = {
        metric: {
            "mean": statistics.mean(
                float(by_fold[fold][metric]) for fold in range(1, 6)
            ),
            "sample_standard_deviation": statistics.stdev(
                float(by_fold[fold][metric]) for fold in range(1, 6)
            ),
        }
        for metric in metrics
    }
    path = output_root / "praise_net" / "five_fold_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return path


def validate_args(args: argparse.Namespace) -> None:
    invalid_folds = [fold for fold in args.folds if fold not in range(1, 6)]
    if invalid_folds:
        raise ValueError(f"Folds must be in 1..5, got {invalid_folds}")
    if len(set(args.folds)) != len(args.folds):
        raise ValueError("--folds contains duplicates")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.image_size < 16 or args.image_size % 8 != 0:
        raise ValueError("--image-size must be at least 16 and divisible by 8")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    print(
        f"device={device}; model=PRAISE-Net; input=[B,1,{args.image_size},"
        f"{args.image_size}]; optimizer=AdamW; scheduler=CosineAnnealingLR; "
        f"deterministic={args.deterministic}; amp={args.amp}"
    )

    results = [
        train_one_fold(args, fold, device)
        for fold in args.folds
    ]
    summary_path = update_summary(args.output_root, results)
    five_fold_path = write_five_fold_summary(args.output_root, results)
    print(f"Fold results: {summary_path.resolve()}")
    if five_fold_path is not None:
        print(f"Five-fold summary: {five_fold_path.resolve()}")


if __name__ == "__main__":
    main()
