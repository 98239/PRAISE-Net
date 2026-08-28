from __future__ import annotations

import torch

from praise_net.losses import PRAISELoss
from praise_net.model import PRAISENet


def test_main_model_forward_and_loss() -> None:
    torch.manual_seed(42)
    model = PRAISENet()
    images = torch.randn(1, 1, 64, 64)
    targets = (torch.rand(1, 1, 64, 64) > 0.9).float()

    outputs = model(images)
    assert outputs["logits"].shape == targets.shape
    assert torch.equal(outputs["logits"], outputs["proposal_logits"])
    assert len(outputs["mtrg_reliability_logits"]) == 4
    assert outputs["raise_boundary_logits"].shape == targets.shape
    assert outputs["raise_signed_distance"].shape == targets.shape
    assert sum(parameter.numel() for parameter in model.parameters()) == 6_514_682

    loss = PRAISELoss()(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
