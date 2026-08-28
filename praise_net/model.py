"""PRAISE-Net architecture used in the manuscript.

The public implementation uses the paper terminology throughout:

* CPG: Context-Preserving Lesion Proposal Generator
* MTRG: Multi-scale Transition Reliability Guidance
* RAISE: Relative Anatomy and Intra-lesion Semantics Enhancer

RAISE is a training-time auxiliary branch. The inference logits are produced
only by the unified proposal head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            group_norm(out_channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return self.activation(x + residual)


class SeparableResidualBlock(nn.Module):
    """Memory-conscious residual block for high-resolution auxiliary paths."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.pointwise_in = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.depthwise = ConvNormAct(
            out_channels, out_channels, kernel_size=3, groups=out_channels
        )
        self.pointwise_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            group_norm(out_channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.pointwise_in(x)
        x = self.depthwise(x)
        x = self.dropout(x)
        x = self.pointwise_out(x)
        return self.activation(x + residual)


class DownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.down = ConvNormAct(in_channels, out_channels, stride=2)
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels, dropout),
            ResidualBlock(out_channels, out_channels, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class MultiRangeContext(nn.Module):
    """Same-scale local/global context aggregation at H/8 x W/8."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_channels = channels // 4
        self.local_branches = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(
                        channels,
                        channels,
                        kernel_size=3,
                        dilation=dilation,
                        groups=channels,
                    ),
                    ConvNormAct(channels, branch_channels, kernel_size=1),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.global_projection = ConvNormAct(channels, branch_channels, kernel_size=1)
        self.residual_fusion = ResidualBlock(branch_channels * 4, channels, dropout=0.08)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_features = [branch(x) for branch in self.local_branches]
        global_feature = F.adaptive_avg_pool2d(x, 1)
        global_feature = self.global_projection(global_feature)
        global_feature = F.interpolate(
            global_feature, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.residual_fusion(
            torch.cat([*local_features, global_feature], dim=1)
        ) + x


class ContextPreservingLesionProposalEncoder(nn.Module):
    """Encoding and same-scale context part of CPG."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = [base_channels * value for value in (1, 2, 4, 8)]
        self.encoder1 = nn.Sequential(
            ResidualBlock(in_channels, c1),
            ResidualBlock(c1, c1),
        )
        self.encoder2 = DownStage(c1, c2, dropout=0.02)
        self.encoder3 = DownStage(c2, c3, dropout=0.05)
        self.encoder4 = DownStage(c3, c4, dropout=0.08)
        self.context = MultiRangeContext(c4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        context = self.context(e4)
        return e1, e2, e3, e4, context


class TransitionReliabilityUnit(nn.Module):
    """One MTRG unit producing guided feature S_i and reliability logits R_i."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.transition_fusion = SeparableResidualBlock(channels * 3, channels)
        self.reliability_head = nn.Sequential(
            ConvNormAct(channels, max(8, channels // 2), kernel_size=3),
            nn.Conv2d(max(8, channels // 2), 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean3 = F.avg_pool2d(x, 3, stride=1, padding=1)
        mean5 = F.avg_pool2d(x, 5, stride=1, padding=2)
        local_contrast = (x - mean3).abs()
        local_transition = (mean3 - mean5).abs()
        transition_feature = self.transition_fusion(
            torch.cat([x, local_contrast, local_transition], dim=1)
        )
        reliability_logits = self.reliability_head(transition_feature)
        reliability = torch.sigmoid(reliability_logits)
        guided_feature = x + transition_feature * reliability
        return guided_feature, reliability_logits


class MultiScaleTransitionReliabilityGuidance(nn.Module):
    """MTRG at E1, E2, E3 and the same-scale context representation."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        channels = [base_channels * value for value in (1, 2, 4, 8)]
        self.mtrg1 = TransitionReliabilityUnit(channels[0])
        self.mtrg2 = TransitionReliabilityUnit(channels[1])
        self.mtrg3 = TransitionReliabilityUnit(channels[2])
        self.mtrg4 = TransitionReliabilityUnit(channels[3])

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        s1, r1 = self.mtrg1(e1)
        s2, r2 = self.mtrg2(e2)
        s3, r3 = self.mtrg3(e3)
        s4, r4 = self.mtrg4(context)
        return [s1, s2, s3, s4], [r1, r2, r3, r4]


class ReliabilityGuidedDecoderBlock(nn.Module):
    """Scale-aligned CPG reconstruction guided by an MTRG reliability field."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.up_projection = ConvNormAct(in_channels, out_channels, kernel_size=1)
        self.skip_projection = (
            nn.Identity()
            if skip_channels == out_channels
            else ConvNormAct(skip_channels, out_channels, kernel_size=1)
        )
        hidden = max(8, out_channels // 2)
        self.reliability_attention = nn.Sequential(
            nn.Conv2d(out_channels * 2 + 1, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.residual_fusion = ResidualBlock(out_channels * 2, out_channels)

    def forward(
        self,
        decoder_feature: torch.Tensor,
        guided_skip: torch.Tensor,
        reliability_logits: torch.Tensor,
    ) -> torch.Tensor:
        decoder_feature = F.interpolate(
            decoder_feature,
            size=guided_skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder_feature = self.up_projection(decoder_feature)
        guided_skip = self.skip_projection(guided_skip)
        reliability = torch.sigmoid(
            F.interpolate(
                reliability_logits,
                size=guided_skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        attention = self.reliability_attention(
            torch.cat([decoder_feature, guided_skip, reliability], dim=1)
        )
        guided_skip = guided_skip * (1.0 + attention * reliability)
        return self.residual_fusion(torch.cat([decoder_feature, guided_skip], dim=1))


class ProposalReconstructionDecoder(nn.Module):
    """Three-stage spatially aligned reconstruction in CPG."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = [base_channels * value for value in (1, 2, 4, 8)]
        self.guided_decoder3 = ReliabilityGuidedDecoderBlock(c4, c3, c3)
        self.guided_decoder2 = ReliabilityGuidedDecoderBlock(c3, c2, c2)
        self.guided_decoder1 = ReliabilityGuidedDecoderBlock(c2, c1, c1)

    def forward(
        self,
        guided_features: list[torch.Tensor],
        reliability_logits: list[torch.Tensor],
    ) -> torch.Tensor:
        s1, s2, s3, s4 = guided_features
        r1, r2, r3, _r4 = reliability_logits
        d3 = self.guided_decoder3(s4, s3, r3)
        d2 = self.guided_decoder2(d3, s2, r2)
        d1 = self.guided_decoder1(d2, s1, r1)
        return d1


class UnifiedProposalHead(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(ResidualBlock(channels, channels), nn.Conv2d(channels, 1, 1))


class RelativeAnatomyIntraLesionSemanticsEnhancer(nn.Module):
    """RAISE training branch for boundary and signed relative-position supervision."""

    def __init__(self, feature_channels: int = 32, number_of_scales: int = 4) -> None:
        super().__init__()
        self.structure_fusion = ResidualBlock(
            feature_channels + number_of_scales, feature_channels
        )
        self.boundary_head = nn.Sequential(
            SeparableResidualBlock(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 1, 1),
        )
        self.signed_distance_head = nn.Sequential(
            SeparableResidualBlock(feature_channels, feature_channels),
            nn.Conv2d(feature_channels, 1, 1),
        )

    def forward(
        self,
        proposal_feature: torch.Tensor,
        reliability_logits: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        full_resolution_fields = [
            field
            if field.shape[-2:] == proposal_feature.shape[-2:]
            else F.interpolate(
                field,
                size=proposal_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            for field in reliability_logits
        ]
        structure_feature = self.structure_fusion(
            torch.cat([proposal_feature, *full_resolution_fields], dim=1)
        )
        boundary_logits = self.boundary_head(structure_feature)
        signed_distance = torch.tanh(self.signed_distance_head(structure_feature))
        return {
            "structure_feature": structure_feature,
            "boundary_logits": boundary_logits,
            "signed_distance": signed_distance,
            "boundary_probability": torch.sigmoid(boundary_logits),
            "boundary_proximity": 1.0 - signed_distance.abs(),
        }


class PRAISENet(nn.Module):
    """Proposal-guided Relative Anatomy and Intra-lesion Semantics Enhancement Network."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError("PRAISE-Net expects a single CT channel")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.cpg_encoder = ContextPreservingLesionProposalEncoder(
            in_channels, base_channels
        )
        self.mtrg = MultiScaleTransitionReliabilityGuidance(base_channels)
        self.cpg_decoder = ProposalReconstructionDecoder(base_channels)
        self.unified_proposal_head = UnifiedProposalHead(base_channels)
        self.raise_enhancer = RelativeAnatomyIntraLesionSemanticsEnhancer(
            base_channels
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.unified_proposal_head[-1].weight, std=0.01)
        nn.init.constant_(self.unified_proposal_head[-1].bias, -2.0)

    def _run(self, module: nn.Module, *inputs):
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if x.ndim != 4 or x.shape[1] != 1:
            raise RuntimeError(f"Expected [B, 1, H, W], received {tuple(x.shape)}")

        e1, e2, e3, _e4, context = self._run(self.cpg_encoder, x)
        guided_features, reliability_logits = self._run(
            self.mtrg, e1, e2, e3, context
        )
        proposal_feature = self._run(
            self.cpg_decoder, guided_features, reliability_logits
        )
        proposal_logits = self._run(self.unified_proposal_head, proposal_feature)
        raise_outputs = self._run(
            self.raise_enhancer, proposal_feature, reliability_logits
        )

        # RAISE supplies training-time structural supervision only.
        # The final inference logits are the unified proposal logits.
        return {
            "logits": proposal_logits,
            "proposal_logits": proposal_logits,
            "proposal_feature": proposal_feature,
            "mtrg_reliability_logits": reliability_logits,
            "raise_boundary_logits": raise_outputs["boundary_logits"],
            "raise_signed_distance": raise_outputs["signed_distance"],
            "raise_boundary_probability": raise_outputs["boundary_probability"],
            "raise_boundary_proximity": raise_outputs["boundary_proximity"],
        }
