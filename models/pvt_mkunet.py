"""BRMSNet: PVTv2-B1, gated HSSD decoding and full-resolution refinement."""

from __future__ import annotations

import math
from functools import partial
from typing import List, Sequence, Tuple

import timm
import torch
import torch.nn.functional as F
from timm.models import named_apply
from torch import nn

MODEL_NAME = "BRMSNet"
PVT_BACKBONE_NAME = "pvt_v2_b1"
PVT_FEATURE_CHANNELS: Tuple[int, ...] = (64, 128, 320, 512)
DECODER_CHANNELS: Tuple[int, ...] = (16, 32, 64, 96, 160)
KERNEL_SIZES: Tuple[int, ...] = (1, 3, 5)


def _init_normal_weights(module: nn.Module, name: str) -> None:
    """Initialize convolution and normalization layers."""

    del name
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def _initialize(module: nn.Module) -> None:
    named_apply(partial(_init_normal_weights), module)


def channel_shuffle(features: torch.Tensor, groups: int) -> torch.Tensor:
    """Shuffle channels between groups after multi-kernel aggregation."""

    batch_size, channels, height, width = features.shape
    if groups <= 0 or channels % groups != 0:
        raise ValueError(
            f"channels ({channels}) must be divisible by groups ({groups})"
        )
    channels_per_group = channels // groups
    features = features.view(batch_size, groups, channels_per_group, height, width)
    features = features.transpose(1, 2).contiguous()
    return features.view(batch_size, channels, height, width)


class ChannelAttention(nn.Module):
    """Channel attention based on global average and maximum pooling."""

    def __init__(self, channels: int, ratio: int) -> None:
        super().__init__()
        reduced_channels = channels // min(channels, ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        _initialize(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_attention = self.fc2(self.activation(self.fc1(self.avg_pool(features))))
        maximum_attention = self.fc2(self.activation(self.fc1(self.max_pool(features))))
        return self.sigmoid(average_attention + maximum_attention)


class SpatialAttention(nn.Module):
    """Spatial attention based on channel-wise mean and maximum maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()
        _initialize(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        channel_mean = torch.mean(features, dim=1, keepdim=True)
        channel_max = torch.max(features, dim=1, keepdim=True).values
        descriptor = torch.cat((channel_mean, channel_max), dim=1)
        return self.sigmoid(self.conv(descriptor))


class GroupedAttentionGate(nn.Module):
    """Gate one encoder skip with its corresponding decoder feature."""

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        intermediate_channels: int,
        groups: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                intermediate_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                intermediate_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=True,
            ),
            nn.BatchNorm2d(intermediate_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.activation = nn.ReLU(inplace=True)
        _initialize(self)

    def forward(
        self,
        decoder_features: torch.Tensor,
        skip_features: torch.Tensor,
    ) -> torch.Tensor:
        decoder_projection = self.W_g(decoder_features)
        skip_projection = self.W_x(skip_features)
        attention = self.psi(self.activation(decoder_projection + skip_projection))
        return skip_features * attention


class MultiKernelDepthwiseConv(nn.Module):
    """Apply 1x1, 3x3 and 5x5 depthwise branches in parallel."""

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int],
        stride: int,
    ) -> None:
        super().__init__()
        self.dwconvs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2,
                        groups=channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for kernel_size in kernel_sizes
            ]
        )
        _initialize(self)

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        return [depthwise_conv(features) for depthwise_conv in self.dwconvs]


class MultiKernelInvertedResidualBlock(nn.Module):
    """Inverted residual decoder block with parallel depthwise kernels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion_factor: int = 2,
        kernel_sizes: Sequence[int] = KERNEL_SIZES,
    ) -> None:
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        self.in_c = in_channels
        self.out_c = out_channels
        self.combined_channels = expanded_channels
        self.pconv1 = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            expanded_channels,
            kernel_sizes,
            stride=1,
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        _initialize(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expanded = self.pconv1(features)
        branch_features = self.multi_scale_dwconv(expanded)
        combined = branch_features[0]
        for branch in branch_features[1:]:
            combined = combined + branch
        shuffled = channel_shuffle(
            combined,
            math.gcd(self.combined_channels, self.out_c),
        )
        projected = self.pconv2(shuffled)
        residual = self.conv1x1(features) if self.in_c != self.out_c else features
        return residual + projected


def build_decoder_block(
    in_channels: int,
    out_channels: int,
) -> nn.Sequential:
    """Build one MRIR stage."""

    return nn.Sequential(MultiKernelInvertedResidualBlock(in_channels, out_channels))


class StemConv(nn.Module):
    """Extract the H/2 skip feature that PVTv2 does not provide."""

    def __init__(self, out_channels: int = DECODER_CHANNELS[0]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            3,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        _initialize(self)

    def full_resolution_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return the pre-pooling feature used by the boundary refiner."""

        return self.act(self.bn(self.conv(images)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pool(self.full_resolution_features(images))


class FullResolutionRefinement(nn.Module):
    """Recover thin structures by refining the upsampled decoder at H x W."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)
        _initialize(self)
        # A zero residual preserves decoder features at initialization.
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(
        self,
        decoder_features: torch.Tensor,
        shallow_features: torch.Tensor,
    ) -> torch.Tensor:
        refined = self.fuse(torch.cat((decoder_features, shallow_features), dim=1))
        return self.activation(decoder_features + refined)


class ChannelAdapter(nn.Module):
    """Map one PVT feature width to the decoder's expected width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        _initialize(self)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(features)))


class BRMSNet(nn.Module):
    """Four training heads ordered as full, eighth, quarter and half resolution."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        channels = DECODER_CHANNELS
        self.encoder = timm.create_model(
            PVT_BACKBONE_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.stem = StemConv(channels[0])
        self.adapter_bottleneck = ChannelAdapter(PVT_FEATURE_CHANNELS[3], channels[4])
        self.adapter_skip4 = ChannelAdapter(PVT_FEATURE_CHANNELS[2], channels[3])
        self.adapter_skip3 = ChannelAdapter(PVT_FEATURE_CHANNELS[1], channels[2])
        self.adapter_skip2 = ChannelAdapter(PVT_FEATURE_CHANNELS[0], channels[1])

        self.decoder1 = build_decoder_block(channels[4], channels[3])
        self.decoder2 = build_decoder_block(channels[3], channels[2])
        self.decoder3 = build_decoder_block(channels[2], channels[1])
        self.decoder4 = build_decoder_block(channels[1], channels[0])
        self.decoder5 = build_decoder_block(channels[0], channels[0])
        self.full_resolution_refinement = FullResolutionRefinement(channels[0])

        self.AG1 = GroupedAttentionGate(
            channels[3], channels[3], channels[3] // 2, channels[3] // 2
        )
        self.AG2 = GroupedAttentionGate(
            channels[2], channels[2], channels[2] // 2, channels[2] // 2
        )
        self.AG3 = GroupedAttentionGate(
            channels[1], channels[1], channels[1] // 2, channels[1] // 2
        )
        self.AG4 = GroupedAttentionGate(
            channels[0], channels[0], channels[0] // 2, channels[0] // 2
        )

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=16)
        self.CA5 = ChannelAttention(channels[0], ratio=16)
        self.SA = SpatialAttention()

        self.out1 = nn.Conv2d(channels[2], 1, kernel_size=1)
        self.out2 = nn.Conv2d(channels[1], 1, kernel_size=1)
        self.out3 = nn.Conv2d(channels[0], 1, kernel_size=1)
        self.out4 = nn.Conv2d(channels[0], 1, kernel_size=1)

    def _extract_encoder_features(self, images: torch.Tensor) -> List[torch.Tensor]:
        shallow_full = self.stem.full_resolution_features(images)
        skip_half = self.stem.pool(shallow_full)
        pvt_features = self.encoder(images)
        return [
            shallow_full,
            skip_half,
            self.adapter_skip2(pvt_features[0]),
            self.adapter_skip3(pvt_features[1]),
            self.adapter_skip4(pvt_features[2]),
            self.adapter_bottleneck(pvt_features[3]),
        ]

    def _attention_upsample(
        self,
        features: torch.Tensor,
        channel_attention: ChannelAttention,
        decoder: nn.Module,
        output_size: Sequence[int],
    ) -> torch.Tensor:
        features = channel_attention(features) * features
        features = self.SA(features) * features
        features = decoder(features)
        features = F.interpolate(
            features,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        return F.relu(features)

    def _decode(
        self,
        encoder_features: Sequence[torch.Tensor],
        return_all: bool,
    ) -> List[torch.Tensor]:
        (
            shallow_full,
            skip_half,
            skip_quarter,
            skip_eighth,
            skip_sixteenth,
            bottleneck,
        ) = encoder_features
        output_size = shallow_full.shape[-2:]

        decoder_sixteenth = self._attention_upsample(
            bottleneck, self.CA1, self.decoder1, skip_sixteenth.shape[-2:]
        )
        decoder_sixteenth = decoder_sixteenth + self.AG1(
            decoder_sixteenth, skip_sixteenth
        )

        decoder_eighth = self._attention_upsample(
            decoder_sixteenth, self.CA2, self.decoder2, skip_eighth.shape[-2:]
        )
        decoder_eighth = decoder_eighth + self.AG2(decoder_eighth, skip_eighth)

        decoder_quarter = self._attention_upsample(
            decoder_eighth, self.CA3, self.decoder3, skip_quarter.shape[-2:]
        )
        decoder_quarter = decoder_quarter + self.AG3(decoder_quarter, skip_quarter)

        decoder_half = self._attention_upsample(
            decoder_quarter, self.CA4, self.decoder4, skip_half.shape[-2:]
        )
        decoder_half = decoder_half + self.AG4(decoder_half, skip_half)

        decoder_full = self._attention_upsample(
            decoder_half, self.CA5, self.decoder5, output_size
        )
        decoder_full = self.full_resolution_refinement(decoder_full, shallow_full)
        logits_full = self.out4(decoder_full)
        if not return_all:
            return [logits_full]
        auxiliary_logits = [
            F.interpolate(
                head(features), size=output_size, mode="bilinear", align_corners=False
            )
            for head, features in (
                (self.out1, decoder_eighth),
                (self.out2, decoder_quarter),
                (self.out3, decoder_half),
            )
        ]
        return [logits_full, *auxiliary_logits]

    def forward(
        self,
        images: torch.Tensor,
        return_all: bool = False,
    ) -> List[torch.Tensor]:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.shape[1] != 3:
            raise ValueError(f"Expected 1 or 3 input channels, got {images.shape[1]}")
        return self._decode(self._extract_encoder_features(images), return_all)


PVTMKUNetB1 = BRMSNet
