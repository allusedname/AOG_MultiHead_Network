from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from partcat_hkg.models.backbones import ResNetFeatureBackbone


class DenseFeatureBackboneV7(nn.Module, ABC):
    """Backbone contract required by the dynamic Stage-1 decoder.

    Implementations must expose ``skip_ch``, ``low_ch``, and ``high_ch`` and
    return a dictionary with ``skip``, ``low``, and ``high`` feature maps.
    ``skip`` is the highest-resolution map and ``high`` is the most semantic map.
    """

    skip_ch: int
    low_ch: int
    high_ch: int

    @abstractmethod
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class ResNetDenseFeatureBackboneV7(DenseFeatureBackboneV7):
    def __init__(self, name: str = "resnet18", *, pretrained: bool = False, freeze: bool = False) -> None:
        super().__init__()
        self.model = ResNetFeatureBackbone(name, pretrained=pretrained, freeze=freeze)
        self.skip_ch = int(self.model.skip_ch)
        self.low_ch = int(self.model.low_ch)
        self.high_ch = int(self.model.high_ch)
        self.name = str(self.model.name)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(image)


def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4:
        # Some timm models return NHWC feature maps.
        if tokens.shape[1] < tokens.shape[-1] and tokens.shape[1] <= 64:
            return tokens.permute(0, 3, 1, 2).contiguous()
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor [B,N,C] or map [B,C,H,W], got {tuple(tokens.shape)}")
    batch, n_tokens, channels = tokens.shape
    side = int(round(math.sqrt(n_tokens)))
    if side * side == n_tokens:
        return tokens.transpose(1, 2).reshape(batch, channels, side, side)
    side = int(round(math.sqrt(max(1, n_tokens - 1))))
    if side * side == n_tokens - 1:
        return tokens[:, 1:].transpose(1, 2).reshape(batch, channels, side, side)
    raise ValueError(f"Cannot infer a square feature map from {tuple(tokens.shape)}")


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in (
            "x_norm_patchtokens",
            "patch_tokens",
            "last_hidden_state",
            "x_prenorm",
            "features",
            "feature_map",
        ):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        tensors = [value for value in output.values() if torch.is_tensor(value)]
        if tensors:
            return tensors[-1]
    if isinstance(output, (list, tuple)):
        tensors = [value for value in output if torch.is_tensor(value)]
        if tensors:
            return tensors[-1]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported timm feature output type: {type(output)!r}")


class TimmDenseFeatureBackboneV7(DenseFeatureBackboneV7):
    """Dense adapter for ConvNeXt, Swin, DINO/ViT, and other timm models.

    Hierarchical models use ``features_only=True``. Token-only ViTs fall back to
    ``forward_features`` and a learned-free interpolation pyramid. This makes the
    ResNet branch optional while preserving the decoder's three-scale contract.
    """

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool = False,
        freeze: bool = False,
        probe_size: int = 224,
    ) -> None:
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("timm is required for non-ResNet dense backbones") from exc

        self.model_name = str(model_name)
        self.mode = "features_only"
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=bool(pretrained),
                features_only=True,
                out_indices=(1, 2, 3),
            )
            channels = list(self.model.feature_info.channels())
            if len(channels) < 3:
                raise RuntimeError("features_only model returned fewer than three levels")
            self.skip_ch, self.low_ch, self.high_ch = [int(x) for x in channels[-3:]]
        except Exception:
            self.mode = "tokens"
            self.model = timm.create_model(self.model_name, pretrained=bool(pretrained), num_classes=0)
            channels = int(getattr(self.model, "num_features", 0) or getattr(self.model, "embed_dim", 0) or 0)
            if channels <= 0:
                with torch.no_grad():
                    probe = torch.zeros(1, 3, int(probe_size), int(probe_size))
                    feature = _tokens_to_map(_extract_tensor(self.model.forward_features(probe)))
                    channels = int(feature.shape[1])
            self.skip_ch = self.low_ch = self.high_ch = channels

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.freeze = bool(freeze)
        self.name = f"timm:{self.model_name}:{self.mode}"

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.freeze:
            self.model.eval()
        if self.mode == "features_only":
            output = self.model(image)
            if not isinstance(output, (list, tuple)) or len(output) < 3:
                raise RuntimeError("timm features_only backbone did not return three feature levels")
            maps = list(output)[-3:]
            # timm feature maps are usually NCHW, but some models return NHWC.
            converted = []
            for feature, expected_channels in zip(maps, (self.skip_ch, self.low_ch, self.high_ch)):
                if feature.ndim != 4:
                    feature = _tokens_to_map(feature)
                elif feature.shape[1] != expected_channels and feature.shape[-1] == expected_channels:
                    feature = feature.permute(0, 3, 1, 2).contiguous()
                converted.append(feature)
            return {"skip": converted[0], "low": converted[1], "high": converted[2]}

        output = self.model.forward_features(image)
        high = _tokens_to_map(_extract_tensor(output))
        # Token ViTs have one semantic map. A lightweight interpolation pyramid
        # preserves the interface; the downstream FPN/refinement learns the fusion.
        low = F.interpolate(high, scale_factor=2.0, mode="bilinear", align_corners=False)
        skip = F.interpolate(high, scale_factor=4.0, mode="bilinear", align_corners=False)
        return {"skip": skip, "low": low, "high": high}


def build_dense_feature_backbone_v7(
    name: str,
    *,
    pretrained: bool = False,
    freeze: bool = False,
    probe_size: int = 224,
) -> DenseFeatureBackboneV7:
    normalized = str(name).strip()
    if normalized.lower() in {"resnet18", "resnet50", "tiny"}:
        return ResNetDenseFeatureBackboneV7(normalized.lower(), pretrained=pretrained, freeze=freeze)
    return TimmDenseFeatureBackboneV7(normalized, pretrained=pretrained, freeze=freeze, probe_size=probe_size)
