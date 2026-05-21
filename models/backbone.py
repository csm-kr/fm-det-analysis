"""ResNet50 + FPN backbone — torchvision 만 사용 (detectron2 금지).

DiffusionDet 동치:
- ResNet50 ImageNet pretrained
- FPN out_channels=256, 4 scale (p2 /4, p3 /8, p4 /16, p5 /32)
- freeze_at=2 (stem + res2 freeze, BN frozen)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d


class ResNet50FPN(nn.Module):
    """returns list[Tensor[B,256,Hi,Wi]] for i in [p2, p3, p4, p5]."""

    OUT_CHANNELS = 256
    STRIDES = (4, 8, 16, 32)

    def __init__(self, pretrained: bool = True, freeze_at: int = 2,
                 fpn_out_channels: int = 256):
        super().__init__()
        assert fpn_out_channels == self.OUT_CHANNELS, "torchvision FPN 은 256 고정"
        weights = "DEFAULT" if pretrained else None
        # detectron2 freeze_at=2 (stem+res2 freeze) → trainable_layers=3 (res3,4,5)
        trainable_layers = max(0, 5 - freeze_at)
        self.body = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=weights,
            trainable_layers=trainable_layers,
            returned_layers=[1, 2, 3, 4],  # res2 → res5
            extra_blocks=None,  # DiffusionDet 은 p6 사용 안 함
            norm_layer=FrozenBatchNorm2d,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = self.body(x)
        return [out["0"], out["1"], out["2"], out["3"]]


def build_backbone(cfg=None) -> ResNet50FPN:
    if cfg is None:
        return ResNet50FPN()
    return ResNet50FPN(
        pretrained=cfg.get("pretrained", True),
        freeze_at=cfg.get("freeze_at", 2),
        fpn_out_channels=cfg.get("fpn_out_channels", 256),
    )
