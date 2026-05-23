"""bbox 좌표계 변환 + IoU/GIoU. detectron2 없이 자체 구현 (torchvision.ops 활용)."""

from __future__ import annotations

import torch
from torchvision.ops import box_iou, generalized_box_iou


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def box_xyxy_normalize(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    """boxes [N,4] xyxy → normalized [0,1] using image_size (H,W) per box or (H,W)."""
    if image_size.ndim == 1:
        h, w = image_size.unbind(-1)
        scale = torch.stack([w, h, w, h]).to(boxes.dtype)
    else:
        h, w = image_size.unbind(-1)
        scale = torch.stack([w, h, w, h], dim=-1).to(boxes.dtype)
    return boxes / scale


def box_xyxy_denormalize(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    if image_size.ndim == 1:
        h, w = image_size.unbind(-1)
        scale = torch.stack([w, h, w, h]).to(boxes.dtype)
    else:
        h, w = image_size.unbind(-1)
        scale = torch.stack([w, h, w, h], dim=-1).to(boxes.dtype)
    return boxes * scale
