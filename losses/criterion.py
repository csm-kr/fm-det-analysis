"""SetCriterion — DiffusionDet 동치 (focal + L1 + GIoU + deep supervision).

deep supervision: 6 head 출력 모두 supervision (학습). loss = sum over K layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from losses.matcher import build_matcher
from utils.box_ops import box_xyxy_normalize


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
                       alpha: float = 0.25, gamma: float = 2.0,
                       reduction: str = "sum") -> torch.Tensor:
    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    return loss


class SetCriterion(nn.Module):
    def __init__(self, num_classes: int, matcher_cfg, focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, class_weight: float = 2.0,
                 l1_weight: float = 5.0, giou_weight: float = 2.0,
                 deep_supervision: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = build_matcher(matcher_cfg)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.class_weight = class_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.deep_supervision = deep_supervision

    def _loss_one_layer(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                        targets: list[dict], image_sizes: torch.Tensor) -> dict:
        """pred_logits: [B,N,C], pred_boxes: [B,N,4] xyxy."""
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        indices = self.matcher(outputs, targets, image_sizes)

        B, N, C = pred_logits.shape
        device = pred_logits.device

        # ─── classification (focal) ──────────────────────────────────
        target_cls = torch.zeros_like(pred_logits)  # [B,N,C]
        num_matched = 0
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            num_matched += pred_idx.numel()
            tgt_labels_b = targets[b]["labels"].to(device).long()
            target_cls[b, pred_idx, tgt_labels_b[tgt_idx]] = 1.0
        num_matched = max(num_matched, 1)

        loss_cls = sigmoid_focal_loss(pred_logits, target_cls,
                                        self.focal_alpha, self.focal_gamma,
                                        reduction="sum") / num_matched

        # ─── box (L1 + GIoU on matched) ──────────────────────────────
        l1_acc = pred_logits.new_zeros(())
        giou_acc = pred_logits.new_zeros(())
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue
            pred_b = pred_boxes[b][pred_idx]                   # [m, 4] xyxy image
            tgt_b = targets[b]["boxes"].to(device).float()[tgt_idx]
            H, W = image_sizes[b].tolist()
            img_hw = torch.tensor([H, W], device=device).float()
            pred_n = box_xyxy_normalize(pred_b, img_hw)
            tgt_n = box_xyxy_normalize(tgt_b, img_hw)
            l1_acc = l1_acc + F.l1_loss(pred_n, tgt_n, reduction="sum")
            giou = generalized_box_iou(pred_b, tgt_b).diag()  # [m]
            giou_acc = giou_acc + (1.0 - giou).sum()
        loss_l1 = l1_acc / num_matched
        loss_giou = giou_acc / num_matched

        return {
            "loss_cls": self.class_weight * loss_cls,
            "loss_l1": self.l1_weight * loss_l1,
            "loss_giou": self.giou_weight * loss_giou,
        }

    def forward(self, outputs: dict, targets: list[dict]) -> dict:
        """outputs from DiffusionDet train mode:
            pred_logits: [B, K, N, C]
            pred_boxes:  [B, K, N, 4]
            image_sizes: [B, 2]
        """
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        image_sizes = outputs["image_sizes"]
        B, K, N, C = logits.shape

        layers = list(range(K)) if self.deep_supervision else [K - 1]
        total = {"loss_cls": logits.new_zeros(()), "loss_l1": logits.new_zeros(()),
                 "loss_giou": logits.new_zeros(())}
        per_layer = []
        for k in layers:
            losses_k = self._loss_one_layer(logits[:, k], boxes[:, k], targets, image_sizes)
            per_layer.append(losses_k)
            for key in total:
                total[key] = total[key] + losses_k[key]

        loss_total = sum(total.values())
        return {**total, "loss_total": loss_total, "per_layer": per_layer}


def build_criterion(num_classes: int, cfg) -> SetCriterion:
    return SetCriterion(
        num_classes=num_classes,
        matcher_cfg=cfg.matcher,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        class_weight=cfg.class_weight,
        l1_weight=cfg.l1_weight,
        giou_weight=cfg.giou_weight,
        deep_supervision=cfg.deep_supervision,
    )
