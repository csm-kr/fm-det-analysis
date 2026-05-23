"""SimOTA matcher — DiffusionDet 동치.

각 GT 에 대해 dynamic k (top-10 GIoU IoU 합) 만큼 가장 낮은 cost 박스 매칭.
한 박스가 여러 GT 에 매칭되면 가장 낮은 cost GT 만 선택.

cost = cost_class * focal_cost + cost_bbox * L1_cost + cost_giou * (1 - GIoU)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

from utils.box_ops import box_xyxy_to_cxcywh, box_xyxy_normalize


class SimOTAMatcher(nn.Module):
    """Returns list[(pred_idx, tgt_idx)] per batch element."""

    def __init__(self, cost_class: float = 2.0, cost_bbox: float = 5.0,
                 cost_giou: float = 2.0, ota_k: int = 5, center_radius: float = 2.5,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.ota_k = ota_k
        self.center_radius = center_radius
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict],
                image_sizes: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """outputs:
            pred_logits: [B, N, C] (single layer)
            pred_boxes:  [B, N, 4] xyxy image coords
           targets: list of dicts with 'boxes' (xyxy image), 'labels'.
           image_sizes: [B, 2] (H, W).
        Returns: list[B] of (pred_idx, tgt_idx) — matched indices.
        """
        B = outputs["pred_logits"].shape[0]
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []
        for b in range(B):
            tgt = targets[b]
            n_tgt = tgt["boxes"].shape[0]
            if n_tgt == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue
            logits_b = outputs["pred_logits"][b]  # [N, C]
            boxes_b = outputs["pred_boxes"][b]     # [N, 4] xyxy image
            N = logits_b.shape[0]

            tgt_boxes = tgt["boxes"].to(logits_b.device).float()       # [M, 4] xyxy image
            tgt_labels = tgt["labels"].to(logits_b.device).long()      # [M]

            # focal cost ─ [N, M]
            prob = logits_b.sigmoid()
            neg_cost = (1 - self.focal_alpha) * (prob ** self.focal_gamma) * \
                       (-(1 - prob + 1e-8).log())
            pos_cost = self.focal_alpha * ((1 - prob) ** self.focal_gamma) * \
                       (-(prob + 1e-8).log())
            cost_cls = pos_cost[:, tgt_labels] - neg_cost[:, tgt_labels]  # [N, M]

            # L1 cost on normalized cxcywh
            H, W = image_sizes[b].tolist()
            img_hw = torch.tensor([H, W], device=logits_b.device).float()
            pred_norm = box_xyxy_normalize(boxes_b, img_hw)
            tgt_norm = box_xyxy_normalize(tgt_boxes, img_hw)
            pred_cxcywh = box_xyxy_to_cxcywh(pred_norm)
            tgt_cxcywh = box_xyxy_to_cxcywh(tgt_norm)
            cost_bbox = torch.cdist(pred_cxcywh, tgt_cxcywh, p=1)  # [N, M]

            # GIoU cost (absolute image coords)
            giou = generalized_box_iou(boxes_b, tgt_boxes)  # [N, M]
            cost_giou = -giou

            cost = (self.cost_class * cost_cls
                    + self.cost_bbox * cost_bbox
                    + self.cost_giou * cost_giou)

            # center-in-box prior (DiffusionDet 동치) — outside boxes → inf
            pred_ctr = (boxes_b[:, :2] + boxes_b[:, 2:]) / 2  # [N, 2] xy
            tgt_x1y1 = tgt_boxes[:, :2]  # [M, 2]
            tgt_x2y2 = tgt_boxes[:, 2:]  # [M, 2]
            in_box = (
                (pred_ctr[:, None, 0] >= tgt_x1y1[:, 0]) &
                (pred_ctr[:, None, 1] >= tgt_x1y1[:, 1]) &
                (pred_ctr[:, None, 0] <= tgt_x2y2[:, 0]) &
                (pred_ctr[:, None, 1] <= tgt_x2y2[:, 1])
            )  # [N, M]
            cost = cost + (~in_box).float() * 1e8

            # dynamic k = sum of top-10 IoU per GT, clamp min 1
            iou = giou.clamp(min=0.0)
            top_iou, _ = iou.topk(min(10, N), dim=0)  # [10, M]
            dynamic_k = top_iou.sum(dim=0).clamp(min=1).long().clamp(max=N)  # [M]

            # matching: for each GT, pick top-k lowest cost
            matched_pred_to_gt = torch.full((N,), -1, dtype=torch.long, device=cost.device)
            matched_cost = torch.full((N,), float("inf"), device=cost.device)
            for m in range(n_tgt):
                k = dynamic_k[m].item()
                if k <= 0:
                    continue
                vals, idxs = cost[:, m].topk(k, largest=False)
                # 한 박스가 여러 GT 와 충돌 시 가장 낮은 cost GT 선택
                for v, idx in zip(vals.tolist(), idxs.tolist()):
                    if v < matched_cost[idx].item():
                        matched_cost[idx] = v
                        matched_pred_to_gt[idx] = m
            valid = matched_pred_to_gt >= 0
            pred_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
            tgt_idx = matched_pred_to_gt[valid]
            indices.append((pred_idx.cpu(), tgt_idx.cpu()))

        return indices


def build_matcher(cfg) -> SimOTAMatcher:
    return SimOTAMatcher(
        cost_class=cfg.cost_class,
        cost_bbox=cfg.cost_bbox,
        cost_giou=cfg.cost_giou,
        ota_k=cfg.get("ota_k", 5),
        center_radius=cfg.get("center_radius", 2.5),
    )
