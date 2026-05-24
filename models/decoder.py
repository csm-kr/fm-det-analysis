"""DiffusionDet Decoder — 6 DetectionHead 반복 (iterative refinement).

각 head:
1. RoIAlign per FPN level → roi_features [B*N, 256, 7, 7]
2. self-attn on proposal_features [N, B, 256]
3. DynamicConv (instance interaction: roi feature × dynamic params from pro)
4. FFN
5. cls_logits + box_delta → apply on proposal_boxes → new boxes (next head)

Deep supervision: 6 head 출력 모두 loss 에 사용 (학습), 마지막만 (평가).
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class SinusoidalPositionEmbedding(nn.Module):
    """time embedding (diffusion timestep → vector)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DynamicConv(nn.Module):
    """Per-instance dynamic convolution — params generated from proposal_features.

    roi_features: [B*N, 49, 256] → output [B*N, 256] via two dynamic linears.
    """

    def __init__(self, d_model: int = 256, dim_dynamic: int = 64, pooler_resolution: int = 7):
        super().__init__()
        self.d_model = d_model
        self.dim_dynamic = dim_dynamic
        self.num_params = d_model * dim_dynamic

        self.dynamic_layer = nn.Linear(d_model, 2 * self.num_params)
        self.norm1 = nn.LayerNorm(dim_dynamic)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(inplace=True)
        num_output = d_model * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, pro_features: torch.Tensor, roi_features: torch.Tensor) -> torch.Tensor:
        """pro: [1, B*N, d_model]. roi: [B*N, 49, d_model]."""
        # generate params
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)  # [B*N, 1, 2*num_params]
        param1 = parameters[:, :, : self.num_params].view(-1, self.d_model, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.d_model)

        x = torch.bmm(roi_features, param1)  # [B*N, 49, dim_dynamic]
        x = self.norm1(x)
        x = self.activation(x)
        x = torch.bmm(x, param2)  # [B*N, 49, d_model]
        x = self.norm2(x)
        x = self.activation(x)

        x = x.flatten(1)  # [B*N, 49*d_model]
        x = self.out_layer(x)  # [B*N, d_model]
        x = self.norm3(x)
        x = self.activation(x)
        return x


class DetectionHead(nn.Module):
    """One iteration of DiffusionDet head — refine box + class.

    Inputs:
        features: list of FPN features [B, 256, Hi, Wi] (p2..p5)
        bboxes: Tensor[B, N, 4] xyxy in image coords
        proposal_features: Tensor[B, N, d_model] (init from time_embed)
        time_embed: Tensor[B, d_model*4]
        image_sizes: Tensor[B, 2] (H, W) for RoIAlign scaling
    Returns:
        new_bboxes: Tensor[B, N, 4] xyxy refined
        class_logits: Tensor[B, N, num_classes]
        new_proposal_features: Tensor[B, N, d_model] (input to next head)
    """

    SCALE_CLAMP = math.log(100000.0 / 16.0)
    BBOX_WEIGHTS = (2.0, 2.0, 1.0, 1.0)

    def __init__(self, num_classes: int = 80, d_model: int = 256, nhead: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.0,
                 num_cls_layers: int = 1, num_reg_layers: int = 3,
                 pooler_resolution: int = 7):
        super().__init__()
        self.d_model = d_model
        self.pooler_resolution = pooler_resolution

        # FPN strides (must match backbone)
        self.fpn_strides = [4, 8, 16, 32]

        # time embedding scale/shift (FiLM-style)
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 2),
        )

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # dynamic conv (instance interaction)
        self.inst_interact = DynamicConv(d_model=d_model, pooler_resolution=pooler_resolution)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU(inplace=True)

        # cls head
        cls_module = []
        for _ in range(num_cls_layers):
            cls_module += [nn.Linear(d_model, d_model, bias=False),
                           nn.LayerNorm(d_model), nn.ReLU(inplace=True)]
        self.cls_module = nn.ModuleList(cls_module)

        # reg head
        reg_module = []
        for _ in range(num_reg_layers):
            reg_module += [nn.Linear(d_model, d_model, bias=False),
                           nn.LayerNorm(d_model), nn.ReLU(inplace=True)]
        self.reg_module = nn.ModuleList(reg_module)

        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)

        # focal loss bias init
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_logits.bias, bias_value)
        nn.init.constant_(self.bboxes_delta.bias, 0.0)

    def _multi_scale_roi_align(self, features: list[torch.Tensor],
                                bboxes_xyxy: torch.Tensor) -> torch.Tensor:
        """assign boxes to FPN levels by size, then RoIAlign per level. Returns [B*N, d_model, 7, 7]."""
        B, N, _ = bboxes_xyxy.shape
        # canonical level assignment (FPN paper) — k = 4 + log2(sqrt(area)/224)
        boxes_flat = bboxes_xyxy.reshape(-1, 4)
        wh = (boxes_flat[:, 2:] - boxes_flat[:, :2]).clamp(min=1e-3)
        scale = torch.sqrt(wh[:, 0] * wh[:, 1])
        k = torch.floor(4 + torch.log2(scale / 224.0 + 1e-6))
        k = k.clamp(min=2, max=5).long() - 2  # 0,1,2,3 → p2..p5

        # batch index
        batch_idx = torch.arange(B, device=bboxes_xyxy.device).repeat_interleave(N).float()
        rois = torch.cat([batch_idx[:, None], boxes_flat], dim=1)  # [B*N, 5]

        out = torch.zeros(B * N, self.d_model, self.pooler_resolution, self.pooler_resolution,
                          device=bboxes_xyxy.device, dtype=features[0].dtype)
        for lvl, feat in enumerate(features):
            mask = (k == lvl)
            if not mask.any():
                continue
            rois_lvl = rois[mask]
            pooled = roi_align(feat, rois_lvl, output_size=self.pooler_resolution,
                                spatial_scale=1.0 / self.fpn_strides[lvl], sampling_ratio=2,
                                aligned=True)
            out[mask] = pooled
        return out  # [B*N, 256, 7, 7]

    @staticmethod
    def _apply_deltas(deltas: torch.Tensor, boxes: torch.Tensor,
                      scale_clamp: float, weights=(2.0, 2.0, 1.0, 1.0)) -> torch.Tensor:
        """deltas: [B,N,4], boxes: [B,N,4] xyxy. Returns new boxes xyxy."""
        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        dx = deltas[..., 0] / weights[0]
        dy = deltas[..., 1] / weights[1]
        dw = deltas[..., 2] / weights[2]
        dh = deltas[..., 3] / weights[3]
        dw = dw.clamp(max=scale_clamp)
        dh = dh.clamp(max=scale_clamp)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        return torch.stack([
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ], dim=-1)

    def forward(self, features: list[torch.Tensor], bboxes: torch.Tensor,
                proposal_features: torch.Tensor | None, time_emb: torch.Tensor):
        B, N, _ = bboxes.shape

        # RoIAlign per scale → [B*N, 256, 7, 7] → reshape to [B*N, 49, 256] for DynamicConv
        roi_features = self._multi_scale_roi_align(features, bboxes)
        roi_features = roi_features.flatten(2).permute(0, 2, 1)  # [B*N, 49, 256]

        # 원본 head.py:247-248 — 첫 head 는 pro 가 None → roi feature spatial mean 으로 init
        if proposal_features is None:
            proposal_features = roi_features.mean(dim=1).reshape(B, N, self.d_model)

        # proposal_features: [B, N, d_model] → [N, B, d_model] for nn.MHA
        pro = proposal_features.permute(1, 0, 2)  # [N, B, d]

        # 1. self-attn (queries between proposals)
        pro2 = self.self_attn(pro, pro, pro)[0]
        pro = pro + self.dropout1(pro2)
        pro = self.norm1(pro)

        # 2. dynamic conv — pro 와 roi 의 (batch, proposal) ordering 정합 필수
        # pro: [N, B, d] → permute → [B, N, d] → reshape → batch-major [1, B*N, d]
        # roi: _multi_scale_roi_align 결과가 batch-major [B*N, 49, d]
        # (원본 head.py:259 동치 — view+permute+reshape)
        pro_for_dyn = pro.permute(1, 0, 2).reshape(1, B * N, self.d_model)
        pro2 = self.inst_interact(pro_for_dyn, roi_features)  # [B*N, d] batch-major
        pro2 = pro2.reshape(B, N, self.d_model).permute(1, 0, 2)  # → [N, B, d]
        pro = pro + self.dropout2(pro2)
        pro = self.norm2(pro)

        # 3. FFN (원본은 FFN 후 FiLM — head.py:265-275)
        pro2 = self.linear2(self.dropout(self.activation(self.linear1(pro))))
        pro = pro + self.dropout3(pro2)
        pro = self.norm3(pro)

        # 4. FiLM-style time modulation — FFN 결과에 적용
        scale_shift = self.block_time_mlp(time_emb)  # [B, 2*d]
        scale, shift = scale_shift.chunk(2, dim=-1)  # [B, d] each
        fc = pro * (scale[None, :, :] + 1.0) + shift[None, :, :]  # [N, B, d]

        # 5. cls / reg heads — FiLM 결과를 분기 입력으로 (원본: fc_feature.clone())
        cls_feat = fc
        for layer in self.cls_module:
            cls_feat = layer(cls_feat)
        reg_feat = fc
        for layer in self.reg_module:
            reg_feat = layer(reg_feat)

        class_logits = self.class_logits(cls_feat).permute(1, 0, 2)  # [B, N, C]
        bboxes_delta = self.bboxes_delta(reg_feat).permute(1, 0, 2)  # [B, N, 4]

        # apply deltas
        new_bboxes = self._apply_deltas(bboxes_delta, bboxes, self.SCALE_CLAMP, self.BBOX_WEIGHTS)
        new_pro = pro.permute(1, 0, 2)  # [B, N, d]
        return new_bboxes, class_logits, new_pro


class Decoder(nn.Module):
    """6 iterations of DetectionHead. forward returns deep supervision outputs."""

    def __init__(self, num_classes: int = 80, num_heads: int = 6, d_model: int = 256,
                 nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        head = DetectionHead(num_classes=num_classes, d_model=d_model, nhead=nhead,
                             dim_feedforward=dim_feedforward, dropout=dropout)
        self.heads = nn.ModuleList([copy.deepcopy(head) for _ in range(num_heads)])
        self.num_heads = num_heads
        self.d_model = d_model

        # time embed
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model * 4),
        )

        # 원본 head.py:112-121 — xavier_uniform_ on all weights, focal bias 재적용
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if p.shape[-1] == num_classes:
                nn.init.constant_(p, bias_value)

    def forward(self, features: list[torch.Tensor], init_bboxes: torch.Tensor,
                t: torch.Tensor, is_eval: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """features: list[Tensor[B,256,Hi,Wi]]. init_bboxes: [B,N,4] xyxy. t: [B] long.
        Returns:
            class_logits: [B, num_heads, N, C] if not is_eval else [B, 1, N, C]
            bboxes: [B, num_heads, N, 4] if not is_eval else [B, 1, N, 4]
        """
        B, N, _ = init_bboxes.shape
        time_emb = self.time_mlp(t.float())  # [B, d*4]
        # 원본 head.py:161 — 첫 head 는 pro=None 으로 시작 (head 내부에서 roi mean 으로 init)
        pro: torch.Tensor | None = None

        bboxes = init_bboxes
        cls_list, bbox_list = [], []
        for head in self.heads:
            bboxes, class_logits, pro = head(features, bboxes, pro, time_emb)
            if not is_eval:
                cls_list.append(class_logits)
                bbox_list.append(bboxes)
            # 원본 DiffusionDet head.py:168 — head 간 gradient 차단 (iterative refinement)
            bboxes = bboxes.detach()

        if is_eval:
            return class_logits[:, None], bboxes[:, None]  # [B,1,N,C], [B,1,N,4]
        else:
            return (
                torch.stack(cls_list, dim=1),   # [B, K, N, C]
                torch.stack(bbox_list, dim=1),  # [B, K, N, 4]
            )


def build_decoder(num_classes: int = 80, cfg=None) -> Decoder:
    if cfg is None:
        return Decoder(num_classes=num_classes)
    return Decoder(
        num_classes=num_classes,
        num_heads=cfg.get("num_heads", 6),
        d_model=cfg.get("hidden_dim", 256),
        nhead=cfg.get("nhead", 8),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        dropout=cfg.get("dropout", 0.0),
    )
