"""DiffusionDet 통합 nn.Module — backbone + sampler + decoder.

학습: GT → noisy boxes → 6-head forward → deep supervision loss
평가: random gaussian → DDIM N step → 최종 박스
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone import build_backbone
from models.decoder import build_decoder
from models.sampler import build_sampler
from utils.box_ops import (
    box_cxcywh_to_xyxy,
    box_xyxy_denormalize,
    box_xyxy_normalize,
    box_xyxy_to_cxcywh,
)


class DiffusionDet(nn.Module):
    def __init__(self, num_classes: int = 80, num_proposals: int = 500,
                 backbone_cfg=None, decoder_cfg=None, sampler_cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.backbone = build_backbone(backbone_cfg)
        self.decoder = build_decoder(num_classes=num_classes, cfg=decoder_cfg)
        self.sampler = build_sampler(sampler_cfg)

    # ─── helpers ───────────────────────────────────────────────────────
    def _prepare_noisy_train_boxes(self, targets: list[dict], device: torch.device,
                                    image_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """학습 시 GT → noisy boxes [B, N, 4] xyxy (image coords). returns (boxes_xyxy, t).
        """
        B = len(targets)
        T = self.sampler.timesteps
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

        boxes_per_batch = []
        for b, tgt in enumerate(targets):
            H, W = image_sizes[b].tolist()
            gt_xyxy = tgt["boxes"].to(device).float()
            if gt_xyxy.numel() == 0:
                # all random
                gt_norm_cxcywh = torch.empty(0, 4, device=device)
            else:
                gt_norm_xyxy = box_xyxy_normalize(gt_xyxy,
                                                   torch.tensor([H, W], device=device).float())
                gt_norm_cxcywh = box_xyxy_to_cxcywh(gt_norm_xyxy)
            # → scaled cxcywh in [-signal_scale, signal_scale]
            x_start = self.sampler.sample_init_boxes_train(gt_norm_cxcywh, self.num_proposals)
            # add noise at timestep t[b]
            noise = torch.randn_like(x_start)
            x_t = self.sampler.q_sample(x_start, t[b:b+1].expand(self.num_proposals), noise)
            # clip back to [-signal_scale, signal_scale]
            x_t = x_t.clamp(-self.sampler.signal_scale, self.sampler.signal_scale)
            # un-scale to [0,1] then to xyxy in image coords
            cxcywh01 = (x_t / self.sampler.signal_scale + 1.0) * 0.5
            cxcywh01 = cxcywh01.clamp(0.0, 1.0)
            xyxy01 = box_cxcywh_to_xyxy(cxcywh01)
            xyxy01[:, 2] = torch.maximum(xyxy01[:, 2], xyxy01[:, 0] + 1e-4)
            xyxy01[:, 3] = torch.maximum(xyxy01[:, 3], xyxy01[:, 1] + 1e-4)
            xyxy_img = box_xyxy_denormalize(xyxy01,
                                              torch.tensor([H, W], device=device).float())
            boxes_per_batch.append(xyxy_img)
        return torch.stack(boxes_per_batch, dim=0), t

    # ─── forward ───────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor, targets: list[dict] | None = None):
        """images: [B, 3, H, W]. targets: list of dicts (학습 시) or None (평가).
        Returns:
            학습: pred_logits [B, K, N, C], pred_boxes [B, K, N, 4]
            평가: pred_logits [B, N, C], pred_boxes [B, N, 4]
        """
        B = images.shape[0]
        device = images.device
        H, W = images.shape[2:]
        image_sizes = torch.tensor([[H, W]] * B, device=device).float()

        features = self.backbone(images)

        if self.training:
            assert targets is not None, "targets required in training"
            init_bboxes, t = self._prepare_noisy_train_boxes(targets, device, image_sizes)
            pred_logits, pred_boxes = self.decoder(features, init_bboxes, t, is_eval=False)
            return {
                "pred_logits": pred_logits,   # [B, K, N, C]
                "pred_boxes": pred_boxes,     # [B, K, N, 4] xyxy image
                "image_sizes": image_sizes,
            }
        else:
            # DDIM N step
            init_bboxes_cxcywh = self.sampler.sample_infer_init_boxes(B, self.num_proposals, device)
            cxcywh01 = (init_bboxes_cxcywh / self.sampler.signal_scale + 1.0) * 0.5
            cxcywh01 = cxcywh01.clamp(0.0, 1.0)
            xyxy01 = box_cxcywh_to_xyxy(cxcywh01)
            xyxy_img = xyxy01 * torch.tensor([W, H, W, H], device=device).float()

            n_steps = self.sampler.num_inference_steps
            T = self.sampler.timesteps
            ts = torch.linspace(-1, T - 1, n_steps + 1).long().to(device)
            ts = ts.flip(0)  # T → 0

            x_t = init_bboxes_cxcywh
            bboxes = xyxy_img
            for i in range(n_steps):
                t_cur = ts[i].repeat(B).clamp(min=0)
                t_next = ts[i + 1].repeat(B).clamp(min=0)
                pred_logits, pred_xyxy_img = self.decoder(features, bboxes,
                                                            t_cur, is_eval=True)
                pred_logits = pred_logits[:, 0]  # [B, N, C]
                pred_xyxy_img = pred_xyxy_img[:, 0]  # [B, N, 4]

                # back-project to scaled cxcywh for DDIM step
                pred_xyxy01 = pred_xyxy_img / torch.tensor([W, H, W, H], device=device).float()
                pred_xyxy01 = pred_xyxy01.clamp(0.0, 1.0)
                pred_cxcywh01 = box_xyxy_to_cxcywh(pred_xyxy01)
                pred_x0 = (pred_cxcywh01 * 2.0 - 1.0) * self.sampler.signal_scale
                pred_x0 = pred_x0.clamp(-self.sampler.signal_scale, self.sampler.signal_scale)

                if i < n_steps - 1:
                    x_t = self.sampler.ddim_step(x_t, t_cur, t_next, pred_x0)
                    cxcywh01 = (x_t / self.sampler.signal_scale + 1.0) * 0.5
                    cxcywh01 = cxcywh01.clamp(0.0, 1.0)
                    xyxy01 = box_cxcywh_to_xyxy(cxcywh01)
                    bboxes = xyxy01 * torch.tensor([W, H, W, H], device=device).float()

            return {
                "pred_logits": pred_logits,   # [B, N, C]
                "pred_boxes": pred_xyxy_img,  # [B, N, 4]
                "image_sizes": image_sizes,
            }


def build_diffusiondet(cfg) -> DiffusionDet:
    return DiffusionDet(
        num_classes=cfg.num_classes,
        num_proposals=cfg.num_proposals,
        backbone_cfg=cfg.get("backbone"),
        decoder_cfg=cfg,
        sampler_cfg=cfg.get("sampler"),
    )
