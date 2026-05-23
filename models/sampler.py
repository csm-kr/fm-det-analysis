"""DDIM sampler — cosine beta schedule, 1000 timesteps, eta=0 (deterministic).

DiffusionDet 동치:
- q_sample: forward (clean → noisy)
- ddim_step: backward 1 step (noisy_t → noisy_{t-1}) given predicted x0
- sample_init_boxes: 학습 시 GT 박스 + padding → noisy 초기 박스
- sample_infer_init_boxes: 평가 시 random gaussian → boxes
- replenish_boxes: 부족한 박스 채우기 (DiffusionDet 의 box renewal)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """DiffusionDet 동치 cosine schedule."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """a[t] reshape to broadcast over x_shape. a:[T], t:[B], x_shape:[B,...]."""
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DiffusionDetSampler(nn.Module):
    """cosine schedule + DDIM eta=0 deterministic sampler."""

    def __init__(self, timesteps: int = 1000, num_inference_steps: int = 4,
                 signal_scale: float = 2.0, box_renewal: bool = True,
                 num_proposals: int = 500):
        super().__init__()
        self.timesteps = timesteps
        self.num_inference_steps = num_inference_steps
        self.signal_scale = signal_scale
        self.box_renewal = box_renewal
        self.num_proposals = num_proposals

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    # ─── forward noising ───────────────────────────────────────────────
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """x_start: clean cxcywh in [-1,1] scaled. t: [B]. noise: same shape as x_start."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_omac = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_ac * x_start + sqrt_omac * noise

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor,
                                  x0: torch.Tensor) -> torch.Tensor:
        sqrt_recip = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip * x_t - x0) / sqrt_recipm1

    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor,
                  x0_pred: torch.Tensor) -> torch.Tensor:
        """eta=0 deterministic DDIM. t_next < t (going backward)."""
        # predict noise from start
        eps = self.predict_noise_from_start(x_t, t, x0_pred)
        a_next = extract(self.alphas_cumprod, t_next, x_t.shape).clamp(min=0.0)
        x_next = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * eps
        return x_next

    # ─── init box sampling ─────────────────────────────────────────────
    def sample_init_boxes_train(self, gt_boxes_norm: torch.Tensor,
                                 num_proposals: int) -> torch.Tensor:
        """학습 시 GT(cxcywh normalized [0,1]) → padding(random) → noisy 초기 박스.
        Returns: [num_proposals, 4] in [-signal_scale, signal_scale] (스케일된 cxcywh).
        """
        n = gt_boxes_norm.shape[0]
        if n < num_proposals:
            pad = torch.rand((num_proposals - n, 4), device=gt_boxes_norm.device)
            # pad cxcy is uniform [0,1], wh is uniform [0,1] but center-biased
            pad[:, 2:] = pad[:, 2:].clamp(min=0.02)
            boxes = torch.cat([gt_boxes_norm, pad], dim=0)
        elif n > num_proposals:
            idx = torch.randperm(n, device=gt_boxes_norm.device)[:num_proposals]
            boxes = gt_boxes_norm[idx]
        else:
            boxes = gt_boxes_norm
        # scale to [-signal_scale, signal_scale]: x' = (x*2-1)*signal_scale
        boxes = (boxes * 2.0 - 1.0) * self.signal_scale
        return boxes

    def sample_infer_init_boxes(self, batch_size: int, num_proposals: int,
                                 device: torch.device) -> torch.Tensor:
        """평가 시 random gaussian → [-signal_scale, signal_scale] 영역 박스. [B, N, 4]."""
        return torch.randn(batch_size, num_proposals, 4, device=device) * self.signal_scale

    def replenish_boxes(self, boxes: torch.Tensor, num_proposals: int,
                        device: torch.device) -> torch.Tensor:
        """box_renewal: 부족하거나 신뢰도 낮은 박스를 가우시안으로 채움."""
        N = boxes.shape[0]
        if N >= num_proposals:
            return boxes[:num_proposals]
        n_new = num_proposals - N
        new = torch.randn(n_new, 4, device=device) * self.signal_scale
        return torch.cat([boxes, new], dim=0)


def build_sampler(cfg=None) -> DiffusionDetSampler:
    if cfg is None:
        return DiffusionDetSampler()
    return DiffusionDetSampler(
        timesteps=cfg.get("timesteps", 1000),
        num_inference_steps=cfg.get("num_inference_steps", 4),
        signal_scale=cfg.get("signal_scale", 2.0),
        box_renewal=cfg.get("box_renewal", True),
    )
