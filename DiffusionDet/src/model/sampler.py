
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import math
import torch
import random
import torch.nn as nn
from utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class DiffusionDetSampler(nn.Module):
    def __init__(self, timesteps=1000, s=0.008, num_proposals=500, scale=1.0):
        super().__init__()

        # cosine noise schedule (fixed during training)
        betas = self.cosine_beta_schedule(timesteps, s)  # [T]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)    # [T]

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        self.timesteps = timesteps
        self.num_proposals = num_proposals
        self.scale = scale

    # --------------------------------------------------------
    # (A) cosine schedule
    # --------------------------------------------------------
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        Cosine schedule from: https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)

        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999).float()

    # --------------------------------------------------------
    # (B) forward noising (clean -> noisy)
    # --------------------------------------------------------
    def q_sample(self, x0, t, noise=None):
        """
        x0 : clean data (500, 4)
        t  : timestep indices (B,)
        noise : noise, (500, 4)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]                # (N,)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]      # (N,)

        # K = 1,N = 500, C = 4
        term1 = torch.einsum("k, n c -> n c", sqrt_alpha_bar, x0)
        term2 = torch.einsum("k, n c -> n c", sqrt_one_minus, noise)

        return term1 + term2

    def sample_init_boxes(self, targets):
        '''
        target 에서 boxes (xyxy_abs) 를 추출하여 t 에 관한 초기 박스를 샘플링합니다.
        '''

        ret_init_boxes, ts = [], []
        
        for target in targets:

            image_size_whwh = target['image_size_whwh']                                                              # whwh abs
            gt_boxes_cxcywh_norm = box_xyxy_to_cxcywh(target['boxes'] / image_size_whwh)                             # cxcywh norm
       
            # device
            device = gt_boxes_cxcywh_norm.device if gt_boxes_cxcywh_norm.numel() > 0 else self.betas.device

            # sample timestep & noise
            t = torch.randint(0, self.timesteps, (1,), device=device).long()                                         # sample t
            noise = torch.randn(self.num_proposals, 4, device=device)                                                # prepare epsilon

            x_start = self._pad_or_trim_boxes(gt_boxes_cxcywh_norm, device)                                          # x0 ~ N(0.5, 0.5)
            x_start = (x_start * 2.0 - 1.0) * self.scale                                                             # 0 - 1 to -1 - 1

            # forward noising for boxes (t is scalar here)
            x = self.q_sample(x_start, t, noise=noise)
            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.                                                                         # -1 ~ 1 to 0 ~ 1

            init_boxes_xyxy_norm = box_cxcywh_to_xyxy(x)                                                            # xyxy norm
            init_boxes_xyxy_abs = init_boxes_xyxy_norm * image_size_whwh                                            # xyxy abs
            ret_init_boxes.append(init_boxes_xyxy_abs)
            ts.append(t)

        ret_init_boxes = torch.stack(ret_init_boxes)
        ts = torch.stack(ts).squeeze(-1)

        return ret_init_boxes, ts

    def _pad_or_trim_boxes(self, boxes, device):
        """
        Adjust boxes to fixed num_proposals by padding with random boxes or trimming.
        boxes: (N, 4) normalized xyxy in [0,1] (may be empty).
        """
        num_gt = boxes.shape[0]

        if num_gt == 0:
            boxes = torch.as_tensor([[0.5, 0.5, 1.0, 1.0]], dtype=torch.float32, device=device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4, device=device) / 6.0 + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_idx = torch.randperm(num_gt, device=device)[: self.num_proposals]
            x_start = boxes[select_idx]
        else:
            x_start = boxes

        return x_start
    
    def sample_infer_init_boxes(self, shape, image_size_whwh):

        init_boxes_cxcywh_norm = torch.randn(shape, device=image_size_whwh.device)
        x = init_boxes_cxcywh_norm
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=1 * self.scale) 
        x_boxes = ((x_boxes / self.scale) + 1) / 2                         # (-1 * scale , 1 * scale) to (0, 1)
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * image_size_whwh
        return x_boxes


    # --------------------------------------------------------
    # Helper: Extract & Broadcast
    # --------------------------------------------------------
    def _extract(self, a, t, x_shape):
        """
        [T] 크기의 텐서 a에서 t 인덱스 값을 뽑아
        x_shape에 맞춰 [B, 1, 1] 형태로 만들어줍니다.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # --------------------------------------------------------
    # Inversion: Predict Noise (기존 버퍼 활용)
    # --------------------------------------------------------
    def predict_noise_from_start(self, x_t, t, x0):
        """
        x_t와 예측된 x0를 이용해 noise(epsilon)를 역산합니다.
        Formula: noise = (x_t - sqrt_alpha * x0) / sqrt(1 - alpha)
        """
        # 1. 기존 버퍼에서 값 가져오기
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # 2. 수식 적용
        return (x_t - sqrt_alpha * x0) / sqrt_one_minus
    
    # DiffusionDetSampler 클래스 내부에 추가
    def ddim_step(self, img, b_start, time, time_next, eta=0.0):
        """
        eta :
        1.0 - ddpm 
        0.0 - ddim 
        실험
        """

        # 
        t_tensor = torch.full((img.shape[0],), time, device=img.device, dtype=torch.long)
        current_pred_noise = self.predict_noise_from_start(img, t_tensor, b_start)

        # time, time_next 가 정수 
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next] if time_next >= 0 else torch.tensor(1.0, device=img.device)

        # generalization form 
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).clamp(min=0).sqrt()

        noise = torch.randn_like(img)
        img_next = b_start * alpha_next.sqrt() + c * current_pred_noise + sigma * noise
        
        return img_next

    def replenish_boxes(self, img, num_proposals):
        """부족한 박스를 채워주는 기능"""
        num_remain = img.shape[1]
        num_to_add = num_proposals - num_remain
        if num_to_add > 0:
            new_boxes = torch.randn(img.shape[0], num_to_add, 4, device=img.device)
            img = torch.cat((img, new_boxes), dim=1)
        return img


if __name__ == "__main__":

    from model.diffusiondet import DiffusionDet

    from torch.utils.data import DataLoader
    from dataset.coco_dataset import COCO_Dataset, build_train_augmentation
    transform_train = build_train_augmentation()

    dataset = COCO_Dataset(
        # data_root="/usr/src/data/coco",
        data_root=r"D:\data\coco",
        split="val",
        transform=transform_train,
    )

    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,               # 원하는 배치 사이즈
        shuffle=True,               # train 은 shuffle 권장
        num_workers=0,              # CPU 코어 수에 맞추기
        pin_memory=False,           # CUDA speed-up
        collate_fn=dataset.collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiffusionDet().to(device)

    batch = next(iter(dataloader))
    result = model(batch)
    print(result)





