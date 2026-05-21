# path 맞추기 위해서 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn

from src.model.encoder import ResNet50FPN
from src.model.decoder import Decoder
from detectron2.layers import batched_nms
from detectron2.structures import ImageList, Boxes, Instances
from src.model.sampler import DiffusionDetSampler

from src.loss.matcher import Matcher

from src.utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class DiffusionDet(nn.Module):
    def __init__(self, scale=1.0, box_renewal=True):
        super().__init__()
        
        # encoder
        self.encoder = ResNet50FPN()

        # decoder
        self.decoder = Decoder()

        # sampler       
        self.sampler = DiffusionDetSampler(scale=scale)
        
        # options
        self.num_timesteps = 1000
        self.num_proposals = 500
        self.size_divisibility = 32

        # matcher
        self.matcher = Matcher()

        self.num_classes = 80
        self.scale = scale
        self.box_renewal = box_renewal
    
    def forward(self, images, targets):
        """
        images : Detectron2.ImageList - images.tensor [B, 3, H, W]
        """
        
        self.device = images.device

        # extract image features
        features_list = self.encoder(images.tensor)

        if self.training:
            init_boxes_gt_xyxy_abs, ts = self.sampler.sample_init_boxes(targets)
            pred_logits, pred_boxes = self.decoder(features_list, init_boxes_gt_xyxy_abs, ts)

        # inference
        else:
            # Inference 시에는 초기 박스를 learnable parameter로 사용
            batch_size = images.tensor.shape[0]
            shape = (batch_size, self.num_proposals, 4)
            
            image_size_tensor = targets[0]['image_size_whwh']
            h = int(image_size_tensor[1].item())
            w = int(image_size_tensor[0].item())
            image_size_tuple = [(h, w)]

            # get b_t
            init_boxes_infer_xyxy_abs = self.sampler.sample_infer_init_boxes(shape, image_size_tensor)

            ########################### Box Renewal ############################
            if self.box_renewal:

                # -------------------------- for ensemble --------------------------
                ensemble_score, ensemble_label, ensemble_coord = [], [], []
                # -------------------------- set time steps --------------------------
                time_steps = torch.linspace(-1, self.num_timesteps - 1, steps=4 + 1)
                reverse_time_steps = list(reversed(time_steps.int().tolist()))
                reverse_time_step_tuple = list(zip(reverse_time_steps[:-1], reverse_time_steps[1:]))

                img = self.box_absolute_to_signal(init_boxes_infer_xyxy_abs, image_size_tuple) 

                for i, (time, time_next) in enumerate(reverse_time_step_tuple):
                    init_boxes_infer_xyxy_abs = self.box_signal_to_absolute(img, image_size_tuple)                
                    t_tensor = torch.full((batch_size,), time, device=self.device, dtype=torch.long)

                    # Decoder inference
                    pred_logits, pred_boxes = self.decoder(features_list, init_boxes_infer_xyxy_abs, t_tensor, is_eval=True)

                    # Latent space로 변환 (b_start)
                    b_start = self.box_absolute_to_signal(pred_boxes, image_size_tuple) 
                    
                    # --- Box Renewal (Filtering) ---
                    threshold = 0.5
                    # pred_logits: [B, num_proposals, num_classes] -> 여기서는 batch=1 가정 (pred_logits[0])
                    
                    scores_sigmoid = torch.sigmoid(pred_logits[0])              # [500, 80]
                    value, _ = torch.max(scores_sigmoid, -1, keepdim=False)     # [500]
                    keep_idx = value > threshold
                    # num_remain = torch.sum(keep_idx)
                    
                    # 현재 단계의 noise 예측
                    current_pred_noise = self.sampler.predict_noise_from_start(img, t_tensor, b_start)

                    # 필터링 
                    img = img[:, keep_idx, :]
                    b_start = b_start[:, keep_idx, :]
                    current_pred_noise = current_pred_noise[:, keep_idx, :]

                    if time_next < 0:
                        img = b_start
                    else:
                        img = self.sampler.ddim_step(img, b_start, time, time_next)

                    img = self.sampler.replenish_boxes(img, self.num_proposals)

                    # --- Ensemble Data Collection ---

                    results = self.inference(pred_logits, pred_boxes, image_size_tuple)
                    if len(results) > 0:
                        # 1. 리스트에서 첫 번째 Instances 객체 추출
                        inst = results[0]

                        h, w = inst.image_size

                        # 2. 인덱스로 접근하기
                        h = inst.image_size[0] # height
                        w = inst.image_size[1] # width
                        
                        # 2. 각 필드 데이터를 텐서 형태로 추출하여 저장
                        # pred_boxes는 Boxes 객체이므로 .tensor를 붙여줘야 cat이 가능합니다.
                        ensemble_coord.append(inst.pred_boxes.tensor) 
                        ensemble_score.append(inst.scores)
                        ensemble_label.append(inst.pred_classes)

                    # --- 최종 결과 산출 (반복문 외부) ---
                if len(ensemble_score) > 0:
                    # 모든 타임스텝의 결과 합치기
                    all_boxes = torch.cat(ensemble_coord, dim=0)
                    all_scores = torch.cat(ensemble_score, dim=0)
                    all_labels = torch.cat(ensemble_label, dim=0)

                    from torchvision.ops import batched_nms
                    # NMS 실행
                    keep = batched_nms(all_boxes, all_scores, all_labels, 0.5) 
                    
                    # 최종 Instances 객체 생성 (사용자님의 image_size_tuple 활용)
                    result = Instances((h, w))
                    result.pred_boxes = Boxes(all_boxes[keep])
                    result.scores = all_scores[keep]
                    result.pred_classes = all_labels[keep]
                    
                    final_results = [result] # 최종 출력 형식
                else:
                    return []
                return final_results
            # ############################ W Box Renewal ############################

            ############################ WO Box Renewal ############################
            else:
                # ------------------------------------------------- wo ddpm -------------------------------------------------
                t = torch.full((batch_size,), 999, device=images.device, dtype=torch.long)
                pred_logits, pred_boxes = self.decoder(features_list, init_boxes_infer_xyxy_abs, t, is_eval=True)
                results = self.inference(pred_logits, pred_boxes, image_size_tuple)
                return results

        return pred_logits, pred_boxes    


    def _get_whwh_tensor(self, image_size_tuple, device, dtype):
        """
        [(h, w), ...] 리스트를 [B, 1, 4] 크기의 (w, h, w, h) 텐서로 변환
        """
        whwh_list = []
        for h, w in image_size_tuple:
            whwh_list.append([w, h, w, h]) # (H, W) -> (W, H, W, H)
            
        # [B, 4] -> [B, 1, 4] (Broadcasting용)
        return torch.tensor(whwh_list, device=device, dtype=dtype).unsqueeze(1)

    def box_absolute_to_signal(self, boxes_abs, image_size_tuple):
        """
        절대 좌표 박스를 Diffusion Signal로 변환
        Args:
            boxes_abs: [B, N, 4] (x1, y1, x2, y2)
            image_size_tuple: list of (h, w) tuples
        """
        # 1. (w, h, w, h) 텐서 생성
        images_whwh = self._get_whwh_tensor(image_size_tuple, boxes_abs.device, boxes_abs.dtype)

        # 2. Normalize: [0, W] -> [0, 1]
        x_norm = boxes_abs / images_whwh
        
        # 3. Convert: xyxy -> cxcywh
        x_cxcywh = box_xyxy_to_cxcywh(x_norm)
        
        # 4. Scale & Shift: [0, 1] -> [-scale, scale]
        x_signal = (x_cxcywh * 2 - 1.) * self.scale
        x_signal = torch.clamp(x_signal, min=-1 * self.scale, max=self.scale)
        
        return x_signal

    def box_signal_to_absolute(self, boxes_signal, image_size_tuple):
        """
        Diffusion Signal을 절대 좌표 박스로 복원
        Args:
            boxes_signal: [B, N, 4] (cx, cy, w, h)
            image_size_tuple: list of (h, w) tuples
        """
        # 1. (w, h, w, h) 텐서 생성
        images_whwh = self._get_whwh_tensor(image_size_tuple, boxes_signal.device, boxes_signal.dtype)

        # 2. Unscale: [-scale, scale] -> [0, 1]
        x_unscaled = boxes_signal / self.scale
        x_norm = (x_unscaled + 1) / 2.
        
        # 3. Convert: cxcywh -> xyxy
        boxes_xyxy = box_cxcywh_to_xyxy(x_norm)
        
        # 4. Denormalize: [0, 1] -> [0, W]
        boxes_abs = boxes_xyxy * images_whwh
        
        return boxes_abs


    def inference(self, box_cls, box_pred, image_sizes):
        """
        Refactored Inference Logic
        - Removed 'repeat' operations for better memory efficiency.
        - Simplified Top-K selection using divmod.
        """

        assert len(box_cls) == len(image_sizes)  # 배치확인 
        results = []

        # if self.use_focal or self.use_fed_loss:
        scores = torch.sigmoid(box_cls)
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(scores, box_pred, image_sizes)):
            result = Instances(image_size)
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]

            keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
            box_pred_per_image = box_pred_per_image[keep]
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]

            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results
    

if __name__ == "__main__":
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
        batch_size=2,                   # 원하는 배치 사이즈
        shuffle=True,                   # train 은 shuffle 권장
        num_workers=0,                  # CPU 코어 수에 맞추기
        pin_memory=False,               # CUDA speed-up
        collate_fn=dataset.collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionDet().to(device)

    print("training : ", model.training)

    for i, batch in enumerate(dataloader):
        # 모델 forward
        outputs = model(batch)

        print("pred_logits:", outputs[0].shape)
        print("pred_boxes:", outputs[1].shape)

        print("batch", batch)

        if i > 10:
            break
