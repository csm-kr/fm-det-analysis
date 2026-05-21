import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou, box_convert

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  #

# from utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from fvcore.nn import sigmoid_focal_loss_jit
from src.loss.matcher import Matcher

class DiffusionDetLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Loss Calculators
        self.cls_criterion = CLS_Loss(num_classes=80)  # 예시로 80 클래스
        self.box_criterion = BOX_Loss()
        self.matcher = Matcher()

    def forward(self, pred_logits, pred_boxes, targets):
        """
        Args:
            pred_logits (Tensor): [Batch, Num_Layers, Num_Proposals, Num_Classes]
            pred_boxes (Tensor):  [Batch, Num_Layers, Num_Proposals, 4]
            targets (List[Dict]): GT targets
            matched_info (List):  [(fg_mask, gt_indices), ...] per batch (외부 Matcher 결과)
            num_boxes (float):    Normalization factor
        """
        
        # Matcher
        num_layers = pred_logits.shape[1] # 6개 레이어 

        losses = {}

        for l in range(num_layers):

            current_logits = pred_logits[:, l]
            current_boxes = pred_boxes[:, l]
            matched_info = self.matcher((current_logits, current_boxes), targets)
            
            l_cls = self.cls_criterion(current_logits, targets, matched_info)
            l_box = self.box_criterion(current_boxes, targets, matched_info)

            # 마지막 layer 는 suffix 가 없음
            if l == num_layers - 1:
                suffix = ""
            else:
                suffix = f"_{l}"

            losses.update({k + suffix: v for k, v in l_cls.items()})
            losses.update({k + suffix: v for k, v in l_box.items()})

        return losses


# =============================================================================
# 하위 Loss Module (입력이 (B, N, C) 형태로 들어오므로 기존 로직 유지 가능)
# =============================================================================

class CLS_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Config에서 Focal Loss 파라미터 로드
        self.focal_loss_alpha = 0.25 
        self.focal_loss_gamma = 2.0  

    def forward(self, pred_logits, targets, matched_info):
        """
        pred_logits: [Batch, Num_Proposals, Num_Classes] (Single Layer)
        """
        batch_size = len(targets)
        
        # 1. Target Class 텐서 준비 (기본값: Background)
        # src_logits -> pred_logits 로 변경
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)

        # 2. 매칭된 위치에 GT Label 할당
        target_classes_o_list = []
        for batch_idx in range(batch_size):
            target_mask, gt_indices = matched_info[batch_idx]
            if len(gt_indices) == 0:
                continue
            
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, target_mask] = target_classes_o[gt_indices]
            target_classes_o_list.append(target_classes_o[gt_indices])
    
        num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1              # 할당된 query 의 갯수

        # 3. One-hot Encoding
        target_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], self.num_classes + 1], dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        
        # Background 클래스(마지막 인덱스) 제거 -> [B, N, num_classes]
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        # 4. Focal Loss 계산
        pred_logits_flat = pred_logits.flatten(0, 1)                             # batch x num_queries, num_classes
        target_classes_onehot_flat = target_classes_onehot.flatten(0, 1)         # batch x num_queries, num_classes

        # (가정: torchvision의 sigmoid_focal_loss 혹은 구현체 사용)
        cls_loss = sigmoid_focal_loss_jit(
            pred_logits_flat, 
            target_classes_onehot_flat, 
            alpha=self.focal_loss_alpha, 
            gamma=self.focal_loss_gamma, 
            reduction="none"
        )
        
        loss_ce = torch.sum(cls_loss) / num_boxes
        loss_ce *= 2.0
        return {'loss_ce': loss_ce}


class BOX_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, targets, matched_info):
        """
        pred_boxes: [Batch, Num_Proposals, 4] (Single Layer, XYXY ABS)
        """

        batch_size = len(targets)
        pred_boxes_xyxy_abs_list = []
        pred_boxes_xyxy_norm_list = []

        gt_boxes_xyxy_abs_list = []
        gt_boxes_xyxy_norm_list = []

        # 배치 돌면서 
        for batch_idx in range(batch_size):
            target_mask, gt_indices = matched_info[batch_idx][0], matched_info[batch_idx][1]
            if len(gt_indices) == 0:
                continue

            image_size = targets[batch_idx]["image_size_whwh"]

            # GT boxes - XYXY ABS
            gt_boxes_xyxy_abs = targets[batch_idx]["boxes"][gt_indices] 
            gt_boxes_xyxy_abs_list.append(gt_boxes_xyxy_abs)

            gt_boxes_xyxy_norm = gt_boxes_xyxy_abs / image_size
            gt_boxes_xyxy_norm_list.append(gt_boxes_xyxy_norm)

            # Prediction - XYXY ABS
            pred_boxes_xyxy_abs = pred_boxes[batch_idx][target_mask]
            pred_boxes_xyxy_abs_list.append(pred_boxes_xyxy_abs)

            pred_boxes_xyxy_norm = pred_boxes_xyxy_abs / image_size
            pred_boxes_xyxy_norm_list.append(pred_boxes_xyxy_norm)

        losses = {}

        # 2. Loss 계산
        if len(pred_boxes_xyxy_abs_list) > 0:

            # for boxes loss
            pred_boxes_xyxy_norm_cat = torch.cat(pred_boxes_xyxy_norm_list)
            gt_boxes_xyxy_norm_cat = torch.cat(gt_boxes_xyxy_norm_list)

            # for giou loss
            pred_boxes_xyxy_abs_cat = torch.cat(pred_boxes_xyxy_abs_list)  
            gt_boxes_xyxy_abs_cat = torch.cat(gt_boxes_xyxy_abs_list)
            
            # num_gt_boxes
            num_boxes = gt_boxes_xyxy_abs_cat.size(0)                     

            loss_bbox = F.l1_loss(pred_boxes_xyxy_norm_cat, gt_boxes_xyxy_norm_cat, reduction='none')
            loss_bbox *= 5.0
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes   

            # GIoU Loss
            loss_giou = 1 - torch.diag(generalized_box_iou(pred_boxes_xyxy_abs_cat, gt_boxes_xyxy_abs_cat))
            loss_giou *= 2.0
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses['loss_bbox'] = pred_boxes.sum() * 0
            losses['loss_giou'] = pred_boxes.sum() * 0

        return losses

