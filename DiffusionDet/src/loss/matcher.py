import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from fvcore.nn import sigmoid_focal_loss_jit
# from .util import box_ops
# from .util.misc import get_world_size, is_dist_avail_and_initialized
# from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from utils.bbox_utils import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


# =============================================================================
# 1. Matcher (SimOTA / Dynamic K) - Updated for preprocess_data
# =============================================================================
class Matcher(nn.Module):
    def __init__(self, lambda_class: float = 2.0, lambda_box: float = 5.0, lambda_giou: float = 2.0):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lambda_class = lambda_class
        self.lambda_box = lambda_box
        self.lambda_giou = lambda_giou

        self.ota_k = 5
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tuple[Tensor, Tensor]): 
                final layer's outputs
                - pred_logits: [B, N, C] (Heads, Batch, Queries, Classes)
                - pred_boxes:  [B, N, 4] (Heads, Batch, Queries, XYXY)
            
            targets (List[Dict]): List containing GT info per image.
                - keys: 'boxes' (XYXY ABS), 'labels', 'image_size' (h, w)
        """
        batched_pred_logits, batched_pred_boxes = outputs
        batched_pred_logits = batched_pred_logits.sigmoid() 
        bs, num_queries = batched_pred_logits.shape[:2]      
    
        # fg_masks = []
        # matched_indices = []
        matched_info = []

        for batch_idx, (pred_logits, pred_boxes) in enumerate(zip(batched_pred_logits, batched_pred_boxes)):  

            # gt labels 및 gt_boxes
            gt_labels = targets[batch_idx]["labels"]
            gt_boxes_xyxy_abs = targets[batch_idx]["boxes"]

            # pred logits 및 pred boxes
            pred_logits = pred_logits
            pred_boxes_xyxy_abs = pred_boxes

            num_gts = len(gt_labels)
            if num_gts == 0:
                fg_mask = torch.zeros(num_queries, dtype=torch.bool, device=self.device)
                target_indices = torch.empty(0, dtype=torch.int64, device=self.device)
                matched_info.append((fg_mask, target_indices))
                continue

            # pred logits 및 pred boxes
            foreground_mask, is_in_boxes_and_center = self.get_in_boxes_info(box_xyxy_to_cxcywh(pred_boxes_xyxy_abs), box_xyxy_to_cxcywh(gt_boxes_xyxy_abs))
            pair_wise_ious = ops.box_iou(pred_boxes_xyxy_abs, gt_boxes_xyxy_abs)

            # Class Cost
            alpha = self.focal_loss_alpha
            gamma = self.focal_loss_gamma
            neg_cost_class = (1 - alpha) * (pred_logits ** gamma) * (-(1 - pred_logits + 1e-8).log())
            pos_cost_class = alpha * ((1 - pred_logits) ** gamma) * (-(pred_logits + 1e-8).log())
            cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]

            # Box Cost
            gt_image_size_out = targets[batch_idx]['image_size_whwh']                   # gt N, 4
            out_bbox = pred_boxes_xyxy_abs / gt_image_size_out                          # [N, 4] XYXY / whwh
            tgt_bbox = gt_boxes_xyxy_abs / gt_image_size_out                            # [N, 4]
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # GIoU Cost
            cost_giou = -generalized_box_iou(pred_boxes_xyxy_abs, gt_boxes_xyxy_abs)

            # Final cost matrix
            cost = self.lambda_box * cost_bbox + self.lambda_class * cost_class + self.lambda_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
            cost[~foreground_mask] = cost[~foreground_mask] + 10000.0                                                                                   # 500, N_gt

            fg_mask, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, gt_boxes_xyxy_abs.shape[0])
            matched_info.append((fg_mask, matched_qidx))

        return matched_info

    def get_in_boxes_info(self, boxes, target_gts):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)                                  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        # 앵커 미할당된 gt 에 대한 조치 
        anchor_matching_gt = matching_matrix.sum(1)
        # 1. 혹시 하나도 못 받은 '거지 GT'가 있니?
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return selected_query, gt_indices # , matched_query_id  # 마스크, gt 인덱스, 쿼리 인덱스 
