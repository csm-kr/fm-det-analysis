"""VOC2007 11-point interpolation mAP@0.5.

산출: dict {metric_primary (=mAP50), mAP50, per_class_ap, num_predictions}.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import box_iou


def _voc_ap_11pt(rec: np.ndarray, prec: np.ndarray) -> float:
    """VOC2007 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = rec >= t
        p = prec[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


@torch.no_grad()
def voc_eval(model, loader, device, num_classes: int = 20,
             iou_thresh: float = 0.5, score_thresh: float = 0.05,
             max_dets_per_image: int = 100) -> dict:
    """VOC2007 mAP@0.5.

    GT 의 difficult flag 는 VOCDetection 의 drop_difficult 처리에 의존 (DATA_CARD 참고).
    """
    model.eval()
    # image_id -> (labels [n], boxes [n,4] xyxy in original image px)
    gts: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    # list[(image_id, cls, score, box [4])]
    preds: list[tuple[int, int, float, torch.Tensor]] = []

    for batch in loader:
        images, targets = batch
        images = images.to(device, non_blocking=True)
        out = model(images)
        scores = out["pred_logits"].sigmoid()
        boxes = out["pred_boxes"]
        B, N, C = scores.shape

        for b in range(B):
            tgt = targets[b]
            image_id = int(tgt["image_id"])
            orig_h, orig_w = tgt["orig_size"]
            cur_h, cur_w = tgt["size"]
            sx = orig_w / max(cur_w, 1)
            sy = orig_h / max(cur_h, 1)

            gt_boxes = tgt["boxes"].clone().cpu().float()
            gt_labels = tgt["labels"].clone().cpu().long()
            gts[image_id] = (gt_labels, gt_boxes)

            scs_b = scores[b].reshape(-1)
            k = min(max_dets_per_image, scs_b.numel())
            top_scores, top_idx = scs_b.topk(k)
            box_idx = top_idx // C
            cls_idx = top_idx % C
            boxes_b = boxes[b][box_idx].clone()
            boxes_b[:, 0::2] = boxes_b[:, 0::2] * sx
            boxes_b[:, 1::2] = boxes_b[:, 1::2] * sy
            for j in range(k):
                s = float(top_scores[j])
                if s < score_thresh:
                    continue
                preds.append((image_id, int(cls_idx[j]), s, boxes_b[j].cpu()))

    per_class_ap: list[float] = []
    for c in range(num_classes):
        gt_count = 0
        gt_match_pool: dict[int, np.ndarray] = {}
        for image_id, (labels, gt_boxes) in gts.items():
            mask = (labels == c)
            cnt = int(mask.sum().item())
            if cnt > 0:
                gt_count += cnt
                gt_match_pool[image_id] = np.zeros(cnt, dtype=bool)
        if gt_count == 0:
            per_class_ap.append(0.0)
            continue

        cls_preds = sorted([p for p in preds if p[1] == c], key=lambda x: -x[2])
        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)
        for i, (image_id, _, _, box) in enumerate(cls_preds):
            if image_id not in gt_match_pool:
                fp[i] = 1.0
                continue
            labels, gt_boxes = gts[image_id]
            cls_mask = (labels == c)
            cls_gt = gt_boxes[cls_mask]
            if cls_gt.numel() == 0:
                fp[i] = 1.0
                continue
            ious = box_iou(box.unsqueeze(0), cls_gt).squeeze(0)
            best_iou, best_idx = ious.max(0)
            if float(best_iou) >= iou_thresh and not gt_match_pool[image_id][int(best_idx)]:
                tp[i] = 1.0
                gt_match_pool[image_id][int(best_idx)] = True
            else:
                fp[i] = 1.0

        if len(cls_preds) == 0:
            per_class_ap.append(0.0)
            continue
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        rec = tp_c / max(gt_count, 1)
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-10)
        per_class_ap.append(_voc_ap_11pt(rec, prec))

    mAP50 = float(np.mean(per_class_ap)) if per_class_ap else 0.0
    return {
        "metric_primary": mAP50,
        "mAP50": mAP50,
        "per_class_ap": [float(x) for x in per_class_ap],
        "num_predictions": int(len(preds)),
        "num_classes": int(num_classes),
        "iou_thresh": float(iou_thresh),
        "score_thresh": float(score_thresh),
    }
