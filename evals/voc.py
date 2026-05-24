"""VOC2007 11-point interpolation mAP@0.5.

산출: dict {metric_primary (=mAP50), mAP50, per_class_ap, num_predictions}.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.ops import batched_nms, box_iou


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
             nms_thresh: float = 0.5, max_dets_per_image: int = 100) -> dict:
    """VOC2007 mAP@0.5.

    inference 후 per-class NMS (torchvision.ops.batched_nms, iou_thresh=nms_thresh) 적용
    → 한 객체에 중복된 head 출력 박스 제거 (원본 DiffusionDet 동치).

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

            # GT 와 prediction 모두 *transformed (cur)* 좌표계에서 비교.
            # transforms.py 의 RandomResize 가 tgt["boxes"] 를 cur 좌표로 scale 하고,
            # model 도 cur 좌표로 박스를 뱉기 때문에 두 쪽 모두 cur 에서 일치시킴
            # (이전 버전: pred 만 sx, sy 로 orig 으로 scale → GT 와 좌표계 mismatch → mAP≈0).
            gt_boxes = tgt["boxes"].clone().cpu().float()
            gt_labels = tgt["labels"].clone().cpu().long()
            gts[image_id] = (gt_labels, gt_boxes)

            # per-class NMS 를 위해 [N, C] → [N*C] 로 펼침. 각 box 는 C 회 복제됨.
            scs_b = scores[b]                          # [N, C]
            bxs_b = boxes[b]                           # [N, 4]
            N_b = scs_b.shape[0]
            flat_boxes = bxs_b.unsqueeze(1).expand(N_b, C, 4).reshape(-1, 4)
            flat_scores = scs_b.reshape(-1)
            flat_classes = torch.arange(C, device=scs_b.device).unsqueeze(0).expand(N_b, C).reshape(-1)
            # score thresh 먼저 — 대부분 (>99%) 이 thresh 미만 → NMS 부담 경감
            keep_mask = flat_scores >= score_thresh
            flat_boxes = flat_boxes[keep_mask]
            flat_scores = flat_scores[keep_mask]
            flat_classes = flat_classes[keep_mask]
            if flat_boxes.numel() == 0:
                continue
            # per-class NMS — idxs=class 라 클래스 마다 독립 NMS
            keep = batched_nms(flat_boxes, flat_scores, flat_classes, iou_threshold=nms_thresh)
            keep = keep[:max_dets_per_image]
            kept_boxes = flat_boxes[keep].cpu()
            kept_scores = flat_scores[keep].cpu()
            kept_classes = flat_classes[keep].cpu()
            for j in range(kept_boxes.shape[0]):
                preds.append((image_id, int(kept_classes[j]), float(kept_scores[j]), kept_boxes[j]))

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
        "nms_thresh": float(nms_thresh),
    }
