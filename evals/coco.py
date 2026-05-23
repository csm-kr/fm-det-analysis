"""COCO mAP@0.5:0.95 (pycocotools COCOeval).

산출: dict {metric_primary, mAP, AP50, AP75, APs, APm, APl, num_predictions}.
detectron2 의존성 0 (CLAUDE.md CRITICAL).
"""

from __future__ import annotations

import io
import contextlib
from typing import Iterable

import torch
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def coco_eval(model, loader, device,
              score_thresh: float = 0.05, max_dets_per_image: int = 100,
              verbose: bool = True) -> dict:
    """COCO 2017 detection mAP.

    Args:
        model: nn.Module — eval forward 가 {pred_logits [B,N,C], pred_boxes [B,N,4 xyxy in current image px]} 반환.
        loader: CocoDetection DataLoader (collate_fn 으로 (images [B,3,H,W], targets list[dict]) 산출).
        device: cuda/cpu.
    Returns:
        dict — primary metric (mAP@0.5:0.95) + per-bucket AP.
    """
    model.eval()
    coco_gt = loader.dataset.coco
    cat_ids = loader.dataset.cat_ids
    idx_to_cat = {i: c for i, c in enumerate(cat_ids)}
    num_classes = len(cat_ids)

    predictions: list[dict] = []
    for batch_idx, batch in enumerate(loader):
        images, targets = batch
        images = images.to(device, non_blocking=True)
        out = model(images)
        pred_logits = out["pred_logits"]   # [B, N, C]
        pred_boxes = out["pred_boxes"]     # [B, N, 4] xyxy in transformed-image px
        scores = pred_logits.sigmoid()
        B, N, C = scores.shape

        for b in range(B):
            tgt = targets[b]
            image_id = int(tgt["image_id"])
            orig_h, orig_w = tgt["orig_size"]
            cur_h, cur_w = tgt["size"]
            sx = orig_w / max(cur_w, 1)
            sy = orig_h / max(cur_h, 1)

            scs_b = scores[b].reshape(-1)
            k = min(max_dets_per_image, scs_b.numel())
            top_scores, top_idx = scs_b.topk(k)
            box_idx = top_idx // C
            cls_idx = top_idx % C
            boxes_b = pred_boxes[b][box_idx].clone()
            boxes_b[:, 0::2] = boxes_b[:, 0::2] * sx
            boxes_b[:, 1::2] = boxes_b[:, 1::2] * sy

            for j in range(k):
                s = float(top_scores[j])
                if s < score_thresh:
                    continue
                x1, y1, x2, y2 = boxes_b[j].tolist()
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(idx_to_cat[int(cls_idx[j])]),
                    "bbox": [x1, y1, w, h],
                    "score": s,
                })

    result = {
        "metric_primary": 0.0,
        "mAP": 0.0, "AP50": 0.0, "AP75": 0.0,
        "APs": 0.0, "APm": 0.0, "APl": 0.0,
        "num_predictions": len(predictions),
        "num_classes": int(num_classes),
        "score_thresh": float(score_thresh),
        "max_dets_per_image": int(max_dets_per_image),
    }
    if not predictions:
        return result

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval_runner = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval_runner.evaluate()
    coco_eval_runner.accumulate()
    if verbose:
        coco_eval_runner.summarize()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval_runner.summarize()

    stats = coco_eval_runner.stats.tolist()
    result.update({
        "metric_primary": float(stats[0]),
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    })
    return result
