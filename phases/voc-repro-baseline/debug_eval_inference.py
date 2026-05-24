"""학습 중 last.pt 로 VOC eval inference + GT/pred 박스 그리기 + 좌표계 fix 후 mAP 재측정.

사용 (학습 도중 GPU 일부 사용):
  TORCH_HOME=/workspace/fm-det/.cache/torch \
  PYTHONPATH=/workspace/fm-det \
  python phases/voc-repro-baseline/debug_eval_inference.py \
      --run_dir runs/20260523-1416-voc-repro-baseline \
      --num_images 8

산출 (debug_out 폴더):
  - overlay_{idx}.png : GT(lime) + pred top-K(red, score≥thresh) overlay (cur 좌표)
  - eval_fixed.json   : 좌표계 fix 한 voc_eval 결과 (small subset)
  - eval_buggy.json   : 기존 voc_eval (orig scale pred + cur GT) 결과
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from datasets.voc.dataset import build_voc_loader
from models import build_diffusiondet

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
]


def _voc_ap_11pt(rec: np.ndarray, prec: np.ndarray) -> float:
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = rec >= t
        p = prec[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return float(ap)


@torch.no_grad()
def voc_eval_local(preds_per_img, gts_per_img, num_classes=20,
                    iou_thresh=0.5, score_thresh=0.05):
    """preds_per_img: list[(image_id, cls, score, box)], gts_per_img: dict[image_id]→(labels, boxes)."""
    per_class_ap = []
    for c in range(num_classes):
        gt_count = 0
        gt_match_pool = {}
        for image_id, (labels, gt_boxes) in gts_per_img.items():
            mask = (labels == c)
            cnt = int(mask.sum().item())
            if cnt > 0:
                gt_count += cnt
                gt_match_pool[image_id] = np.zeros(cnt, dtype=bool)
        if gt_count == 0:
            per_class_ap.append(0.0)
            continue
        cls_preds = sorted([p for p in preds_per_img if p[1] == c], key=lambda x: -x[2])
        tp = np.zeros(len(cls_preds), dtype=np.float32)
        fp = np.zeros(len(cls_preds), dtype=np.float32)
        for i, (image_id, _, _, box) in enumerate(cls_preds):
            if image_id not in gt_match_pool:
                fp[i] = 1.0
                continue
            labels, gt_boxes = gts_per_img[image_id]
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
    return float(np.mean(per_class_ap)) if per_class_ap else 0.0, per_class_ap


def _draw(img_chw: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
          pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_classes: torch.Tensor,
          mean, std, score_thresh=0.3, out_path: str = ""):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = (img_chw.detach().cpu() * s + m).clamp(0, 1)
    pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    draw = ImageDraw.Draw(pil)
    # GT (lime)
    for j in range(gt_boxes.shape[0]):
        x1, y1, x2, y2 = gt_boxes[j].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        ln = int(gt_labels[j])
        name = VOC_CLASSES[ln] if ln < len(VOC_CLASSES) else str(ln)
        draw.text((x1 + 2, max(y1 - 12, 0)), f"GT:{name}", fill="lime")
    # pred (red)
    for j in range(pred_boxes.shape[0]):
        sc = float(pred_scores[j])
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = pred_boxes[j].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        cn = int(pred_classes[j])
        name = VOC_CLASSES[cn] if cn < len(VOC_CLASSES) else str(cn)
        draw.text((x1 + 2, min(y2 + 2, pil.size[1] - 12)), f"{name}:{sc:.2f}", fill="red")
    pil.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--score_thresh_viz", type=float, default=0.3)
    ap.add_argument("--score_thresh_eval", type=float, default=0.05)
    ap.add_argument("--ckpt", default="last.pt", choices=["last.pt", "best.pt"])
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "debug_out"
    out_dir.mkdir(exist_ok=True)

    # cfg 복원
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.data.batch_size = 1     # GPU 메모리 절약 (학습 도중 동시 실행)
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt}  num_images={args.num_images}")

    # loader
    loader = build_voc_loader(cfg.data, split="eval", seed=int(cfg.seed))

    # model
    cfg.model.num_classes = cfg.data.num_classes
    model = build_diffusiondet(cfg.model).to(device)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {ckpt_path}  (epoch={state.get('epoch')} iter={state.get('iter')})")

    preds_buggy, preds_fixed = [], []
    gts = {}

    for img_idx, (images, targets) in enumerate(loader):
        if img_idx >= args.num_images:
            break
        images = images.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(images)
        scores = out["pred_logits"].sigmoid().float()  # [1, N, C]
        boxes = out["pred_boxes"].float()              # [1, N, 4] xyxy in CUR coords

        tgt = targets[0]
        image_id = int(tgt["image_id"])
        orig_h, orig_w = tgt["orig_size"]
        cur_h, cur_w = tgt["size"]
        sx = orig_w / max(cur_w, 1)
        sy = orig_h / max(cur_h, 1)

        # GT 는 cur 좌표 (transforms.py:46-51 가 scale 함)
        gt_boxes_cur = tgt["boxes"].clone().cpu().float()
        gt_labels = tgt["labels"].clone().cpu().long()
        gts[image_id] = (gt_labels, gt_boxes_cur)  # eval 비교 baseline: cur 좌표

        # top-K 추출
        scs = scores[0].reshape(-1)
        k = min(100, scs.numel())
        top_scores, top_idx = scs.topk(k)
        N, C = scores.shape[1], scores.shape[2]
        box_idx = top_idx // C
        cls_idx = top_idx % C
        boxes_cur = boxes[0][box_idx].cpu()             # CUR 좌표 (model 출력)
        boxes_orig = boxes_cur.clone()                  # ORIG 좌표 (기존 voc_eval 적용)
        boxes_orig[:, 0::2] = boxes_orig[:, 0::2] * sx
        boxes_orig[:, 1::2] = boxes_orig[:, 1::2] * sy

        for j in range(k):
            s = float(top_scores[j])
            if s < args.score_thresh_eval:
                continue
            preds_buggy.append((image_id, int(cls_idx[j]), s, boxes_orig[j]))
            preds_fixed.append((image_id, int(cls_idx[j]), s, boxes_cur[j]))

        # overlay (cur 좌표 — 이미지 텐서 좌표계와 일치)
        out_png = out_dir / f"overlay_{img_idx:02d}_{tgt['voc_id'].replace('/', '_')}.png"
        _draw(images[0], gt_boxes_cur, gt_labels,
              boxes_cur[:20], top_scores[:20].cpu(), cls_idx[:20].cpu(),
              list(cfg.data.mean), list(cfg.data.std),
              score_thresh=args.score_thresh_viz, out_path=str(out_png))
        print(f"img[{img_idx}] {tgt['voc_id']}  cur=({cur_h},{cur_w}) orig=({orig_h},{orig_w})  "
              f"top1_score={float(top_scores[0]):.3f} top1_cls={VOC_CLASSES[int(cls_idx[0])]}  "
              f"→ {out_png.name}")

    # subset eval
    mAP_buggy, ap_b = voc_eval_local(preds_buggy, gts,
                                       num_classes=int(cfg.data.num_classes),
                                       score_thresh=args.score_thresh_eval)
    mAP_fixed, ap_f = voc_eval_local(preds_fixed, gts,
                                       num_classes=int(cfg.data.num_classes),
                                       score_thresh=args.score_thresh_eval)
    print()
    print(f"=== {args.num_images} 장 subset 결과 ===")
    print(f"  buggy  (pred ORIG, gt CUR)  mAP@0.5 = {mAP_buggy:.4f}   ← evals/voc.py 현재 동작")
    print(f"  fixed  (pred CUR,  gt CUR)  mAP@0.5 = {mAP_fixed:.4f}   ← 좌표계 일치")
    print(f"  ratio  fixed / buggy = {(mAP_fixed / max(mAP_buggy, 1e-12)):.1f}×")
    (out_dir / "eval_buggy.json").write_text(json.dumps(
        {"mAP50": mAP_buggy, "per_class_ap": ap_b}, indent=2))
    (out_dir / "eval_fixed.json").write_text(json.dumps(
        {"mAP50": mAP_fixed, "per_class_ap": ap_f}, indent=2))


if __name__ == "__main__":
    main()
