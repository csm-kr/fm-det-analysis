"""VOC eval 좌표계 sanity check — model inference 없이 (CPU only).

확인:
  - tgt["boxes"] 가 어느 좌표계인가 (orig vs current/transformed)
  - tgt["orig_size"] / tgt["size"] 차이
  - voc_eval 가 prediction 을 orig 로 scale 하는 sx, sy 가 GT 에는 적용 안 됨 → mismatch
"""

from __future__ import annotations

from omegaconf import OmegaConf

from datasets.voc.dataset import build_voc_loader


def main():
    cfg = OmegaConf.load("configs/data/voc.yaml")
    cfg.batch_size = 4
    cfg.num_workers = 0
    cfg.pin_memory = False
    loader = build_voc_loader(cfg, split="eval", seed=42)

    print(f"eval_split = {cfg.eval_split}")
    print(f"eval transforms: short_sides={list(cfg.transforms.eval.short_sides)} max={cfg.transforms.eval.max_size}")
    print()

    for batch_idx, (images, targets) in enumerate(loader):
        print(f"=== batch {batch_idx}  images.shape = {tuple(images.shape)} ===")
        for b, tgt in enumerate(targets):
            orig_h, orig_w = tgt["orig_size"]
            cur_h, cur_w = tgt["size"]
            sx = orig_w / max(cur_w, 1)
            sy = orig_h / max(cur_h, 1)
            boxes = tgt["boxes"]
            if boxes.numel() == 0:
                print(f"  img[{b}] voc_id={tgt['voc_id']} no boxes")
                continue
            bx_min = boxes[:, 0].min().item()
            bx_max = boxes[:, 2].max().item()
            by_min = boxes[:, 1].min().item()
            by_max = boxes[:, 3].max().item()
            print(f"  img[{b}] voc_id={tgt['voc_id']:>10s}  "
                  f"orig=({orig_h},{orig_w})  cur=({cur_h},{cur_w})  "
                  f"sx,sy=({sx:.3f},{sy:.3f})  "
                  f"box_x∈[{bx_min:.1f},{bx_max:.1f}]  box_y∈[{by_min:.1f},{by_max:.1f}]  "
                  f"max(box_xy) vs (cur_w,cur_h)=({cur_w},{cur_h}) vs (orig_w,orig_h)=({orig_w},{orig_h})")
            # 어디 좌표계인지 추정: bx_max 가 cur_w 안인가 orig_w 안인가
            in_cur = bx_max <= cur_w + 1 and by_max <= cur_h + 1
            in_orig = bx_max <= orig_w + 1 and by_max <= orig_h + 1
            if in_cur and not in_orig:
                tag = "CUR only"
            elif in_orig and not in_cur:
                tag = "ORIG only"
            elif in_cur and in_orig:
                tag = "BOTH (orig==cur or boxes small)"
            else:
                tag = "NEITHER (!)"
            print(f"           → tgt['boxes'] frame: {tag}")
        if batch_idx >= 1:
            break

    print()
    print("결론 추정:")
    print("  - 학습/eval 둘 다 RandomResize 가 박스를 cur 좌표로 scale 함 (transforms.py:46-51).")
    print("  - voc_eval (evals/voc.py:62-64) 는 prediction 을 sx, sy 로 orig 로 scale.")
    print("  - 하지만 GT(tgt['boxes']) 는 그대로 cur 좌표 → IoU mismatch → mAP≈0.")
    print()
    print("Fix 후보:")
    print("  A) voc_eval 에서 prediction 스케일링 제거 (둘 다 cur 좌표로 비교)")
    print("  B) voc_eval 에서 GT 도 sx, sy 로 orig 좌표로 scale (VOC 전통).")
    print("  → A 가 변경 최소. mAP 의미는 동일.")


if __name__ == "__main__":
    main()
