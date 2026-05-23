# Step 0: evals-coco

## 읽어야 할 파일

- `/docs/EVAL_PROTOCOL.md` — 평가 프로토콜
- `/datasets/coco/dataset.py` — CocoDetection (`coco`, `cat_ids`, `cat_id_to_idx`) + build_coco_loader
- `/models/diffusiondet.py` — eval mode 출력 `{pred_logits [B,N,C], pred_boxes [B,N,4]}`

## 작업

`evals/coco.py` 작성 — `coco_eval(model, loader, device)` 함수로 pycocotools COCOeval 을 호출해 mAP@0.5:0.95 산출.

핵심 흐름:
- model.eval() 후 dataloader 순회.
- predictions = list of `{image_id, category_id, bbox: [x,y,w,h], score}` (pycocotools format).
- pred_boxes 는 transformed image 좌표 → `targets[b]["orig_size"] / size` 비율로 원본 px 로 rescale.
- score = sigmoid(pred_logits), top-K (max_dets_per_image=100) 박스 × class 조합.
- COCOeval(coco_gt, coco_dt, "bbox") → stats[0] = AP@0.5:0.95.

반환: dict — `{metric_primary, mAP, AP50, AP75, APs, APm, APl, num_predictions, ...}`. `metric_primary` 가 `eval.json` 의 표준 키 (harness §5-2).

## Acceptance Criteria

```bash
test -f evals/coco.py && python3 -c 'from evals.coco import coco_eval; print(coco_eval.__name__)'
```

## 검증 절차

1. `python3 -c 'from evals.coco import coco_eval'` 동작.
2. `phases/entrypoints-evals/index.json` step 0 status → `completed` + summary.

## 금지사항

- **detectron2 import 금지.** 이유: CLAUDE.md CRITICAL — 내부 동작을 직접 재구현하는 게 fm-det 의 존재 이유.
- **GT 의 iscrowd=1 box 를 prediction-side 에 섞지 마라.** 이유: pycocotools 가 자동 처리하므로 prediction 만 던지면 OK.
- 가중치 git add 금지.
