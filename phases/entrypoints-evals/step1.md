# Step 1: evals-voc

## 읽어야 할 파일

- `/docs/EVAL_PROTOCOL.md`
- `/datasets/voc/dataset.py` — VOCDetection
- `/evals/coco.py` (참고)

## 작업

`evals/voc.py` 작성 — `voc_eval(model, loader, device, num_classes=20, iou_thresh=0.5)` 로 **VOC2007 11-point interpolation mAP@0.5** 산출.

핵심:
- predictions = list of `(image_id, class, score, box [4] xyxy in original px)`.
- GT = per image `(labels, boxes)`. Difficult flag 처리는 VOCDetection 의 drop_difficult 에 의존 (DATA_CARD 참고).
- per-class: predictions score 내림차순 정렬 → IoU≥0.5 매칭 → cumulative TP/FP → precision/recall curve → 11-point interpolation AP.
- mAP@0.5 = mean(per_class_ap).

반환: dict — `{metric_primary (=mAP50), mAP50, per_class_ap, num_predictions, ...}`.

## Acceptance Criteria

```bash
test -f evals/voc.py && python3 -c 'from evals.voc import voc_eval; print(voc_eval.__name__)'
```

## 검증 절차

1. `python3 -c 'from evals.voc import voc_eval'` 동작.
2. `phases/entrypoints-evals/index.json` step 1 status → `completed`.

## 금지사항

- **per-class AP 계산 전 GT count 0 인 클래스를 mean 에 포함시키지 마라.** 이유: VOC 표준은 GT 0 인 class 의 AP 를 0 으로 두지만 일부 구현은 skip 함 — DATA_CARD 의 컨벤션과 일치 유지.
- **11-point interpolation 을 all-point COCO-style 로 바꾸지 마라.** 이유: VOC2007 의 정통 metric. 변경 시 비교 baseline 무효.
