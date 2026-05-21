# Step 1: analyze-coco-train

## 읽어야 할 파일

- `/CLAUDE.md`
- `/docs/DATA_CARD.md`
- 직전 step 산출물 (`data/coco/annotations/instances_train2017.json`)
- 비교 자료: `runs/data-sanity-analyze-20260521-1335/stats.json` (val 분포)

## 작업

`datasets/coco/sanity.py` 를 train split 으로 실행. 코드는 val 과 동일 — split 인자만 변경.

```bash
python -m datasets.coco.sanity --split train --data-root data --seed 42
# → runs/data-sanity-analyze-{ts}/stats.json
```

기대치 (DATA_CARD): `num_images == 118287` / `num_classes == 80` / `bbox_xyxy_valid == true`.

## Acceptance Criteria

```bash
python -m datasets.coco.sanity --split train --seed 42

jq -e '
  .num_images == 118287 and
  .num_classes == 80 and
  .bbox_xyxy_valid == true and
  .class_id_valid == true
' runs/data-sanity-analyze-*/stats.json
```

## 검증 절차

1. AC 실행 (train 은 약 30~60초 소요 — annotation 만 로딩).
2. stats.json 의 `num_images` 와 `num_annotations` 가 DATA_CARD 와 일치.
3. step 1 status 갱신.

## 금지사항

- **detectron2 import 절대 금지.**
- **annotation 임의 필터링 금지** (val 과 같은 룰).
- **val stats.json 을 덮어쓰지 마라** — 새 `runs/data-sanity-analyze-{ts}/` 디렉터리.
