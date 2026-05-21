# Step 1: analyze-voc

## 읽어야 할 파일

- `/CLAUDE.md`
- `/docs/DATA_CARD.md` (VOC 분포 표)
- 직전 step 산출물 (`data/voc/VOCdevkit/`)

## 작업

`datasets/voc/sanity.py` 가 VOC 의 3 split (voc07-trainval, voc07-test, voc12-trainval) 모두 한 번에 XML 파싱 + 통계.

```bash
python -m datasets.voc.sanity --data-root data --seed 42
# → runs/data-sanity-voc-analyze-{ts}/stats.json
```

기대치:
- `num_classes == 20`
- 각 split 의 `bbox_xyxy_valid == true`, `class_id_valid == true`
- VOC 합본 trainval = 16,551 (DATA_CARD)
- `difficult=1` 박스 수 별도 카운트 (학습 step 에서 제외 정책 자료)

## Acceptance Criteria

```bash
python -m datasets.voc.sanity --seed 42

jq -e '
  .num_classes == 20 and
  .bbox_xyxy_valid == true and
  .class_id_valid == true and
  .splits["voc07-trainval"].num_images == 5011 and
  .splits["voc07-test"].num_images == 4952 and
  .splits["voc12-trainval"].num_images == 11540
' runs/data-sanity-voc-analyze-*/stats.json
```

## 검증 절차

1. AC 실행 — XML 파싱이라 1-2분.
2. stats.json 의 split 별 num_annotations 확인.
3. `class_distribution` 합계 == `num_annotations` 자체 일관성.
4. step 1 status 갱신.

## 금지사항

- **detectron2 / pycocotools 사용 금지.** VOC 는 XML 파싱 (`xml.etree`).
- **annotation 임의 필터링 금지** — 본 step 은 원본 그대로 분포. difficult 박스도 카운트는 하되 분리 표시.
- **test split 의 어노테이션을 학습에 사용 금지** (EVAL_PROTOCOL.md — 본 step 은 분석만).
