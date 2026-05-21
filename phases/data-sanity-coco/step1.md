# Step 1: analyze-coco-val

## 읽어야 할 파일

- `/CLAUDE.md` (CRITICAL — detectron2 금지 / 시드 의무)
- `/docs/DATA_CARD.md` (분포 표 — 비교 기준)
- `/docs/EVAL_PROTOCOL.md` (분할 정책)
- 직전 step 산출물: `runs/data-sanity-download-{ts}/manifest.json` + `data/coco/annotations/instances_val2017.json`

## 작업

pycocotools 로 `instances_val2017.json` 을 로딩하고 분포·결측·박스 통계를 산출한다.

### 1. 진입점

`datasets/coco/sanity.py` 신설 (루트 평탄).

시그니처:
```python
def analyze(split: str, data_root: Path, run_dir: Path, seed: int) -> dict:
    """COCO 의 annotation 을 분석 → stats dict 반환 (run_dir/stats.json 에 dump).
    """
```

CLI:
```bash
python -m datasets.coco.sanity --split val --data-root data --seed 42
# → runs/data-sanity-analyze-{YYYYMMDD-HHmm}/stats.json
```

### 2. 분석 항목

pycocotools 의 `COCO()` 로 로딩 후:

| 키 | 의미 | 검증 기준 |
|----|------|----------|
| `num_images` | 이미지 수 | val 이면 5000 (DATA_CARD) |
| `num_annotations` | bbox annotation 수 | > 0 |
| `num_classes` | 카테고리 수 | 80 (COCO instance) |
| `num_images_with_ann` | annotation 가진 이미지 수 | ≤ num_images |
| `num_images_no_ann` | background-only 이미지 수 | num_images - num_images_with_ann |
| `class_distribution` | { cat_id: count, ... } 80 개 키 | 모두 ≥ 0 |
| `bbox_xyxy_valid` | 모든 box 가 x1<x2, y1<y2, x1≥0, y1≥0, x2≤W, y2≤H | true |
| `class_id_valid` | 모든 ann 의 category_id 가 80 클래스 안 | true |
| `bbox_area_stats` | { mean, median, p10, p90, min, max } (절대 픽셀²) | 비어 있지 않음 |
| `bbox_aspect_ratio_stats` | { mean, median, p10, p90 } (w/h) | 비어 있지 않음 |
| `image_size_stats` | { width: {...}, height: {...} } (px) | 비어 있지 않음 |
| `seed` | 42 | 인자값 그대로 |

### 3. 코드 룰

- `from pycocotools.coco import COCO` 만 사용.
- 박스 좌표계는 COCO 원본인 `[x, y, w, h]` (xywh). xyxy 검증을 위해 `x2 = x + w`, `y2 = y + h` 로 변환 후 검사.
- `class_id_valid` 는 `coco.getCatIds()` 가 반환하는 80 개 안에 속하는지 검사.
- 분포 통계는 numpy 의 `percentile` 사용.
- random sampling 없음 — 모든 통계는 결정적. seed 는 manifest 일관성용으로만 받음.

### 4. stats.json 스키마

```json
{
  "split": "val",
  "computed_at": "ISO 8601 (UTC)",
  "data_root": "data",
  "seed": 42,
  "num_images": 5000,
  "num_annotations": 36781,
  "num_classes": 80,
  "num_images_with_ann": 4952,
  "num_images_no_ann": 48,
  "class_distribution": { "1": 10777, "2": 1918, ...80 keys },
  "bbox_xyxy_valid": true,
  "class_id_valid": true,
  "bbox_area_stats": { "mean": ..., "median": ..., "p10": ..., "p90": ..., "min": ..., "max": ... },
  "bbox_aspect_ratio_stats": { "mean": ..., "median": ..., "p10": ..., "p90": ... },
  "image_size_stats": {
    "width": { "mean": ..., "median": ..., "min": ..., "max": ... },
    "height": { "mean": ..., "median": ..., "min": ..., "max": ... }
  }
}
```

(`num_annotations`, `num_images_with_ann` 등 숫자는 실측치로 채움 — 위 값은 일반적인 reference)

## Acceptance Criteria

```bash
python -m datasets.coco.sanity --split val --data-root data --seed 42

ls runs/data-sanity-analyze-*/stats.json | head -1

# 자동 검증
jq -e '
  .num_images == 5000 and
  .num_classes == 80 and
  .bbox_xyxy_valid == true and
  .class_id_valid == true and
  (.num_annotations | type == "number" and . > 0)
' runs/data-sanity-analyze-*/stats.json
```

## 검증 절차

1. AC 명령 실행.
2. stats.json 의 `bbox_xyxy_valid` / `class_id_valid` 모두 true 확인. false 면 어떤 인덱스에서 깨졌는지 stderr 에 출력 + step error.
3. `class_distribution` 의 합계 == `num_annotations` 검증 (자체 일관성).
4. `phases/data-sanity-coco/index.json` 의 step 1 status 갱신 — `completed` + `summary` ("val: 5000장 / N ann / 80클래스 / xyxy valid").

## 금지사항

- **detectron2.data.datasets 의 register_coco_instances 사용 금지.** 이유: CRITICAL — detectron2 import 자체 금지.
- **test split (`test2017`) 사용 금지.** 이유: EVAL_PROTOCOL.md — 누설 시 모든 결과 무효.
- **annotation 을 임의로 필터링하지 마라** (예: iscrowd 제외, small box 제외 등). 이유: 평가 프로토콜 변경 금지. 본 step 은 *원본 그대로* 의 분포를 본다.
- **분석 결과를 stdout 출력으로만 남기지 마라.** 모든 결과는 `stats.json` 에 dump — 후속 step (시각화 / 리포트) 의 입력.
