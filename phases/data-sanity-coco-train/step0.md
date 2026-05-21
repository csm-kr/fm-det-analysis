# Step 0: download-coco-train

## 읽어야 할 파일

- `/CLAUDE.md`
- `/docs/DATA_CARD.md`
- 직전 phase: `phases/data-sanity-coco/` (val 다운 + 분석 — 본 phase 의 train 도 동일 파이프라인)

## 작업

COCO 2017 **train2017** + annotations 다운. `datasets/coco/download.py` 의 `--split train` 옵션 사용. annotations 는 val 다운 시 이미 받은 상태일 가능성 — `wget -c` 가 skip 처리.

```bash
python -m datasets.coco.download --target coco --split train --data-root data --seed 42
# → data/coco/train2017/{*.jpg} × 118,287
# → runs/data-sanity-download-{YYYYMMDD-HHmm}/manifest.json
```

- 크기: train2017.zip **~18 GB** (네트워크 따라 30분~1시간)
- annotations_trainval2017.zip 은 이미 받은 상태라 wget -c 가 skip
- 무결성: `image_count == 118287`, `target_annotation == instances_train2017.json` 존재

## Acceptance Criteria

```bash
python -m datasets.coco.download --target coco --split train --seed 42

ls data/coco/train2017 | wc -l  # 118287
ls data/coco/annotations/instances_train2017.json

jq -e '.image_count == 118287 and .ann_files >= 4 and .integrity_ok == true' \
   runs/data-sanity-download-*/manifest.json
```

## 검증 절차

1. 위 명령 실행 (background 권장 — 다운 시간 김).
2. `runs/data-sanity-download-{ts}/manifest.json` 의 `integrity_ok == true` 확인.
3. `phases/data-sanity-coco-train/index.json` 의 step 0 status 갱신.

## 금지사항

- **detectron2 import 절대 금지.**
- **train 다운 중 분석/시각화 step 동시 진행 금지** — pycocotools 가 부분 다운 파일을 로딩 시 깨질 수 있음. 다운 완료 후 진행.
- **`data/` 디렉터리 `git add` 금지.**
