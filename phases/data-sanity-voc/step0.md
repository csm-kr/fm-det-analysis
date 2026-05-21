# Step 0: download-voc

## 읽어야 할 파일

- `/CLAUDE.md`
- `/docs/DATA_CARD.md` (VOC 행 — 5011 + 4952 + 11540)

## 작업

PASCAL VOC 2007 trainval/test + VOC 2012 trainval 다운. `datasets/voc/download.py` 는 3 tar 모두 다운 + tarfile (stdlib) 로 압축 해제. 합 ~2.9 GB.

```bash
python -m datasets.voc.download --data-root data --seed 42
# → data/voc/VOCdevkit/{VOC2007, VOC2012}/{Annotations, JPEGImages, ImageSets, ...}
# → runs/data-sanity-voc-download-{ts}/manifest.json
```

압축 해제 위치는 공식 `VOCdevkit/` 구조 — DATA_CARD 의 레이아웃과 일치(상위 정리 필요 시 후속 phase 에서).

## Acceptance Criteria

```bash
python -m datasets.voc.download --seed 42

cat data/voc/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt | wc -l  # 5011
cat data/voc/VOCdevkit/VOC2007/ImageSets/Main/test.txt | wc -l       # 4952
cat data/voc/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt | wc -l   # 11540

jq -e '.integrity_ok == true and
       .image_counts["voc07-trainval"] == 5011 and
       .image_counts["voc07-test"] == 4952 and
       .image_counts["voc12-trainval"] == 11540' \
   runs/data-sanity-voc-download-*/manifest.json
```

## 검증 절차

1. AC 실행 (background 권장 — host.robots.ox.ac.uk 서버 느린 편).
2. manifest.json 의 `integrity_ok == true` 확인.
3. step 0 status 갱신.

## 금지사항

- **detectron2 import 금지.**
- **VOC tar 를 git add 금지** (data/ gitignore).
- **`difficult=1` 박스를 학습에 사용 금지** (VOC 공식 컨벤션). 본 step 은 분석만, 학습 step 에서 강제.
