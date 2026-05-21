# Step 2: visualize-voc

## 읽어야 할 파일

- `/CLAUDE.md`
- 직전 step 산출물 (`runs/data-sanity-voc-analyze-{ts}/stats.json`)

## 작업

`datasets/voc/visualize.py` 로 4 figure 생성. default split = `voc07-trainval` (가장 표준 baseline 분할).

```bash
python -m datasets.voc.visualize --split voc07-trainval --data-root data --seed 42
# → runs/data-sanity-voc-vis-{ts}/figs/{class_dist,bbox_size,image_size,samples}.png
```

다른 split 도 보고 싶으면 추가 명령 (예: `--split voc12-trainval`). 본 step 의 기본 출력은 voc07-trainval.

## Acceptance Criteria

```bash
python -m datasets.voc.visualize --split voc07-trainval --seed 42

test -f runs/data-sanity-voc-vis-*/figs/class_dist.png && \
test -f runs/data-sanity-voc-vis-*/figs/bbox_size.png && \
test -f runs/data-sanity-voc-vis-*/figs/image_size.png && \
test -f runs/data-sanity-voc-vis-*/figs/samples.png
```

## 검증 절차

1. AC 실행.
2. 4 png 모두 > 10KB.
3. samples 의 image_id 가 seed=42 로 결정적인지 — vis_manifest.json 확인.
4. step 2 status 갱신.

## 금지사항

- **detectron2 visualizer 금지.**
- **시드 미고정 sample 금지.**
- **figure 화면 띄우기 금지** (`plt.show()`).
