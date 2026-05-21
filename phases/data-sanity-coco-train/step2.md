# Step 2: visualize-coco-train

## 읽어야 할 파일

- `/CLAUDE.md`
- 직전 step 산출물 (`runs/data-sanity-analyze-{ts}/stats.json` train)
- val 비교: `runs/data-sanity-vis-20260521-1335/figs/*.png`

## 작업

`datasets/coco/visualize.py` 를 train split 으로 실행.

```bash
python -m datasets.coco.visualize --split train --data-root data --seed 42
# → runs/data-sanity-vis-{ts}/figs/{class_dist,bbox_size,image_size,samples}.png
```

train 은 118K 장이라 `samples.png` 의 9 장 sample 도 train 풀에서 결정적 선택 (seed=42).

## Acceptance Criteria

```bash
python -m datasets.coco.visualize --split train --seed 42

test -f runs/data-sanity-vis-*/figs/class_dist.png && \
test -f runs/data-sanity-vis-*/figs/bbox_size.png && \
test -f runs/data-sanity-vis-*/figs/image_size.png && \
test -f runs/data-sanity-vis-*/figs/samples.png
```

## 검증 절차

1. AC 실행 — train 의 sample 이미지 로딩 시간이 길 수 있음 (Pillow 9 장 × 평균 600KB).
2. 4 png 모두 > 10KB.
3. step 2 status 갱신.

## 금지사항

- **detectron2 visualizer 사용 금지.**
- **시드 미고정 sample 선택 금지.**
- **figure 화면 띄우기 금지** (`plt.show()`).
