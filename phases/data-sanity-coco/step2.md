# Step 2: visualize-coco-val

## 읽어야 할 파일

- `/CLAUDE.md` (CRITICAL — detectron2 금지 / 시드 의무)
- `/docs/DATA_CARD.md` (분포 표 — 시각화 결과 sanity check 기준)
- 직전 step 산출물: `runs/data-sanity-analyze-{ts}/stats.json` + `data/coco/{val2017, annotations}/`

## 작업

step1 의 통계와 원본 이미지/어노테이션을 사용해 4 종 figure 를 생성한다.

### 1. 진입점

`datasets/coco/visualize.py` 신설 (루트 평탄).

CLI:
```bash
python -m datasets.coco.visualize --split val --data-root data \
       --stats-json runs/data-sanity-analyze-{ts}/stats.json --seed 42
# → runs/data-sanity-vis-{YYYYMMDD-HHmm}/figs/{class_dist,bbox_size,image_size,samples}.png
```

`--stats-json` 미지정 시 `runs/data-sanity-analyze-*/stats.json` 중 가장 최근을 자동 선택.

### 2. 생성할 figure 4 종

#### (a) `class_dist.png` — 80 클래스 분포 막대
- x: 클래스 이름 (COCO 80 — `coco.loadCats(coco.getCatIds())` 의 `name`)
- y: annotation 개수 (log scale)
- 정렬: 개수 내림차순
- 크기: 14 × 6 inch, dpi 120
- 가로 라벨 회전 75°, 폰트 작게

#### (b) `bbox_size.png` — 2 panel
- 왼쪽: bbox area 히스토그램 (log-x, log-y). 빈 100 개 (logspace 10² ~ 10⁶).
- 오른쪽: aspect ratio (w/h) 히스토그램 (선형 x in [0, 5], 그 이상 clip 표시). 빈 50 개.
- 두 panel 모두 step1 의 통계(p10/median/p90) 점선으로 overlay.

#### (c) `image_size.png` — 이미지 해상도 2D 히스토그램
- x: width, y: height
- bins: 50×50, 색상 log scale
- 보조선: aspect 1:1, 4:3, 16:9 (점선)

#### (d) `samples.png` — 3×3 GT bbox overlay 그리드
- 9 장 샘플 — `numpy.random.default_rng(seed).choice(image_ids, 9, replace=False)` 로 결정적 선택.
- 각 이미지에 그 이미지의 GT box 전부 오버레이 — pillow `ImageDraw.rectangle` + 클래스 이름 텍스트.
- 클래스별 색상은 80 색 HSV cycle 고정.
- 각 subplot 제목: `image_id={id} | {N} boxes`
- 큰 이미지는 max(800, 800) 으로 리사이즈 후 시각화. box 좌표도 같은 비율로.

### 3. 코드 룰

- backend: `matplotlib.use("Agg")` (헤드리스).
- pycocotools 의 `COCO` 재로딩 (또는 stats 만으로 가능한 그림은 stats 사용).
- 모든 figure 는 `plt.savefig(..., bbox_inches="tight", dpi=120)`.
- random 은 `numpy.random.default_rng(seed)` 하나만. seed 미지정 차단 (assert).

### 4. 출력

```
runs/data-sanity-vis-{ts}/
└── figs/
    ├── class_dist.png
    ├── bbox_size.png
    ├── image_size.png
    └── samples.png
```

추가로 `runs/data-sanity-vis-{ts}/vis_manifest.json` — 사용된 stats-json 경로, sample image_ids 목록, seed 기록 (재현성).

## Acceptance Criteria

```bash
python -m datasets.coco.visualize --split val --data-root data --seed 42

ls runs/data-sanity-vis-*/figs/

# 자동 검증
test -f runs/data-sanity-vis-*/figs/class_dist.png && \
test -f runs/data-sanity-vis-*/figs/bbox_size.png && \
test -f runs/data-sanity-vis-*/figs/image_size.png && \
test -f runs/data-sanity-vis-*/figs/samples.png
```

## 검증 절차

1. AC 명령 실행.
2. 4 개 png 모두 존재 + 각 파일 크기 > 10KB (빈 figure 방지) 확인.
3. `vis_manifest.json` 의 `sample_image_ids` 가 seed=42 로 결정적인지 — 한 번 더 실행 시 같은 id 나오는지 한 번 검증.
4. `phases/data-sanity-coco/index.json` 의 step 2 status 갱신.

## 금지사항

- **detectron2.utils.visualizer 사용 금지.** 이유: CRITICAL — 자체 구현 (pillow + matplotlib).
- **시드 미고정 sample 선택 금지.** 이유: 재현성. seed=42 면 항상 같은 9 장이 나와야 함.
- **figure 안에 클래스 이름 외 텍스트 (모델 prediction 등) 임의 추가 금지.** 본 step 은 GT only.
- **figure 를 화면에 띄우지 마라** (`plt.show()`). 이유: 컨테이너 headless. `savefig` 만.
