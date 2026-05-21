# Step 3: report-and-review

## 읽어야 할 파일

- `/CLAUDE.md`
- `/docs/DATA_CARD.md`
- step 0/1/2 산출물:
  - `runs/data-sanity-download-{ts}/manifest.json`
  - `runs/data-sanity-analyze-{ts}/stats.json`
  - `runs/data-sanity-vis-{ts}/figs/*.png`

## 작업

세 step 의 산출물을 묶어 `report.md` 한 문서로 정리한 뒤, **CP-1** 사용자 검토로 phase 를 멈춘다.

### 1. 진입점

`datasets/coco/report.py` 신설 (루트 평탄, 짧은 generator).

CLI:
```bash
python -m datasets.coco.report --seed 42
# 가장 최근 runs/data-sanity-download-* / analyze-* / vis-* 를 자동 모아 묶음
# → runs/data-sanity-report-{YYYYMMDD-HHmm}/report.md
```

`--download-run`, `--analyze-run`, `--vis-run` 인자로 특정 run 지정 가능 (재현용).

### 2. report.md 섹션 구성

```markdown
# Data Sanity Report — COCO 2017 val

- **생성일**: ...
- **데이터셋**: COCO 2017
- **분할**: val
- **시드**: 42
- **참조 phase**: `phases/data-sanity-coco/`

## 다운로드 결과
{manifest.json 의 핵심 표 — files, sizes, sha256 prefix 12자, integrity_ok}

## 분포 통계
{stats.json 의 핵심 표 — num_images, num_annotations, num_classes,
 num_images_with_ann, num_images_no_ann}

### 박스 통계
{bbox_area_stats, bbox_aspect_ratio_stats 표}

### 이미지 해상도
{image_size_stats 표}

### 클래스 분포 상위/하위 5
{class_distribution 의 top5 / bottom5 표}

## 시각화
![](../data-sanity-vis-{ts}/figs/class_dist.png)
![](../data-sanity-vis-{ts}/figs/bbox_size.png)
![](../data-sanity-vis-{ts}/figs/image_size.png)
![](../data-sanity-vis-{ts}/figs/samples.png)

## DATA_CARD 표와의 일치 여부
| 항목 | DATA_CARD | 측정 | 일치 |
|------|-----------|------|------|
| val 이미지 수 | 5,000 | {...} | ✅/❌ |
| 클래스 수 | 80 | {...} | ✅/❌ |
| (필요시 추가) | | | |

## 다음 단계
- [ ] train2017 (18GB) 다운로드 phase 추가 — P0 baseline 학습 직전
- [ ] PASCAL VOC 다운로드는 별도 phase (`data-sanity-voc`) — P3 시점
- [ ] 본 phase 의 결과로 `datasets/coco.py` 의 GT 로딩 인터페이스 확정 (P0 baseline phase 시작 자료)
```

상대 경로 (`../data-sanity-vis-{ts}/figs/*.png`) 는 같은 `runs/` 부모 안의 sibling 디렉터리 참조. GitHub web 에서도 동작.

### 3. 코드 룰

- 외부 라이브러리 없이 표준 라이브러리만 사용 (`json`, `pathlib`, `datetime`, `textwrap`).
- `seed` 인자 의무 (random 없어도).
- 산출 디렉터리: `runs/data-sanity-report-{YYYYMMDD-HHmm}/report.md`.

### 4. DATA_CARD 갱신 (본 step 마무리)

step 0 의 발견으로 DATA_CARD.md 의 `scripts/data_download.sh` 라인이 outdated — 같은 줄을 `python -m datasets.coco.download` (datasets/coco/ 모듈) 로 surgical 갱신. 본 step 의 검증 절차 4 번에서 수행.

### 5. CP-1 사용자 검토

step 3 완료 후 execute.py 가 인터랙티브 prompt 띄움 (또는 직접 진행 시 사용자에게 알림). 사용자가 검토할 항목:

- 다운로드 무결성 (integrity_ok)
- 분포 통계가 DATA_CARD 와 일치
- 시각화 4 종이 정상적으로 그려졌는지
- 다음 단계 — train2017 다운로드로 갈지 / 다른 phase 로 갈지

## Acceptance Criteria

```bash
python -m datasets.coco.report --seed 42

ls runs/data-sanity-report-*/report.md | head -1

# 자동 검증
test -f runs/data-sanity-report-*/report.md && \
grep -q '## 분포 통계' runs/data-sanity-report-*/report.md && \
grep -q '## 시각화' runs/data-sanity-report-*/report.md && \
grep -q '## 다음 단계' runs/data-sanity-report-*/report.md
```

## 검증 절차

1. AC 명령 실행.
2. report.md 의 이미지 링크 4 개가 실제 파일을 가리키는지 확인.
3. DATA_CARD 일치 여부 표가 모두 ✅ 인지 확인 (❌ 있으면 `phases/data-sanity-coco/index.json` 의 step 3 status 를 `blocked` + `blocked_reason` 으로 — 사용자 결정 필요).
4. DATA_CARD.md 의 `scripts/data_download.sh` 라인을 `python -m datasets.coco.download` (datasets/coco/ 모듈) 로 surgical Edit.
5. `phases/data-sanity-coco/index.json` 의 step 3 status — TTY 사용자 검토 통과 시 `approved`, 그렇지 않으면 `awaiting-review`.
6. **docs/PENSIEVE.md 갱신** (CLAUDE.md 규칙) — "마지막 업데이트" / "지금 어디" (data-sanity-coco 완료) / "다음 한 가지" (P0 baseline 시작 / train2017 다운로드).

## 금지사항

- **report.md 안에 결론을 임의로 추가 마라** (예: "이 분포로는 학습 어렵다"). 본 step 은 *기술적 사실의 묶음*. 해석은 사용자 검토 시 결정.
- **manifest/stats/vis 의 원본 값을 임의 가공/요약 마라** — 표 안에 그대로 옮긴다. 작은 반올림은 허용 (소수점 2 자리).
- **DATA_CARD 의 다른 라인을 동시에 수정 마라** — surgical changes 원칙. `scripts/data_download.sh` 한 줄만.
- **CP-1 prompt 의 사용자 응답을 임의로 가정하고 다음 phase 로 진행 마라.** rejected/awaiting-review 면 phase 중단.
