# Step 3: report-and-review-train

## 읽어야 할 파일

- `/CLAUDE.md`
- step 0/1/2 산출물 (train)
- 비교 자료: `runs/data-sanity-report-20260521-1335/report.md` (val report)

## 작업

`datasets/coco/report.py` 로 train report 작성.

```bash
python -m datasets.coco.report --seed 42
# → runs/data-sanity-report-{ts}/report.md (자동으로 가장 최근 train 산출물 묶음)
```

report.md 의 "다음 단계" 섹션은 train 완료 후 P0 baseline 학습 준비로.

CP-1 — 사용자 검토:
- train 다운 무결성
- train 분포 통계가 DATA_CARD 와 일치 (118,287 / 80)
- 시각화 4 종 정상
- val 과의 차이 (있다면 보고)

## Acceptance Criteria

```bash
python -m datasets.coco.report --seed 42

test -f runs/data-sanity-report-*/report.md && \
grep -q '## 분포 통계' runs/data-sanity-report-*/report.md && \
grep -q '## 시각화' runs/data-sanity-report-*/report.md
```

## 검증 절차

1. AC 실행.
2. report.md 의 이미지 링크 4 개가 실제 파일 가리키는지 확인.
3. step 3 status 갱신 (CP-1 → approved / awaiting-review).
4. **docs/PENSIEVE.md 갱신** — 지금 어디 (train 완료) / 다음 한 가지 (P0 baseline) / 최근 변경.

## 금지사항

- **report.md 에 해석/결론 임의 추가 금지** — 기술적 사실 묶음만.
- **manifest/stats 값 임의 가공 금지.**
- **CP-1 prompt 응답 임의 가정 금지.**
