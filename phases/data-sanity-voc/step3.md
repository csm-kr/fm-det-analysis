# Step 3: report-and-review-voc

## 읽어야 할 파일

- `/CLAUDE.md`
- step 0/1/2 산출물 (VOC)

## 작업

`datasets/voc/report.py` 로 묶음 작성.

```bash
python -m datasets.voc.report --seed 42
# → runs/data-sanity-voc-report-{ts}/report.md
```

report.md 내용:
- 다운로드 결과 (3 tar 표)
- DATA_CARD 분포 일치 표 (5011 / 4952 / 11540 + 합본 16551)
- split 별 분포 통계 (3 split — images / annotations / difficult / valid)
- 박스 통계 / 이미지 해상도 (voc07-trainval)
- 클래스 top/bottom 5 (voc07-trainval)
- 시각화 4 종 (voc07-trainval)
- 다음 단계 — evals/voc.py mAP@0.5 / datasets/voc/dataset.py

CP-1 — 사용자 검토:
- VOC 다운 무결성
- 4 분할 (07 trainval / 07 test / 12 trainval / 07+12 trainval) 모두 DATA_CARD 일치
- difficult 박스 비율 합리적인지

## Acceptance Criteria

```bash
python -m datasets.voc.report --seed 42

test -f runs/data-sanity-voc-report-*/report.md && \
grep -q 'PASCAL VOC' runs/data-sanity-voc-report-*/report.md && \
grep -q '## split 별 분포 통계' runs/data-sanity-voc-report-*/report.md
```

## 검증 절차

1. AC 실행.
2. report.md 의 이미지 링크 4 개 존재 확인.
3. DATA_CARD 일치 표 모두 ✅.
4. step 3 status 갱신 (CP-1 → approved / awaiting-review).
5. docs/PENSIEVE.md 갱신.

## 금지사항

- **report 에 해석/결론 임의 추가 금지** — 기술적 사실 묶음.
- **CP-1 prompt 응답 임의 가정 금지.**
