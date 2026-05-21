# Evaluation Protocol

## 평가 데이터
- 평가 분할: [DATA_CARD.md](DATA_CARD.md) 의 `test` (hold-out)
- 학습/검증과 절대 분리. test 누설 시 모든 결과 무효.
- 외부 비교용 공개 벤치마크: {예: 없음 / Korean-Receipt-Bench v1}

## 메트릭
| 이름 | 정의 | 임계값 (출시 기준) |
|------|------|----------------|
| {품목 F1} | {품목명·수량 모두 일치 시 정답} | {≥ 0.85} |
| {금액 정확도} | {합계 ±0원} | {≥ 0.95} |
| {추론 p95} | {1장 처리 시간 95퍼센타일} | {< 1s} |

임계값 미달이면 출시 보류 — ADR 의 결정과 일치해야 한다.

## Baseline
- 비교 기준: {예: 룰베이스 OCR / 이전 모델 v1.0 / 외부 SOTA}
- 동일한 평가 분할에서 측정. baseline 도 함께 기록 의무.

## 실행 방법
```bash
python -m src.eval --run-id {YYYYMMDD-HHmm-tag} --split test
# 결과는 runs/{id}/eval.json 에 저장
```

## 통계적 유의성
- 시드 {예: 3개} 평균 ± 표준편차로 보고
- baseline 과의 차이는 {예: paired t-test / bootstrap 95% CI} 로 판정
- 오차 범위 안의 차이는 "의미 있는 개선"으로 보고하지 않는다

## 슬라이스 분석 (Sub-group)
주 메트릭 외에 다음 슬라이스도 측정해 편향을 추적:
- {예: 영수증 종류별 — 한식/카페/편의점}
- {예: 이미지 품질 — 선명/보통/흐림}
- {예: 영수증 길이 — 5줄 이하 / 5~20 / 20+ }

## 리포트
- `runs/{id}/eval.json` 생성 의무
- 주요 결과는 [MODEL_CARD.md](MODEL_CARD.md) 의 "평가 결과 요약" 표에 반영
