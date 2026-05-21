# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *이 프레임워크 / 이 추적 도구 / 이 시드 정책*을 골랐는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 평가 프로토콜은 [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md).

## 철학
{예: 재현성 최우선 / 단순한 baseline 부터 / 평가 지표 변경 금지}

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) / [DATA_CARD.md](./DATA_CARD.md) / [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) 등의 섹션을 명시적으로 가리켜야 한다.

---

## ADR-001: {프레임워크 (예: PyTorch 2.x + Lightning)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {팀 역량, 모델군, 분산 학습 필요성}
- **결정**: {프레임워크 + 버전}
- **대안**: {TensorFlow, JAX, Trainer API 등}
- **결과**: {제약/이점}

## ADR-002: {실험 추적 도구 (예: W&B)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {여러 사람의 실험 비교 필요성, 외부 SaaS 사용 가능 여부}
- **결정**: {W&B / TensorBoard / MLflow / 자체 CSV}
- **대안**: {대안 + 이유}
- **결과**: {제약/이점}

## ADR-003: {시드 / 재현성 정책}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {연구/제품 단계, 비교 신뢰도} — [EXPERIMENTS.md#시드-정책](./EXPERIMENTS.md), [EVAL_PROTOCOL.md#통계적-유의성](./EVAL_PROTOCOL.md) 참고
- **결정**: {예: 모든 학습 스크립트 --seed 인자 의무. random/numpy/torch/cuda 고정.
            데이터 버전 변경 시 data/CHANGELOG.md 한 줄 의무}
- **대안**: {베스트 에포트 / 무관심}
- **결과**: {제약/이점}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 실험 운영: [EXPERIMENTS.md](./EXPERIMENTS.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
- 모델: [MODEL_CARD.md](./MODEL_CARD.md)
- 평가: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
